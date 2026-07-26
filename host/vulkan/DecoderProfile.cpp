// Copyright 2026 Google LLC
// SPDX-License-Identifier: MIT

#include "DecoderProfile.h"

#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "gfxstream/common/logging.h"
#include "goldfish_vk_marshaling.h"

namespace gfxstream {
namespace vk {
namespace {

// Opcodes come from two disjoint ranges (OP_vkFirst_old 20000..30000 and OP_vkFirst 200000000
// onward), so index by offset within whichever range the opcode falls in rather than allocating for
// the gap between them.
constexpr uint32_t kOldSpan = OP_vkLast_old - OP_vkFirst_old;
constexpr uint32_t kNewSpan = 4096;  // the generated table is a few hundred entries; leave room
constexpr uint32_t kSlots = kOldSpan + kNewSpan;

struct Slot {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> nanos{0};
    std::atomic<uint64_t> maxNanos{0};
};

struct Profile {
    Slot slots[kSlots];
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> batchPackets{0};
    std::atomic<uint64_t> batchNanos{0};
    std::atomic<uint64_t> maxBatchPackets{0};
    std::atomic<uint64_t> subBatches{0};
    std::atomic<uint64_t> subBatchPackets{0};
    std::atomic<uint64_t> subBatchNanos{0};
    std::atomic<uint64_t> ticks{0};
    std::mutex dumpMutex;
    std::chrono::steady_clock::time_point lastDump{std::chrono::steady_clock::now()};
    double dumpSec = 10.0;
};

Profile* profile() {
    static Profile* p = [] {
        const char* env = getenv("GFXSTREAM_DECODER_PROFILE");
        if (!env || env[0] == '0') return static_cast<Profile*>(nullptr);
        auto* fresh = new Profile();
        if (const char* sec = getenv("GFXSTREAM_DECODER_PROFILE_SEC")) {
            const double v = atof(sec);
            if (v > 0.0) fresh->dumpSec = v;
        }
        return fresh;
    }();
    return p;
}

uint64_t nowNanos() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// -1 when the opcode is outside both ranges (a corrupt packet, worth not indexing on).
int64_t slotFor(uint32_t opcode) {
    if (opcode >= OP_vkFirst_old && opcode < OP_vkLast_old) {
        return opcode - OP_vkFirst_old;
    }
    if (opcode >= OP_vkFirst && opcode - OP_vkFirst < kNewSpan) {
        return kOldSpan + (opcode - OP_vkFirst);
    }
    return -1;
}

uint32_t opcodeForSlot(uint32_t slot) {
    return slot < kOldSpan ? OP_vkFirst_old + slot : OP_vkFirst + (slot - kOldSpan);
}

void dump(Profile* p, double elapsed) {
    struct Row {
        uint32_t slot;
        uint64_t count;
        uint64_t nanos;
        uint64_t maxNanos;
    };
    std::vector<Row> rows;
    uint64_t totalCount = 0, totalNanos = 0;
    for (uint32_t i = 0; i < kSlots; ++i) {
        // Read-and-clear, so each report covers its own window rather than a since-boot average
        // that flattens whatever just changed.
        const uint64_t c = p->slots[i].count.exchange(0, std::memory_order_relaxed);
        const uint64_t n = p->slots[i].nanos.exchange(0, std::memory_order_relaxed);
        const uint64_t m = p->slots[i].maxNanos.exchange(0, std::memory_order_relaxed);
        if (c) {
            rows.push_back({i, c, n, m});
            totalCount += c;
            totalNanos += n;
        }
    }
    const uint64_t batches = p->batches.exchange(0, std::memory_order_relaxed);
    const uint64_t batchPackets = p->batchPackets.exchange(0, std::memory_order_relaxed);
    const uint64_t batchNanos = p->batchNanos.exchange(0, std::memory_order_relaxed);
    const uint64_t maxBatch = p->maxBatchPackets.exchange(0, std::memory_order_relaxed);
    const uint64_t subBatches = p->subBatches.exchange(0, std::memory_order_relaxed);
    const uint64_t subPackets = p->subBatchPackets.exchange(0, std::memory_order_relaxed);
    const uint64_t subNanos = p->subBatchNanos.exchange(0, std::memory_order_relaxed);
    if (rows.empty()) return;

    // Rank by time, not by count: the interesting opcode is the one the guest waits on, and a rare
    // call that blocks for milliseconds matters more than a frequent one that costs nanoseconds.
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.nanos > b.nanos; });

    std::string byTime;
    const size_t shown = std::min<size_t>(rows.size(), 12);
    for (size_t i = 0; i < shown; ++i) {
        const Row& r = rows[i];
        byTime += "\n    ";
        byTime += api_opcode_to_string(opcodeForSlot(r.slot));
        byTime += " " + std::to_string(static_cast<uint64_t>(r.nanos / elapsed / 1000000)) + "ms/s";
        byTime += " (" + std::to_string(static_cast<uint64_t>(100.0 * r.nanos / totalNanos)) + "% of";
        byTime += " dispatch, n=" + std::to_string(static_cast<uint64_t>(r.count / elapsed)) + "/s";
        byTime += " avg=" + std::to_string(r.nanos / r.count / 1000) + "us";
        byTime += " max=" + std::to_string(r.maxNanos / 1000) + "us)";
    }

    GFXSTREAM_WARNING(
        "DECODERPROF over %.1fs: %llu opcodes/s in %llu distinct calls; host dispatch "
        "%llums/s (%.1f%% of one thread). batches %llu/s, avg %llu packets each, avg %lluus each, "
        "largest %llu packets -- a synchronous call at the end of a batch waits for the rest. "
        "cmdbuf replays %llu/s, avg %llu vkCmd each, avg %lluus each.%s",
        elapsed, static_cast<unsigned long long>(totalCount / elapsed),
        static_cast<unsigned long long>(rows.size()),
        static_cast<unsigned long long>(totalNanos / elapsed / 1000000),
        100.0 * totalNanos / elapsed / 1e9,
        static_cast<unsigned long long>(batches / elapsed),
        static_cast<unsigned long long>(batches ? batchPackets / batches : 0),
        static_cast<unsigned long long>(batches ? batchNanos / batches / 1000 : 0),
        static_cast<unsigned long long>(maxBatch),
        static_cast<unsigned long long>(subBatches / elapsed),
        static_cast<unsigned long long>(subBatches ? subPackets / subBatches : 0),
        static_cast<unsigned long long>(subBatches ? subNanos / subBatches / 1000 : 0),
        byTime.c_str());
}

void maybeDump(Profile* p) {
    // Check the clock once every 4096 packets rather than on every one: at a few hundred thousand
    // opcodes a second, an extra steady_clock read per packet would be its own measurable cost.
    if ((p->ticks.fetch_add(1, std::memory_order_relaxed) & 0xfff) != 0) return;

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(p->dumpMutex);
    const double elapsed = std::chrono::duration<double>(now - p->lastDump).count();
    if (elapsed < p->dumpSec) return;
    p->lastDump = now;
    dump(p, elapsed);
}

}  // namespace

uint64_t decoderProfileNow() { return nowNanos(); }

uint64_t decoderProfileBegin() { return profile() ? nowNanos() : 0; }

// Nanoseconds attributed to opcodes decoded inside the call currently being timed on this thread.
// subDecode() runs on the same thread as the flush that invoked it, so a plain thread_local is
// enough -- no nesting deeper than one level exists.
thread_local uint64_t tNestedNanos = 0;

void record(Profile* p, uint32_t opcode, uint64_t dt) {
    const int64_t slot = slotFor(opcode);
    if (slot < 0) return;
    Slot& s = p->slots[slot];
    s.count.fetch_add(1, std::memory_order_relaxed);
    s.nanos.fetch_add(dt, std::memory_order_relaxed);
    uint64_t prevMax = s.maxNanos.load(std::memory_order_relaxed);
    while (dt > prevMax &&
           !s.maxNanos.compare_exchange_weak(prevMax, dt, std::memory_order_relaxed)) {
    }
}

void decoderProfileEndInner(uint32_t opcode, uint64_t start) {
    if (!start) return;
    Profile* p = profile();
    if (!p) return;
    const uint64_t dt = nowNanos() - start;
    record(p, opcode, dt);
    // Accumulate for the enclosing call to subtract. Siblings add up here; they must not subtract
    // from each other -- an earlier version did, and it zeroed out every vkCmd* after the first.
    tNestedNanos += dt;
    maybeDump(p);
}

void decoderProfileEnd(uint32_t opcode, uint64_t start) {
    if (!start) return;
    Profile* p = profile();
    if (!p) return;

    const uint64_t elapsed = nowNanos() - start;
    // Self time: what is left after the calls this one contained.
    const uint64_t dt = elapsed > tNestedNanos ? elapsed - tNestedNanos : 0;
    tNestedNanos = 0;
    record(p, opcode, dt);
    maybeDump(p);
}

// Clears the nesting accumulator. Call before starting a batch that is not nested inside another
// timed call, so a previous batch's total cannot leak into this one's first opcode.
void decoderProfileResetNesting() { tNestedNanos = 0; }

void decoderProfileSubBatch(uint64_t packets, uint64_t nanos) {
    Profile* p = profile();
    if (!p || !packets) return;
    p->subBatches.fetch_add(1, std::memory_order_relaxed);
    p->subBatchPackets.fetch_add(packets, std::memory_order_relaxed);
    p->subBatchNanos.fetch_add(nanos, std::memory_order_relaxed);
}

void decoderProfileBatch(uint64_t packets, uint64_t nanos) {
    Profile* p = profile();
    if (!p || !packets) return;
    p->batches.fetch_add(1, std::memory_order_relaxed);
    p->batchPackets.fetch_add(packets, std::memory_order_relaxed);
    p->batchNanos.fetch_add(nanos, std::memory_order_relaxed);
    uint64_t prevMax = p->maxBatchPackets.load(std::memory_order_relaxed);
    while (packets > prevMax &&
           !p->maxBatchPackets.compare_exchange_weak(prevMax, packets, std::memory_order_relaxed)) {
    }
}

}  // namespace vk
}  // namespace gfxstream
