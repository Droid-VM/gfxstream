// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright DroidVM contributors
// Additional permissions apply; see ADDITIONAL-PERMISSIONS in the repository root.
//
// This file previously carried "Copyright 2026 Google LLC". That was boilerplate copied
// from a neighbouring file: it does not exist in any upstream gfxstream ref and its whole
// history is DroidVM. Removed as a misattribution, not as a relicensing of Google work.

#include "DecoderProfile.h"

#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <pthread.h>
#include <unistd.h>
#endif

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
constexpr uint32_t kNewSpan = 4096;   // dense prefix: the generated table's sequential opcodes
constexpr uint32_t kHashSpan = 4096;  // fold hash-assigned opcodes (added commands) into here
constexpr uint32_t kSlots = kOldSpan + kNewSpan + kHashSpan;

// Where an opcode's time sits, not just its mean and worst. A mean of 78us can be 78us every time,
// or 25us nine times out of ten and milliseconds on the tenth -- the guest feels those very
// differently, and only the second kind stalls the ring behind it.
constexpr uint64_t kHistEdgesUs[6] = {25, 50, 100, 200, 400, 800};
constexpr size_t kHistBuckets = 7;

struct Slot {
    // The opcode that landed here. Folded slots cannot be inverted from the index, and an opcode
    // reported as OP_UNKNOWN_API_CALL is useless -- the two biggest entries in the first run after
    // the fold was added were both unnamed.
    std::atomic<uint32_t> opcode{0};
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> nanos{0};
    std::atomic<uint64_t> maxNanos{0};
    std::atomic<uint64_t> hist[kHistBuckets];
};

struct Profile {
    Slot slots[kSlots];
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> batchPackets{0};
    std::atomic<uint64_t> batchNanos{0};
    std::atomic<uint64_t> maxBatchPackets{0};
    // Wall vs CPU for the flush residue, split by whether that flush hit the slow mode. The two
    // populations are what matter: if the slow ones show wall >> cpu they were waiting, not working.
    std::atomic<uint64_t> flushFastWall{0}, flushFastCpu{0}, flushFastN{0};
    std::atomic<uint64_t> flushFastCmds{0}, flushSlowCmds{0};
    std::atomic<uint64_t> flushSlowWall{0}, flushSlowCpu{0}, flushSlowN{0};
    std::atomic<uint64_t> flushPhaseNanos[(size_t)FlushPhase::kCount];
    std::atomic<uint64_t> flushPhaseCount[(size_t)FlushPhase::kCount];
    std::atomic<uint64_t> subBatches{0};
    std::atomic<uint64_t> subBatchPackets{0};
    std::atomic<uint64_t> subBatchNanos{0};
    // Which decoder thread this belongs to. gfxstream names each render thread after the guest
    // process that owns its context, so this reads as "Xwayland", "gnome-shell", "Render thread".
    std::string name;
};

// One profile per decoder thread, not one for the whole host.
//
// A single shared table sums every context together, and on a desktop that is three unrelated
// workloads at once: the game, Xwayland moving its output, and the compositor redrawing the screen
// once per game frame. Per-thread CPU showed those as ~75% + ~50% + ~50%, so roughly half of what
// the merged table attributed to "the game" was the desktop. Any per-frame figure derived from it
// -- render passes per frame, draws per frame -- was wrong by that much.
//
// Threads register on first use and are never unregistered: a render thread that exits while the
// dump is walking the list would otherwise have to be reference-counted, and the leak is bounded
// by kMaxProfiles for a facility that is off unless GFXSTREAM_DECODER_PROFILE is set.
// High enough that no thread doing real work lands in the overflow bucket. Bringing up a GNOME
// session creates a render thread per context -- gnome-shell alone has nine -- so a cap of 24 put
// Minecraft, started last, in with everything else and cost it its name.
constexpr size_t kMaxProfiles = 96;
std::mutex gProfilesMu;
std::vector<Profile*> gProfiles;
Profile* gOverflow = nullptr;  // everything past kMaxProfiles, lumped together
std::atomic<uint64_t> gTicks{0};
std::chrono::steady_clock::time_point gLastDump{std::chrono::steady_clock::now()};
double gDumpSec = 10.0;

bool profilingEnabled() {
    static const bool on = [] {
        const char* env = getenv("GFXSTREAM_DECODER_PROFILE");
        if (!env || env[0] == '0') return false;
        if (const char* sec = getenv("GFXSTREAM_DECODER_PROFILE_SEC")) {
            const double v = atof(sec);
            if (v > 0.0) gDumpSec = v;
        }
        return true;
    }();
    return on;
}

Profile* profile() {
    if (!profilingEnabled()) return nullptr;
    thread_local Profile* mine = [] {
        std::lock_guard<std::mutex> lock(gProfilesMu);
        if (gProfiles.size() >= kMaxProfiles) {
            if (!gOverflow) {
                gOverflow = new Profile();
                gOverflow->name = "(overflow)";
            }
            return gOverflow;
        }
        auto* fresh = new Profile();
        char nm[32] = {};
#ifdef __linux__
        pthread_getname_np(pthread_self(), nm, sizeof(nm));
#endif
        // Carry the tid: several render threads share a guest process name (six are called
        // "Xwayland", nine "gnome-shell"), so the name alone cannot be joined against per-thread
        // CPU -- and that join is the whole point, since a thread's CPU and its dispatch time
        // differ by however long it spent spinning for the guest.
        fresh->name = nm[0] ? nm : "?";
#ifdef __linux__
        fresh->name += "/" + std::to_string(gettid());
#endif
        gProfiles.push_back(fresh);
        return fresh;
    }();
    return mine;
}

// CPU time consumed by the calling thread, as opposed to wall clock. The gap between the two is
// time the thread was not running: descheduled, or blocked inside the kernel or the driver. No
// amount of tuning the code in between closes that gap.
uint64_t threadCpuNanos() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

uint64_t nowNanos() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// -1 when the opcode is outside both ranges (a corrupt packet, worth not indexing on).
//
// The new range is not densely packed: codegen assigns hash-derived values to commands added to
// vk_gfxstream.xml (vkResetCommandPoolAsyncGOOGLE is 283932813, not OP_vkFirst+n), which is what
// keeps existing opcodes from shifting when one is inserted. Indexing by offset therefore misses
// them entirely -- the first such opcode this profiler met simply never appeared in its output,
// and the only visible symptom was the distinct-call count dropping by one. Fold anything past the
// dense prefix into the same table; a collision would merge two opcodes' numbers, which for a
// diagnostic beats not seeing one at all.
int64_t slotFor(uint32_t opcode) {
    if (opcode >= OP_vkFirst_old && opcode < OP_vkLast_old) {
        return opcode - OP_vkFirst_old;
    }
    if (opcode >= OP_vkFirst && opcode < OP_vkLast) {
        const uint32_t off = opcode - OP_vkFirst;
        return kOldSpan + (off < kNewSpan ? off : kNewSpan + (off % kHashSpan));
    }
    return -1;
}

// Only exact for the dense ranges; a folded slot cannot be inverted, so report the fold index and
// let api_opcode_to_string() fail loudly rather than print a plausible wrong name.
uint32_t opcodeForSlot(uint32_t slot) {
    if (slot < kOldSpan) return OP_vkFirst_old + slot;
    const uint32_t off = slot - kOldSpan;
    if (off < kNewSpan) return OP_vkFirst + off;
    return OP_vkFirst + off;  // folded: not a real opcode, printed as unknown
}

void dump(Profile* p, double elapsed) {
    struct Row {
        uint32_t slot;
        uint32_t opcode;
        uint64_t count;
        uint64_t nanos;
        uint64_t maxNanos;
        uint64_t hist[kHistBuckets];
    };
    std::vector<Row> rows;
    uint64_t totalCount = 0, totalNanos = 0;
    for (uint32_t i = 0; i < kSlots; ++i) {
        // Read-and-clear, so each report covers its own window rather than a since-boot average
        // that flattens whatever just changed.
        const uint64_t c = p->slots[i].count.exchange(0, std::memory_order_relaxed);
        const uint64_t n = p->slots[i].nanos.exchange(0, std::memory_order_relaxed);
        const uint64_t m = p->slots[i].maxNanos.exchange(0, std::memory_order_relaxed);
        Row row{i, p->slots[i].opcode.load(std::memory_order_relaxed), c, n, m, {}};
        for (size_t b = 0; b < kHistBuckets; ++b) {
            row.hist[b] = p->slots[i].hist[b].exchange(0, std::memory_order_relaxed);
        }
        if (c) {
            rows.push_back(row);
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
    const uint64_t fWall = p->flushFastWall.exchange(0, std::memory_order_relaxed);
    const uint64_t fCpu = p->flushFastCpu.exchange(0, std::memory_order_relaxed);
    const uint64_t fN = p->flushFastN.exchange(0, std::memory_order_relaxed);
    const uint64_t sWall = p->flushSlowWall.exchange(0, std::memory_order_relaxed);
    const uint64_t sCpu = p->flushSlowCpu.exchange(0, std::memory_order_relaxed);
    const uint64_t sN = p->flushSlowN.exchange(0, std::memory_order_relaxed);
    const uint64_t fCmds = p->flushFastCmds.exchange(0, std::memory_order_relaxed);
    const uint64_t sCmds = p->flushSlowCmds.exchange(0, std::memory_order_relaxed);
    uint64_t phaseNs[(size_t)FlushPhase::kCount], phaseN[(size_t)FlushPhase::kCount];
    for (size_t i = 0; i < (size_t)FlushPhase::kCount; ++i) {
        phaseNs[i] = p->flushPhaseNanos[i].exchange(0, std::memory_order_relaxed);
        phaseN[i] = p->flushPhaseCount[i].exchange(0, std::memory_order_relaxed);
    }
    // Idle threads still hold a table; printing 25 zero rows for each of them buries the ones that
    // matter. 1% of one thread is the floor for being worth a block.
    if (rows.empty() || totalNanos < elapsed * 1e7) return;

    // Rank by time, not by count: the interesting opcode is the one the guest waits on, and a rare
    // call that blocks for milliseconds matters more than a frequent one that costs nanoseconds.
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.nanos > b.nanos; });

    // One log line per opcode. Accumulating them into a single message truncates: logcat caps a
    // single entry, and at 25 opcodes plus histograms only the first four survived -- which is
    // exactly the deep part of the table this was widened to reach.
    std::string byTime;
    // Deep enough to reach the structural calls -- render pass begin/end, queue submits, present
    // -- not just the per-draw traffic that dominates by count. A fixed per-frame GPU cost was
    // measured (~0.63ms, 37% of GPU time at renderDistance 12) and finding it needs the calls that
    // happen a handful of times a frame, which sit far below the top twelve.
    const size_t shown = std::min<size_t>(rows.size(), 25);
    for (size_t i = 0; i < shown; ++i) {
        const Row& r = rows[i];
        byTime.clear();  // one line, one log entry -- see above
        {
            const char* name = api_opcode_to_string(r.opcode);
            byTime += name ? name : "OP_null";
            // Commands added to vk_gfxstream.xml get hash-assigned opcodes the generated name
            // table does not cover; print the number so the entry is still identifiable.
            if (!name || !strcmp(name, "OP_UNKNOWN_API_CALL")) {
                byTime += "(" + std::to_string(r.opcode) + ")";
            }
        }
        byTime += " " + std::to_string(static_cast<uint64_t>(r.nanos / elapsed / 1000000)) + "ms/s";
        byTime += " (" + std::to_string(static_cast<uint64_t>(100.0 * r.nanos / totalNanos)) + "% of";
        byTime += " dispatch, n=" + std::to_string(static_cast<uint64_t>(r.count / elapsed)) + "/s";
        byTime += " avg=" + std::to_string(r.nanos / r.count / 1000) + "us";
        byTime += " max=" + std::to_string(r.maxNanos / 1000) + "us)";
        // Distribution for the handful that matter; a long tail here is what stalls the ring.
        if (i < 5) {
            byTime += "\n        us <25:" + std::to_string(r.hist[0]) +
                      " <50:" + std::to_string(r.hist[1]) +
                      " <100:" + std::to_string(r.hist[2]) +
                      " <200:" + std::to_string(r.hist[3]) +
                      " <400:" + std::to_string(r.hist[4]) +
                      " <800:" + std::to_string(r.hist[5]) +
                      " 800+:" + std::to_string(r.hist[6]);
        }
        GFXSTREAM_WARNING("DECODERPROF[%s][%02zu] %s", p->name.c_str(), i, byTime.c_str());
    }

    GFXSTREAM_WARNING(
        "DECODERPROF[%s] over %.1fs: %llu opcodes/s in %llu distinct calls; host dispatch "
        "%llums/s (%.1f%% of one thread). batches %llu/s, avg %llu packets each, avg %lluus each, "
        "largest %llu packets -- a synchronous call at the end of a batch waits for the rest. "
        "cmdbuf replays %llu/s, avg %llu vkCmd each, avg %lluus each. flush residue: "
        "handle-lookup %lluus/s over %llu, pool-free %lluus/s over %llu. "
        "flush wall-vs-cpu: fast n=%llu wall=%lluus cpu=%lluus cmds=%llu | slow n=%llu "
        "wall=%lluus cpu=%lluus cmds=%llu (per call).",
        p->name.c_str(), elapsed, static_cast<unsigned long long>(totalCount / elapsed),
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
        static_cast<unsigned long long>(phaseNs[0] / elapsed / 1000),
        static_cast<unsigned long long>(phaseN[0] / elapsed),
        static_cast<unsigned long long>(phaseNs[1] / elapsed / 1000),
        static_cast<unsigned long long>(phaseN[1] / elapsed),
        static_cast<unsigned long long>(fN),
        static_cast<unsigned long long>(fN ? fWall / fN / 1000 : 0),
        static_cast<unsigned long long>(fN ? fCpu / fN / 1000 : 0),
        static_cast<unsigned long long>(fN ? fCmds / fN : 0),
        static_cast<unsigned long long>(sN),
        static_cast<unsigned long long>(sN ? sWall / sN / 1000 : 0),
        static_cast<unsigned long long>(sN ? sCpu / sN / 1000 : 0),
        static_cast<unsigned long long>(sN ? sCmds / sN : 0));
}

void maybeDump(Profile*) {
    // Check the clock once every 4096 packets rather than on every one: at a few hundred thousand
    // opcodes a second, an extra steady_clock read per packet would be its own measurable cost.
    if ((gTicks.fetch_add(1, std::memory_order_relaxed) & 0xfff) != 0) return;

    const auto now = std::chrono::steady_clock::now();
    // Whichever thread trips the interval dumps every thread's table, so all of them cover the
    // same window and can be compared against each other and against per-thread CPU.
    std::lock_guard<std::mutex> lock(gProfilesMu);
    const double elapsed = std::chrono::duration<double>(now - gLastDump).count();
    if (elapsed < gDumpSec) return;
    gLastDump = now;
    for (Profile* other : gProfiles) dump(other, elapsed);
    if (gOverflow) dump(gOverflow, elapsed);
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
    s.opcode.store(opcode, std::memory_order_relaxed);
    s.count.fetch_add(1, std::memory_order_relaxed);
    s.nanos.fetch_add(dt, std::memory_order_relaxed);
    uint64_t prevMax = s.maxNanos.load(std::memory_order_relaxed);
    while (dt > prevMax &&
           !s.maxNanos.compare_exchange_weak(prevMax, dt, std::memory_order_relaxed)) {
    }
    const uint64_t us = dt / 1000;
    size_t b = 0;
    while (b < 6 && us >= kHistEdgesUs[b]) ++b;
    s.hist[b].fetch_add(1, std::memory_order_relaxed);
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

// Command count for the flush currently being timed on this thread, so the wall/cpu split can be
// reported against how much that flush was actually carrying.
thread_local uint64_t tFlushCmds = 0;

void decoderProfileFlushWallCpu(uint64_t wallNanos, uint64_t cpuNanos) {
    Profile* p = profile();
    if (!p) return;
    // 100us splits the two modes cleanly -- the measured distribution is 88% under 25us and 12%
    // between 400 and 800us, with almost nothing between.
    const bool slow = wallNanos > 100000;
    if (slow) {
        p->flushSlowWall.fetch_add(wallNanos, std::memory_order_relaxed);
        p->flushSlowCpu.fetch_add(cpuNanos, std::memory_order_relaxed);
        p->flushSlowN.fetch_add(1, std::memory_order_relaxed);
        p->flushSlowCmds.fetch_add(tFlushCmds, std::memory_order_relaxed);
    } else {
        p->flushFastWall.fetch_add(wallNanos, std::memory_order_relaxed);
        p->flushFastCpu.fetch_add(cpuNanos, std::memory_order_relaxed);
        p->flushFastN.fetch_add(1, std::memory_order_relaxed);
        p->flushFastCmds.fetch_add(tFlushCmds, std::memory_order_relaxed);
    }
    tFlushCmds = 0;
}

uint64_t decoderProfileThreadCpuNow() { return profile() ? threadCpuNanos() : 0; }

std::atomic<uint64_t> sSubmitsHandled{0};
void decoderNoteSubmitHandled() { sSubmitsHandled.fetch_add(1, std::memory_order_relaxed); }
uint64_t decoderSubmitsHandled() { return sSubmitsHandled.load(std::memory_order_relaxed); }

void decoderProfileFlushPhase(FlushPhase phase, uint64_t nanos) {
    Profile* p = profile();
    if (!p) return;
    p->flushPhaseNanos[(size_t)phase].fetch_add(nanos, std::memory_order_relaxed);
    p->flushPhaseCount[(size_t)phase].fetch_add(1, std::memory_order_relaxed);
}

void decoderProfileSubBatch(uint64_t packets, uint64_t nanos) {
    Profile* p = profile();
    if (!p || !packets) return;
    tFlushCmds = packets;
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
