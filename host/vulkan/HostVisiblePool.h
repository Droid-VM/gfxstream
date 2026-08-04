// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright DroidVM contributors
// Additional permissions apply; see ADDITIONAL-PERMISSIONS in the repository root.

//
// gfxstream host-visible PRE-ALLOC pool sub-allocator (Goal 3).
//
// In pre-alloc mode crosvm blesses ONE GpuPool memfd region (a slice of the
// guest_mem memfd) before GH_VM_START: it is SHARE'd into the protected guest's
// stage-2 and folio-backed once, at boot. Instead of creating a fresh memfd per
// host-visible VkDeviceMemory (which then needs a runtime SHARE + guest accept),
// gfxstream sub-allocates from this pool. The backing pages are always pool
// pages, so the guest's stage-2 mapping is permanently valid — zero runtime
// parcels, zero per-blob hypercalls.
//
// A long session fragments the pool, so a single allocation may be satisfied by
// several non-contiguous chunks. The chunk list travels to the guest (map_blob
// response), where the virtio-gpu driver io_remaps each chunk into a contiguous
// userspace VA (the multi-run mmap path). The host likewise maps
// the chunks into a contiguous host VA (MAP_FIXED of the same pool fd at each
// chunk offset — legal: same fd, different offset, never touches stage-2).
//
// Env (set by crosvm build_vm when NCTX_GFX_SHMEM_POOL_MB is present):
//   GFXSTREAM_POOL_FD          the pool memfd (owned by crosvm; do not close)
//   GFXSTREAM_POOL_FD_OFFSET   byte offset of the pool slice within that memfd
//   GFXSTREAM_POOL_GPA         guest physical address of the pool base
//   GFXSTREAM_POOL_SIZE        pool size in bytes
#pragma once

#include <fcntl.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdint>
#include <cstdint>
#include <cstdio>

#include "GfxstreamDiag.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace gfxstream {
namespace vk {

// One physically-contiguous span within the pool (pool-relative byte offset).
struct HostVisiblePoolChunk {
    uint64_t offset;  // byte offset from the pool base (NOT the memfd base)
    uint64_t size;
};

class HostVisiblePool {
   public:
    // Returns the process-wide pool if crosvm launched us in pre-alloc mode
    // (all four env vars present and consistent), else nullptr. Callers treat
    // nullptr as "not in pre-alloc mode -> use the legacy fresh-memfd path".
    static HostVisiblePool* get() {
        static HostVisiblePool* sPool = fromEnv();
        return sPool;
    }

    int memfd() const { return mMemfd; }
    // Byte offset within the memfd of `chunk.offset == 0` (pool base).
    uint64_t memfdBase() const { return mMemfdBase; }
    uint64_t gpa() const { return mGpa; }
    uint64_t size() const { return mSize; }

    // DroidVM host-near-stateless pool routing: the guest owns the pool_mm allocator and picks the
    // offset; the host just needs a VA to read/write that blob's bytes. Map the WHOLE pool once and
    // hand out `base + offset` -- pure pointer arithmetic, zero per-blob mmap, zero per-blob state.
    // Returns nullptr if `offset` is out of range or the one-time mmap failed.
    void* hvaForOffset(uint64_t offset) {
        if (offset >= mSize) return nullptr;
        void* base = baseHva();
        return base ? static_cast<void*>(static_cast<char*>(base) + offset) : nullptr;
    }

    // The whole-pool host mapping, created lazily on first use and never unmapped (it lives for the
    // VM's lifetime, like the pool region itself). Shared with the guest via the pool GPA.
    void* baseHva() {
        std::call_once(mBaseOnce, [this]() {
            void* p = mmap(nullptr, mSize, PROT_READ | PROT_WRITE, MAP_SHARED, mMemfd, mMemfdBase);
            mBaseHva = (p == MAP_FAILED) ? nullptr : p;
            if (!mBaseHva) {
                fprintf(stderr, "GFXPOOL: whole-pool mmap failed fd=%d off=0x%llx size=%lluMB\n",
                        mMemfd, (unsigned long long)mMemfdBase,
                        (unsigned long long)(mSize >> 20));
            }
        });
        return mBaseHva;
    }

    // Bytes currently handed out (pool total minus the sum of free extents).
    // For VK_EXT_memory_budget reporting. Takes the same lock alloc()/free() use.
    uint64_t usedBytes() {
        std::lock_guard<std::mutex> lk(mMu);
        uint64_t freeBytes = 0;
        for (const auto& e : mFree) freeBytes += e.second;
        return mSize - freeBytes;
    }

    // Allocate `size` bytes (rounded up to `align`, which must be a power of 2
    // and a multiple of the page size). Prefers a single contiguous chunk; if
    // none is large enough it splits across the largest free chunks. Returns an
    // empty vector on OOM (no partial allocation is left behind). The returned
    // offsets are pool-relative; add memfdBase() for an mmap offset, or gpa()
    // for the guest physical address.
    std::vector<HostVisiblePoolChunk> alloc(uint64_t size, uint64_t align) {
        if (!size) return {};
        const uint64_t reqSize = size;
        const uint64_t a = align ? align : kMinAlign;
        size = roundUp(size, a);

        std::lock_guard<std::mutex> lk(mMu);

        // Fast path: a single free block covers it (aligned).
        for (auto it = mFree.begin(); it != mFree.end(); ++it) {
            uint64_t base = roundUp(it->first, a);
            uint64_t slack = base - it->first;
            if (it->second >= slack + size) {
                carve(it, base, size);
                logAllocLocked("alloc", reqSize, a, size, {{base, size}});
                return {{base, size}};
            }
        }

        // Fragmented path: greedily consume the largest free blocks until the
        // request is satisfied. Each chunk is page/align-aligned so the guest
        // can io_remap it and the host can MAP_FIXED it.
        std::vector<HostVisiblePoolChunk> chunks;
        uint64_t remaining = size;
        // Collect free blocks sorted largest-first for fewest chunks.
        std::vector<std::pair<uint64_t, uint64_t>> blocks(mFree.begin(), mFree.end());
        std::sort(blocks.begin(), blocks.end(),
                  [](const auto& x, const auto& y) { return x.second > y.second; });
        for (const auto& b : blocks) {
            if (!remaining) break;
            uint64_t base = roundUp(b.first, a);
            if (base >= b.first + b.second) continue;  // alignment ate the block
            uint64_t avail = (b.first + b.second) - base;
            uint64_t take = alignDown(std::min(avail, remaining), a);
            if (!take) continue;
            chunks.push_back({base, take});
            remaining -= take;
        }
        if (remaining) {
            // Not enough total space even fragmented -> OOM, leave pool intact.
            logAllocLocked("alloc-OOM", reqSize, a, size, {});
            return {};
        }
        // Commit: carve every chosen chunk out of the free map.
        for (const auto& c : chunks) {
            auto it = findContaining(c.offset);
            carve(it, c.offset, c.size);
        }
        logAllocLocked("alloc-split", reqSize, a, size, chunks);
        return chunks;
    }

    void free(const std::vector<HostVisiblePoolChunk>& chunks) {
        std::lock_guard<std::mutex> lk(mMu);
        for (const auto& c : chunks) {
            insertAndCoalesce(c.offset, c.size);
        }
        logAllocLocked("free", 0, 0, 0, chunks);
    }

   private:
    static constexpr uint64_t kMinAlign = 4096;

    // One line per pool transaction. This is the only view of what the pool actually holds --
    // it is what `gfx-host-mb` gets sized from, so it reports the request as it arrived
    // (`req`) separately from what the pool charged for it (`chg`): a caller that pads its
    // request is invisible in the used total otherwise. Per-blob, not per-frame; the ASG rings
    // are created once per context, so this is a handful of lines for a whole session.
    void logAllocLocked(const char* what, uint64_t reqSize, uint64_t align, uint64_t chargedSize,
                        const std::vector<HostVisiblePoolChunk>& chunks) {
        uint64_t freeBytes = 0, largest = 0;
        for (const auto& e : mFree) {
            freeBytes += e.second;
            largest = std::max(largest, e.second);
        }
        const uint64_t used = mSize - freeBytes;
        if (used > mHighWater) mHighWater = used;
        char head[64] = {0};
        if (!chunks.empty()) {
            snprintf(head, sizeof(head), " off=0x%llx",
                     (unsigned long long)chunks[0].offset);
        }
        GFXSTREAM_DIAG_PRINT(
                "GFXPOOL: %s req=%lluKB chg=%lluKB align=%lluKB nchunks=%zu%s "
                "used=%lluKB high=%lluKB free=%lluKB largest=%lluKB blocks=%zu\n",
                what, (unsigned long long)(reqSize >> 10), (unsigned long long)(chargedSize >> 10),
                (unsigned long long)(align >> 10), chunks.size(), head,
                (unsigned long long)(used >> 10), (unsigned long long)(mHighWater >> 10),
                (unsigned long long)(freeBytes >> 10), (unsigned long long)(largest >> 10),
                mFree.size());
    }

    static uint64_t roundUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
    static uint64_t alignDown(uint64_t v, uint64_t a) { return v & ~(a - 1); }

    // mFree: offset -> length, non-overlapping, coalesced.
    std::map<uint64_t, uint64_t> mFree;
    uint64_t mHighWater = 0;
    std::mutex mMu;
    int mMemfd = -1;
    uint64_t mMemfdBase = 0;
    uint64_t mGpa = 0;
    uint64_t mSize = 0;
    // Whole-pool host mapping for the guest-decided (pointer-placement) path; see baseHva().
    void* mBaseHva = nullptr;
    std::once_flag mBaseOnce;

    std::map<uint64_t, uint64_t>::iterator findContaining(uint64_t off) {
        auto it = mFree.upper_bound(off);
        if (it == mFree.begin()) return mFree.end();
        --it;
        if (off >= it->first && off < it->first + it->second) return it;
        return mFree.end();
    }

    // Remove [base, base+len) from the free block pointed to by `it` (which must
    // contain it), keeping the leading/trailing slack free.
    void carve(std::map<uint64_t, uint64_t>::iterator it, uint64_t base, uint64_t len) {
        uint64_t bStart = it->first;
        uint64_t bLen = it->second;
        mFree.erase(it);
        if (base > bStart) mFree[bStart] = base - bStart;              // leading slack
        uint64_t tailStart = base + len;
        uint64_t bEnd = bStart + bLen;
        if (tailStart < bEnd) mFree[tailStart] = bEnd - tailStart;     // trailing slack
    }

    void insertAndCoalesce(uint64_t off, uint64_t len) {
        // Merge with a left-adjacent block.
        if (!mFree.empty()) {
            auto it = mFree.upper_bound(off);
            if (it != mFree.begin()) {
                auto prev = std::prev(it);
                if (prev->first + prev->second == off) {
                    off = prev->first;
                    len += prev->second;
                    mFree.erase(prev);
                }
            }
        }
        // Merge with a right-adjacent block.
        auto next = mFree.find(off + len);
        if (next != mFree.end()) {
            len += next->second;
            mFree.erase(next);
        }
        mFree[off] = len;
    }

    static HostVisiblePool* fromEnv() {
        const char* fdS = getenv("GFXSTREAM_POOL_FD");
        const char* offS = getenv("GFXSTREAM_POOL_FD_OFFSET");
        const char* gpaS = getenv("GFXSTREAM_POOL_GPA");
        const char* szS = getenv("GFXSTREAM_POOL_SIZE");
        if (!fdS || !szS) return nullptr;  // not in pre-alloc mode
        int fd = (int)strtol(fdS, nullptr, 0);
        uint64_t sz = strtoull(szS, nullptr, 0);
        if (fd < 0 || !sz) return nullptr;
        auto* p = new HostVisiblePool();
        p->mMemfd = fd;
        p->mMemfdBase = offS ? strtoull(offS, nullptr, 0) : 0;
        p->mGpa = gpaS ? strtoull(gpaS, nullptr, 0) : 0;
        p->mSize = sz;
        // Guard: never allocate at pool offset 0 (== the pool's base GPA). The Gunyah RM
        // has a known defect with a blob landing exactly at the base of a SHARE'd window --
        // the analogous guard for the older per-blob host_visible_mm/BAR mechanism is
        // virtgpu_kms.c's host_visible_guard (2MiB reserved at the BAR base after the RM
        // rejected mem_share there with EINVAL, surfacing as repeated RESOURCE_MAP_BLOB
        // failures / "Failed to create virtgpu AddressSpaceStream"). This pool never had
        // that guard, so its very first allocation (always offset 0 on an empty free-list --
        // typically the ASG ring, the first blob any client requests) hits the same failure
        // mode every boot.
        constexpr uint64_t kBaseGuardBytes = 2ull << 20;  // 2MiB, matches the BAR guard
        if (sz > kBaseGuardBytes) {
            p->mFree[kBaseGuardBytes] = sz - kBaseGuardBytes;
        } else {
            p->mFree[0] = sz;  // pool smaller than the guard: leave unguarded rather than empty
        }
        fprintf(stderr,
                "GFXPOOL: pre-alloc pool fd=%d memfdBase=0x%llx gpa=0x%llx size=%lluMB\n",
                fd, (unsigned long long)p->mMemfdBase, (unsigned long long)p->mGpa,
                (unsigned long long)(sz >> 20));
        return p;
    }
};

}  // namespace vk
}  // namespace gfxstream
