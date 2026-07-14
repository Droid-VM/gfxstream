// Copyright 2026 DroidVM
// SPDX-License-Identifier: MIT
//
// 2MB-folio backing for gfxstream host-visible memory (task #15).
//
// Every emulated-host-visible VkDeviceMemory on the udmabuf path is backed by
// a memfd. When an allocation routes here, the memfd is rounded up to a 2MB
// multiple and collapsed into order-9 folios (MADV_COLLAPSE) *before* the
// udmabuf is created (udmabuf pins the pages, which would block a later
// collapse). Effects:
//
//  * gunyah_share_66 coalesces each folio into a single 2MB mem_entry, so a
//    16MB CoherentMemory block shares as 8 entries instead of 4096.
//  * crosvm is a tracked gunyah-VM owner: the order-9 allocation is
//    intercepted and served from the gh_hugepage_reserve pool (VM reserve
//    quota, not app RAM), and the final order-9 free is reclaimed back into
//    the pool. Lifecycle is bound to the VkDeviceMemory: create = order-9
//    alloc, destroy = unshare + free.
//
// The folio must never be split (no partial unmap / hole punch of a 2MB
// chunk), or its 4K frees become invisible to the reserve pool's free hook
// and the pages are lost to the pool until the owner dies.
//
// Config comes from crosvm's --gpu sub-options (vram-limit / vram-folio-threshold /
// vram-exceed-policy), plumbed in as env vars:
//   GFXSTREAM_VRAM_LIMIT_MB           folio budget, MiB (default 1024); 0 disables folio
//                                     backing (every host-visible alloc shares 4K pages)
//   GFXSTREAM_VRAM_FOLIO_THRESHOLD_KB routing gate, KiB (default 1024 = 1MB): an allocation
//                                     >= this takes a reserved 2MB folio; smaller ones share
//                                     4K pages directly (no 2MB rounding tax). 0 => all folio.
//   GFXSTREAM_VRAM_EXCEED_POLICY      on budget/collapse failure: "fallback" (default, drop
//                                     to the 4K path) or "oom" (VK_ERROR_OUT_OF_DEVICE_MEMORY)
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#endif

namespace gfxstream {
namespace vk {

constexpr uint64_t kFolioSize = 2ULL * 1024 * 1024;

struct HostVisibleFolioConfig {
    // What to do when the folio budget is exhausted (or MADV_COLLAPSE fails).
    enum ExceedPolicy {
        kFallback = 0,  // drop the allocation to the 4K dynamic path
        kOom,           // fail with VK_ERROR_OUT_OF_DEVICE_MEMORY
    };

    uint64_t quotaBytes = 0;      // 0 => folio backing disabled (all allocs on the 4K path)
    uint64_t thresholdBytes = 0;  // allocs >= this route to folio; 0 => all
    ExceedPolicy exceedPolicy = kFallback;

    bool enabled() const { return quotaBytes > 0; }
    bool oomOnExceed() const { return exceedPolicy == kOom; }

    bool routesToFolio(uint64_t size) const { return enabled() && size >= thresholdBytes; }

    static HostVisibleFolioConfig fromEnv() {
        HostVisibleFolioConfig c;
        // vram-limit: folio budget MiB (0 disables folio backing).
        const char* limit = getenv("GFXSTREAM_VRAM_LIMIT_MB");
        c.quotaBytes = (limit ? strtoull(limit, nullptr, 0) : 1024) << 20;
        // vram-folio-threshold: min alloc KiB for a folio; smaller share 4K. 0 => all folio.
        const char* th = getenv("GFXSTREAM_VRAM_FOLIO_THRESHOLD_KB");
        long long v = th ? strtoll(th, nullptr, 0) : 1024;
        c.thresholdBytes = (v > 0) ? (static_cast<uint64_t>(v) << 10) : 0;
        // vram-exceed-policy: "oom" or "fallback" (default).
        const char* pol = getenv("GFXSTREAM_VRAM_EXCEED_POLICY");
        c.exceedPolicy = (pol && !strcmp(pol, "oom")) ? kOom : kFallback;
        return c;
    }
};

// Process-global folio budget. Charged when an allocation is routed to folio
// backing, released when the VkDeviceMemory is destroyed.
class HostVisibleFolioQuota {
   public:
    static bool tryCharge(uint64_t bytes, uint64_t cap) {
        auto& u = used();
        uint64_t cur = u.load(std::memory_order_relaxed);
        do {
            if (cur + bytes > cap) return false;
        } while (!u.compare_exchange_weak(cur, cur + bytes, std::memory_order_relaxed));
        return true;
    }

    static void release(uint64_t bytes) { used().fetch_sub(bytes, std::memory_order_relaxed); }

    static uint64_t usedBytes() { return used().load(std::memory_order_relaxed); }

   private:
    static std::atomic<uint64_t>& used() {
        static std::atomic<uint64_t> sUsed{0};
        return sUsed;
    }
};

// RAII holder for a quota charge so every early-error return between charging
// and MemoryInfo bookkeeping releases automatically. transfer() hands the
// accounting responsibility to MemoryInfo::folioBytes.
class HostVisibleFolioCharge {
   public:
    HostVisibleFolioCharge() = default;
    ~HostVisibleFolioCharge() {
        if (mBytes) HostVisibleFolioQuota::release(mBytes);
    }
    HostVisibleFolioCharge(const HostVisibleFolioCharge&) = delete;
    HostVisibleFolioCharge& operator=(const HostVisibleFolioCharge&) = delete;

    bool tryCharge(uint64_t bytes, uint64_t cap) {
        if (mBytes || !HostVisibleFolioQuota::tryCharge(bytes, cap)) return false;
        mBytes = bytes;
        return true;
    }

    uint64_t bytes() const { return mBytes; }
    uint64_t transfer() { return std::exchange(mBytes, uint64_t(0)); }

   private:
    uint64_t mBytes = 0;
};

#if defined(__linux__)
// Collapse a 2MB-multiple-sized memfd into order-9 folios. Mirrors the proven
// RingBlob recipe (VirtioGpuResource.cpp): fault the shmem in through a
// PMD-aligned temporary mapping, then MADV_COLLAPSE. Must run before any
// long-term pin of the pages (udmabuf creation, gunyah share) — extra page
// references block the collapse.
//
// Returns 0 on success, else the errno of the failing step. On failure the
// memfd simply stays 4K-backed; sharing still works with more mem_entries.
inline int CollapseMemfdToFolios(int fd, uint64_t size) {
#ifndef MADV_COLLAPSE
    constexpr int kMadvCollapse = 25;
#else
    constexpr int kMadvCollapse = MADV_COLLAPSE;
#endif
    if (fd < 0 || !size || (size & (kFolioSize - 1))) return EINVAL;

    int result = ENOMEM;
    void* rsv =
        mmap(nullptr, size + kFolioSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (rsv == MAP_FAILED) return errno;

    uintptr_t alignedAddr =
        (reinterpret_cast<uintptr_t>(rsv) + kFolioSize - 1) & ~(kFolioSize - 1);
    void* aligned = mmap(reinterpret_cast<void*>(alignedAddr), size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, fd, 0);
    if (aligned != MAP_FAILED) {
        madvise(aligned, size, MADV_HUGEPAGE);
        std::memset(aligned, 0, size);
        result = madvise(aligned, size, kMadvCollapse) ? errno : 0;
    } else {
        result = errno;
    }
    munmap(rsv, size + kFolioSize);
    return result;
}
#endif  // __linux__

}  // namespace vk
}  // namespace gfxstream
