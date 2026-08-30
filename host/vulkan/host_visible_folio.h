// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright DroidVM contributors
// Additional permissions apply; see ADDITIONAL-PERMISSIONS in the repository root.

// Optional 2 MiB-folio backing for the shmem gfxstream itself allocates for emulated host-visible
// VkDeviceMemory. This is deliberately a renderer allocation policy, not part of crosvm's generic
// memory-registration path: an imported ColorBuffer/DMA-BUF already has backing chosen by its host
// driver and must be shared as-is.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace gfxstream {
namespace host {
namespace vk {

constexpr uint64_t kFolioSize = 2ULL * 1024 * 1024;

struct HostVisibleFolioConfig {
    enum ExceedPolicy {
        kFallback = 0,
        kOom,
    };

    // Zero disables folio backing. crosvm sets this only for gfxstream host-alloc with dynamic
    // VRAM enabled; an unset environment therefore retains normal upstream allocation behavior.
    uint64_t quotaBytes = 0;
    uint64_t thresholdBytes = 1024 * 1024;
    ExceedPolicy exceedPolicy = kFallback;

    bool enabled() const { return quotaBytes != 0; }
    bool oomOnExceed() const { return exceedPolicy == kOom; }
    bool routesToFolio(uint64_t size) const { return enabled() && size >= thresholdBytes; }

    static HostVisibleFolioConfig fromEnv() {
        HostVisibleFolioConfig config;
        if (const char* limit = getenv("GFXSTREAM_VRAM_LIMIT_MB")) {
            config.quotaBytes = strtoull(limit, nullptr, 0) << 20;
        }
        if (const char* threshold = getenv("GFXSTREAM_VRAM_FOLIO_THRESHOLD_KB")) {
            config.thresholdBytes = strtoull(threshold, nullptr, 0) << 10;
        }
        if (const char* policy = getenv("GFXSTREAM_VRAM_EXCEED_POLICY")) {
            if (!strcmp(policy, "oom")) config.exceedPolicy = kOom;
        }
        return config;
    }
};

class HostVisibleFolioQuota {
   public:
    static bool tryCharge(uint64_t bytes, uint64_t cap) {
        auto& current = used();
        uint64_t value = current.load(std::memory_order_relaxed);
        do {
            if (bytes > cap || value > cap - bytes) return false;
        } while (!current.compare_exchange_weak(value, value + bytes,
                                                std::memory_order_relaxed));
        return true;
    }

    static void release(uint64_t bytes) { used().fetch_sub(bytes, std::memory_order_relaxed); }
    static uint64_t usedBytes() { return used().load(std::memory_order_relaxed); }

   private:
    static std::atomic<uint64_t>& used() {
        static std::atomic<uint64_t> value{0};
        return value;
    }
};

// Holds a quota reservation across the allocation's error paths. Once VkDeviceMemory has been
// recorded, transfer() moves responsibility to MemoryInfo::folioBytes.
class HostVisibleFolioCharge {
   public:
    ~HostVisibleFolioCharge() { release(); }
    HostVisibleFolioCharge() = default;
    HostVisibleFolioCharge(const HostVisibleFolioCharge&) = delete;
    HostVisibleFolioCharge& operator=(const HostVisibleFolioCharge&) = delete;

    bool tryCharge(uint64_t bytes, uint64_t cap) {
        if (mBytes || !HostVisibleFolioQuota::tryCharge(bytes, cap)) return false;
        mBytes = bytes;
        return true;
    }

    void release() {
        if (!mBytes) return;
        HostVisibleFolioQuota::release(mBytes);
        mBytes = 0;
    }

    uint64_t bytes() const { return mBytes; }
    uint64_t transfer() { return std::exchange(mBytes, uint64_t(0)); }

   private:
    uint64_t mBytes = 0;
};

#if defined(__linux__)
// Grow the physical shmem backing to whole-folio size and collapse it before udmabuf pins the
// pages. The SharedMemory object's logical size and the Vulkan allocation remain unchanged; only
// the underlying memfd has an inaccessible rounded tail, matching gfxstream's former behavior.
inline int CollapseMemfdToFolios(int fd, uint64_t roundedSize) {
#ifndef MADV_COLLAPSE
    constexpr int kMadvCollapse = 25;
#else
    constexpr int kMadvCollapse = MADV_COLLAPSE;
#endif
    if (fd < 0 || !roundedSize || (roundedSize & (kFolioSize - 1))) return EINVAL;
    if (ftruncate(fd, static_cast<off_t>(roundedSize)) != 0) return errno;

    void* reservation = mmap(nullptr, roundedSize + kFolioSize, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reservation == MAP_FAILED) return errno;

    const uintptr_t alignedAddress =
        (reinterpret_cast<uintptr_t>(reservation) + kFolioSize - 1) & ~(kFolioSize - 1);
    void* mapping = mmap(reinterpret_cast<void*>(alignedAddress), roundedSize,
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    int result = 0;
    if (mapping == MAP_FAILED) {
        result = errno;
    } else {
        (void)madvise(mapping, roundedSize, MADV_HUGEPAGE);
        std::memset(mapping, 0, roundedSize);
        if (madvise(mapping, roundedSize, kMadvCollapse) != 0) result = errno;
    }
    munmap(reservation, roundedSize + kFolioSize);
    return result;
}
#endif

}  // namespace vk
}  // namespace host
}  // namespace gfxstream
