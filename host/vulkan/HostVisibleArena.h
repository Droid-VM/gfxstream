// Copyright 2026 DroidVM
// SPDX-License-Identifier: MIT
//
// Production arena for gfxstream host-visible memory: bless a fixed, user-sized
// pool of folio-backed blocks once and recycle them, so per-BO share/unshare
// churn (the source of RM EPERM / poisoning / long-uptime crashes) goes away.
//
// This header is the config + routing decision (v1 scaffold).  The blessed
// recycle pool that consumes it is wired into the udmabuf host-visible branch
// of VkDecoderGlobalState::on_vkAllocateMemory / on_vkFreeMemory.  See
// GFXSTREAM_ARENA_PLAN.md.
//
// Config (all env, nothing hardcoded):
//   GFXSTREAM_ARENA_MB          total prealloc budget, MiB (default 1024)
//   GFXSTREAM_ARENA_THRESHOLD_KB size gate, KiB:
//     0   -> all dynamic (arena off; current per-BO behavior)   [default]
//     -1  -> all prealloc (every host-visible goes to arena;
//            arena full / won't fit -> VK_ERROR_OUT_OF_DEVICE_MEMORY)
//     >0  -> hybrid: size < threshold -> arena, else dynamic per-BO
#pragma once

#include <cstdint>
#include <cstdlib>

namespace gfxstream {
namespace vk {

struct HostVisibleArenaConfig {
    enum Mode {
        kDynamicAll = 0,  // arena off
        kHybrid = 1,      // gate by thresholdBytes
        kArenaAll = 2,    // everything -> arena, overflow -> OOM
    };

    Mode mode = kDynamicAll;
    uint64_t capBytes = 0;        // total prealloc budget
    uint64_t thresholdBytes = 0;  // hybrid gate

    bool enabled() const { return mode != kDynamicAll; }

    // Does a host-visible allocation of `size` route to the arena pool?
    bool routesToArena(uint64_t size) const {
        switch (mode) {
            case kArenaAll:
                return true;  // overflow handled by the pool (-> OOM)
            case kHybrid:
                return size < thresholdBytes;
            case kDynamicAll:
            default:
                return false;
        }
    }

    static HostVisibleArenaConfig fromEnv() {
        HostVisibleArenaConfig c;

        const char* mb = getenv("GFXSTREAM_ARENA_MB");
        uint64_t capMb = mb ? strtoull(mb, nullptr, 0) : 1024;
        c.capBytes = capMb << 20;

        const char* th = getenv("GFXSTREAM_ARENA_THRESHOLD_KB");
        long thkb = th ? strtol(th, nullptr, 0) : 0;  // default: off until validated
        if (thkb == 0) {
            c.mode = kDynamicAll;
        } else if (thkb < 0) {
            c.mode = kArenaAll;
            c.thresholdBytes = UINT64_MAX;
        } else {
            c.mode = kHybrid;
            c.thresholdBytes = static_cast<uint64_t>(thkb) << 10;
        }
        return c;
    }
};

}  // namespace vk
}  // namespace gfxstream
