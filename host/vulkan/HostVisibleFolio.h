// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright DroidVM contributors
// Additional permissions apply; see ADDITIONAL-PERMISSIONS in the repository root.

//
// Host-visible blob backing bridge.
//
// Folio backing (rounding a host-visible blob's shmem up to 2MB and MADV_COLLAPSE'ing it into
// order-9 folios) and the host-visible VRAM quota used to live here, in the gfxstream host. They
// now live in the crosvm GPU backend, dispatched per-hypervisor via the Vm trait: Gunyah protected
// folds each blob so its later stage-2 SHARE never splits a hyp block shared with LENT guest RAM
// (the exec-strip); KVM / gzvm / unprotected are a no-op. This keeps the GPU backend
// backend-agnostic and lets every hypervisor reuse one folio path.
//
// gfxstream now only forwards the blob's shmem fd to the VMM (crosvm) via the prepare/release
// callbacks, set from STREAM_RENDERER_PARAM_{PREPARE,RELEASE}_BLOB_BACKING_CALLBACK. `prepare` MUST
// run before gfxstream creates the udmabuf for the blob: the udmabuf pins the pages and extra
// references block the VMM's MADV_COLLAPSE.
#pragma once

#include <cstdint>

namespace gfxstream {

// Process-global host-visible blob-backing callbacks, populated by
// virtio-gpu-gfxstream-renderer.cpp from the init params.
class HostVisibleBlobBacking {
   public:
    static HostVisibleBlobBacking& get() {
        static HostVisibleBlobBacking sInstance;
        return sInstance;
    }

    void set(int64_t (*prepare)(void*, int, uint64_t), void (*release)(void*, uint64_t),
             void* userData) {
        mPrepare = prepare;
        mRelease = release;
        mUserData = userData;
    }

    bool enabled() const { return mPrepare != nullptr; }

    // Ask the VMM to back `memfd` (size `size`) per its policy. Returns the bytes charged against
    // the VMM's VRAM quota (0 => not folio-backed / no callback), or a NEGATIVE value when the VMM
    // refused the backing (--runtime-share exceed-policy=oom with the reserve exhausted): the
    // caller must then fail the allocation (VK_ERROR_OUT_OF_DEVICE_MEMORY) instead of proceeding
    // on 4K pages. Call BEFORE any udmabuf pin.
    int64_t prepare(int memfd, uint64_t size) {
        if (!mPrepare) return 0;
        return mPrepare(mUserData, memfd, size);
    }

    // Refund a charge previously returned by prepare().
    void release(uint64_t charged) {
        if (mRelease && charged) mRelease(mUserData, charged);
    }

   private:
    int64_t (*mPrepare)(void*, int, uint64_t) = nullptr;
    void (*mRelease)(void*, uint64_t) = nullptr;
    void* mUserData = nullptr;
};

}  // namespace gfxstream
