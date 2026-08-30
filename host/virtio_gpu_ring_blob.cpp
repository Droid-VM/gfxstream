// Copyright (C) 2024 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "virtio_gpu_ring_blob.h"

#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "gfxstream/common/logging.h"
#include "gfxstream/virtio-gpu-gfxstream-renderer.h"

namespace gfxstream {
namespace host {

using gfxstream::base::SharedMemory;

RingBlob::RingBlob(uint32_t id,
                   uint64_t size,
                   uint64_t alignment,
                   std::variant<std::unique_ptr<AlignedMemory>,
                                std::unique_ptr<SharedMemory>,
                                std::unique_ptr<ExternalMemory>> memory) :
    mId(id), mSize(size), mAlignment(alignment), mMemory(std::move(memory)) {}

bool RingBlob::isExportable() const {
    return std::holds_alternative<std::unique_ptr<SharedMemory>>(mMemory);
}

gfxstream::base::SharedMemory::handle_type RingBlob::releaseHandle() {
    if (!isExportable()) {
        return SharedMemory::invalidHandle();
    }
    return std::get<std::unique_ptr<SharedMemory>>(mMemory)->releaseHandle();
}

gfxstream::base::SharedMemory::handle_type RingBlob::dupHandle() {
    if (!isExportable()) {
        return SharedMemory::invalidHandle();
    }
    auto& shmem = std::get<std::unique_ptr<SharedMemory>>(mMemory);
    SharedMemory::handle_type fd = shmem->getFd();
    if (fd == SharedMemory::invalidHandle()) {
        return SharedMemory::invalidHandle();
    }
#ifdef _WIN32
    // Handle duplication is not implemented here; Gunyah recycle is only used on
    // Linux/Android, so fall back to transferring ownership on Windows.
    return shmem->releaseHandle();
#else
    return ::dup(fd);
#endif
}

void* RingBlob::map() {
    if (std::holds_alternative<std::unique_ptr<AlignedMemory>>(mMemory)) {
        return std::get<std::unique_ptr<AlignedMemory>>(mMemory)->addr;
    } else if (std::holds_alternative<std::unique_ptr<ExternalMemory>>(mMemory)) {
        return std::get<std::unique_ptr<ExternalMemory>>(mMemory)->addr;
    } else {
        return std::get<std::unique_ptr<SharedMemory>>(mMemory)->get();
    }
}

/*static*/
std::unique_ptr<RingBlob> RingBlob::CreateWithShmem(uint32_t id, uint64_t size) {
    const std::string name = "gfxstream-ringblob-shmem-" + std::to_string(id);

    auto shmem = std::make_unique<SharedMemory>(name, size);
    int ret = shmem->create(0600);
    if (ret) {
        GFXSTREAM_ERROR("Failed to allocate ring blob shared memory.");
        return nullptr;
    }

    return std::unique_ptr<RingBlob>(new RingBlob(id, size, 1, std::move(shmem)));
}

/*static*/
std::unique_ptr<RingBlob> RingBlob::CreateWithHostMemory(uint32_t id, uint64_t size, uint64_t alignment) {
    auto memory = std::make_unique<AlignedMemory>(alignment, size);
    if (memory->addr == nullptr) {
        GFXSTREAM_ERROR("Failed to allocate ring blob host memory.");
        return nullptr;
    }

    return std::unique_ptr<RingBlob>(new RingBlob(id, size, alignment, std::move(memory)));
}

/*static*/
std::unique_ptr<RingBlob> RingBlob::CreateFromPool(uint32_t id, uint64_t size, void* hva,
                                                   int64_t poolOffset) {
    // `hva` is a borrowed pointer into the once-mapped pool (HostVisiblePool::hvaForOffset). The
    // guest reaches the same pages via the pool GPA; the host reads/writes the ASG ring through
    // this VA. ExternalMemory does not own the memory, so freeing the RingBlob leaves the pool
    // mapping intact (it lives for the VM's lifetime). No per-blob mmap.
    if (!hva) {
        GFXSTREAM_ERROR("RINGBLOB-POOL: null pool hva (poolOffset=%lld)", (long long)poolOffset);
        return nullptr;
    }
    auto memory = std::make_unique<ExternalMemory>(hva);
    auto blob = std::unique_ptr<RingBlob>(new RingBlob(id, size, 1, std::move(memory)));
    blob->mPoolOffset = poolOffset;
    return blob;
}

#ifdef GFXSTREAM_BUILD_WITH_SNAPSHOT_FRONTEND_SUPPORT

using gfxstream::host::snapshot::VirtioGpuRingBlobSnapshot;

std::optional<VirtioGpuRingBlobSnapshot> RingBlob::Snapshot() {
    VirtioGpuRingBlobSnapshot snapshot;

    snapshot.set_id(mId);
    snapshot.set_size(mSize);
    snapshot.set_alignment(mAlignment);
    if (std::holds_alternative<std::unique_ptr<SharedMemory>>(mMemory)) {
        snapshot.set_type(VirtioGpuRingBlobSnapshot::TYPE_SHARED_MEMORY);
    } else if (std::holds_alternative<std::unique_ptr<ExternalMemory>>(mMemory)) {
        // Borrowed memory: the bytes are in the VMM's pool, and the guest reaches them through a
        // mapping this process does not own. Restoring would hand back a private host allocation
        // under the same guest mapping -- the guest would read pages nobody writes. There is no
        // snapshot type that means "borrowed", so refuse rather than produce a bad one.
        GFXSTREAM_ERROR("Cannot snapshot a pool-resident ring blob (id %llu).",
                        (unsigned long long)mId);
        return std::nullopt;
    } else {
        snapshot.set_type(VirtioGpuRingBlobSnapshot::TYPE_HOST_MEMORY);
    }

    void* mapped = map();
    if (!mapped) {
        GFXSTREAM_ERROR("Failed to map ring blob memory for snapshot.");
        return std::nullopt;
    }
    snapshot.set_memory(mapped, mSize);

    return snapshot;
}

/*static*/ std::optional<std::unique_ptr<RingBlob>> RingBlob::Restore(
        const VirtioGpuRingBlobSnapshot& snapshot) {

    std::unique_ptr<RingBlob> resource;
    if (snapshot.type() == VirtioGpuRingBlobSnapshot::TYPE_SHARED_MEMORY) {
        resource = RingBlob::CreateWithShmem(snapshot.id(), snapshot.size());
    } else {
        resource = RingBlob::CreateWithHostMemory(snapshot.id(), snapshot.size(), snapshot.alignment());
    }
    if (!resource) {
        return std::nullopt;
    }

    void* mapped = resource->map();
    if (!mapped) {
        GFXSTREAM_ERROR("Failed to map ring blob memory for restore.");
        return std::nullopt;
    }

    std::memcpy(mapped, snapshot.memory().c_str(), snapshot.memory().size());

    return resource;
}

#endif

}  // namespace host
}  // namespace gfxstream
