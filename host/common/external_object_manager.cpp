// Copyright 2019 The Android Open Source Project
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
#include "gfxstream/host/external_object_manager.h"

#include <utility>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

namespace gfxstream {
namespace host {

void CloseBlobDescriptor(BlobDescriptorType& descriptorInfo) {
#if defined(__ANDROID__)
    if (!descriptorInfo.handle) {
        return;
    }
    switch (descriptorInfo.streamHandleType) {
        case STREAM_HANDLE_TYPE_MEM_OPAQUE_FD:
        case STREAM_HANDLE_TYPE_MEM_DMABUF:
        case STREAM_HANDLE_TYPE_MEM_SHM:
        case STREAM_HANDLE_TYPE_MEM_POOL:
            close(static_cast<int>(descriptorInfo.handle));
            break;
        case STREAM_HANDLE_TYPE_PLATFORM_AHB:
            AHardwareBuffer_release(
                reinterpret_cast<AHardwareBuffer*>(descriptorInfo.handle));
            break;
        default:
            break;
    }
    descriptorInfo.handle = 0;
#else
    (void)descriptorInfo;  // ManagedDescriptor closes itself.
#endif
}

static ExternalObjectManager* sMapping() {
    static ExternalObjectManager* s = new ExternalObjectManager;
    return s;
}

// static
ExternalObjectManager* ExternalObjectManager::get() { return sMapping(); }

void ExternalObjectManager::addMapping(uint32_t ctxId, uint64_t blobId, void* addr,
                                       uint32_t caching) {
    struct HostMemInfo info = {
        .addr = addr,
        .caching = caching,
    };

    auto key = std::make_pair(ctxId, blobId);
    std::lock_guard<std::mutex> lock(mMutex);
    mHostMemInfos.insert(std::make_pair(key, info));
}

std::optional<HostMemInfo> ExternalObjectManager::removeMapping(uint32_t ctxId, uint64_t blobId) {
    auto key = std::make_pair(ctxId, blobId);
    std::lock_guard<std::mutex> lock(mMutex);
    auto found = mHostMemInfos.find(key);
    if (found != mHostMemInfos.end()) {
        std::optional<HostMemInfo> ret = found->second;
        mHostMemInfos.erase(found);
        return ret;
    }

    return std::nullopt;
}

void ExternalObjectManager::addBlobDescriptorInfo(uint32_t ctxId, uint64_t blobId,
                                                  BlobDescriptorValueType descriptor,
                                                  uint32_t streamHandleType, uint32_t caching,
                                                  std::optional<VulkanInfo> vulkanInfoOpt,
                                                  int64_t poolOffset) {
    struct BlobDescriptorInfo info = {
        .descriptorInfo =
            {
#if defined(__ANDROID__)
                .handle = descriptor,
#else
                .descriptor = std::move(descriptor),
#endif
                .streamHandleType = streamHandleType,
            },
        .caching = caching,
        .vulkanInfoOpt = vulkanInfoOpt,
        .poolOffset = poolOffset,
    };

    auto key = std::make_pair(ctxId, blobId);
    std::lock_guard<std::mutex> lock(mMutex);
    // insert() does not overwrite, so re-exporting the same (context, blob) used to drop the
    // freshly dup'd descriptor on the floor -- and on Android nothing closed it. Close the stale
    // entry and keep the newest export.
    auto found = mBlobDescriptorInfos.find(key);
    if (found != mBlobDescriptorInfos.end()) {
        CloseBlobDescriptor(found->second.descriptorInfo);
        mBlobDescriptorInfos.erase(found);
    }
    mBlobDescriptorInfos.insert(std::make_pair(key, std::move(info)));
}

void ExternalObjectManager::removeContextBlobDescriptorInfos(uint32_t ctxId) {
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto it = mBlobDescriptorInfos.begin(); it != mBlobDescriptorInfos.end();) {
        if (it->first.first == ctxId) {
            CloseBlobDescriptor(it->second.descriptorInfo);
            it = mBlobDescriptorInfos.erase(it);
        } else {
            ++it;
        }
    }
}

std::optional<BlobDescriptorInfo> ExternalObjectManager::removeBlobDescriptorInfo(uint32_t ctxId,
                                                                                  uint64_t blobId) {
    auto key = std::make_pair(ctxId, blobId);
    std::lock_guard<std::mutex> lock(mMutex);
    auto found = mBlobDescriptorInfos.find(key);
    if (found != mBlobDescriptorInfos.end()) {
        std::optional<BlobDescriptorInfo> ret = std::move(found->second);
        mBlobDescriptorInfos.erase(found);
        return ret;
    }

    return std::nullopt;
}

void ExternalObjectManager::addSyncDescriptorInfo(uint32_t ctxId, uint64_t syncId,
                                                  ManagedDescriptor descriptor,
                                                  uint32_t streamHandleType) {
    SyncDescriptorInfo info = {
        .descriptor = std::move(descriptor),
        .streamHandleType = streamHandleType,
    };

    auto key = std::make_pair(ctxId, syncId);
    std::lock_guard<std::mutex> lock(mMutex);
    mSyncDescriptorInfos.insert(std::make_pair(key, std::move(info)));
}

std::optional<SyncDescriptorInfo> ExternalObjectManager::removeSyncDescriptorInfo(uint32_t ctxId,
                                                                                  uint64_t syncId) {
    auto key = std::make_pair(ctxId, syncId);
    std::lock_guard<std::mutex> lock(mMutex);
    auto found = mSyncDescriptorInfos.find(key);
    if (found != mSyncDescriptorInfos.end()) {
        std::optional<SyncDescriptorInfo> ret = std::move(found->second);
        mSyncDescriptorInfos.erase(found);
        return ret;
    }

    return std::nullopt;
}

}  // namespace host
}  // namespace gfxstream
