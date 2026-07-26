// Copyright 2024 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expresso or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>

#include "VulkanDispatch.h"
#include "gfxstream/ThreadAnnotations.h"

namespace gfxstream {
namespace vk {

class DeviceOpTracker;
using DeviceOpTrackerPtr = std::shared_ptr<DeviceOpTracker>;

using DeviceOpWaitable = std::shared_future<void>;

inline bool IsDone(const DeviceOpWaitable& waitable) {
    return waitable.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

enum class DeviceOpStatus { kPending, kDone, kFailure };

// Helper class to track the completion of host operations for a specific VkDevice.
class DeviceOpTracker {
   public:
    DeviceOpTracker(VkDevice device, VulkanDispatch* deviceDispatch);
    ~DeviceOpTracker();

    DeviceOpTracker(const DeviceOpTracker& rhs) = delete;
    DeviceOpTracker& operator=(const DeviceOpTracker& rhs) = delete;

    DeviceOpTracker(DeviceOpTracker&& rhs) = delete;
    DeviceOpTracker& operator=(DeviceOpTracker&& rhs) = delete;

    // Transfers ownership of the fence to this helper and marks that the given fence
    // can be destroyed once the waitable has finished.
    void AddPendingGarbage(DeviceOpWaitable waitable, VkFence fence);

    // Transfers ownership of the semaphore to this helper and marks that the given
    // semaphore can be destroyed once the waitable has finished.
    void AddPendingGarbage(DeviceOpWaitable waitable, VkSemaphore semaphore);

    // Checks for completion of previously submitted waitables and destroys dependent
    // objects.
    void PollAndProcessGarbage();

    // Completes the tracked operations submitted with these fences, for a caller that has already
    // established the fences are signalled. Asks the driver nothing.
    //
    // This exists because a waitable's promise could previously only be fulfilled from inside a
    // sweep, so a caller that knew perfectly well the work was done still had to pay for a walk of
    // the whole queue -- hundreds of microseconds per entry, since vkGetFenceStatus is expensive on
    // this driver -- to have that fact recorded. Measured on Minecraft, that was 3.7ms per
    // vkResetFences and about a quarter of all host dispatch, spent re-discovering completions the
    // caller had confirmed 5us earlier.
    void CompleteOpsForSignalledFences(const VkFence* fences, uint32_t fenceCount);

    void OnDestroyDevice();

   private:
    VkDevice mDevice = VK_NULL_HANDLE;
    VulkanDispatch* mDeviceDispatch = nullptr;

    friend class DeviceOpBuilder;

    // A pending operation is a fence plus what to do when it signals, rather than an opaque
    // polling closure. Keeping the fence visible is what lets a sweep ask about every queued
    // operation in one driver call instead of one call each -- see PollAndProcessGarbage().
    struct PendingOp {
        VkFence fence;  // VK_NULL_HANDLE for an operation that is complete on arrival
        std::shared_ptr<std::promise<void>> promise;
        bool destroyFenceOnCompletion;
        std::chrono::time_point<std::chrono::system_clock> timepoint;
    };
    void AddPendingDeviceOp(VkFence fence, std::shared_ptr<std::promise<void>> promise,
                            bool destroyFenceOnCompletion);
    // Completes op (fulfilling its promise, destroying its fence if owned) if its fence has
    // signalled. Returns whether it is still pending.
    DeviceOpStatus PollOne(const PendingOp& op, bool assumeSignalled);
    std::mutex mPollFunctionsMutex;
    std::deque<PendingOp> mPollFunctions GUARDED_BY(mPollFunctionsMutex);

    // Reclaiming completed operations means asking the driver whether they are done, and
    // vkGetFenceStatus is not the cheap non-blocking read its name suggests: on this device's
    // driver a single call averages milliseconds. Doing that from a guest-facing call -- the
    // submit path used to -- puts a multi-millisecond driver query in the middle of every
    // submit, and the guest ends up spinning in the transport waiting for it. Sweep from a
    // thread of our own instead, so the cost never lands on a path the guest is waiting on.
    void StartPollThreadIfNeeded();
    void StopPollThread();
    std::once_flag mPollThreadOnce;
    std::thread mPollThread;
    std::mutex mPollThreadMutex;
    std::condition_variable mPollThreadCv;
    std::atomic<bool> mPollThreadStopping{false};


    struct PendingGarbage {
        DeviceOpWaitable waitable;
        std::variant<VkFence, VkSemaphore> obj;
        std::chrono::time_point<std::chrono::system_clock> timepoint;
    };
    std::mutex mPendingGarbageMutex;
    std::deque<PendingGarbage> mPendingGarbage GUARDED_BY(mPendingGarbageMutex);
};

class DeviceOpBuilder {
   public:
    DeviceOpBuilder(DeviceOpTracker& tracker);

    DeviceOpBuilder(const DeviceOpBuilder& rhs) = delete;
    DeviceOpBuilder& operator=(const DeviceOpBuilder& rhs) = delete;

    DeviceOpBuilder(DeviceOpBuilder&& rhs) = delete;
    DeviceOpBuilder& operator=(DeviceOpBuilder&& rhs) = delete;

    ~DeviceOpBuilder();

    // Returns a VkFence that can be used to track resource usage for
    // host ops if a VkFence is not already readily available. This
    // DeviceOpBuilder and its underlying DeviceOpTracker maintain
    // ownership of the VkFence and will destroy it when then host op
    // has completed.
    VkFence CreateFenceForOp();

    // Returns a waitable that can be used to check whether a host op
    // has completed.
    DeviceOpWaitable OnQueueSubmittedWithFence(VkFence fence);

   private:
    DeviceOpTracker& mTracker;

    std::optional<VkFence> mCreatedFence;
    std::optional<VkFence> mSubmittedFence;
};

}  // namespace vk
}  // namespace gfxstream