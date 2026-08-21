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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "device_op_tracker.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <type_traits>

#include "gfxstream/common/logging.h"

namespace gfxstream {
namespace host {
namespace vk {
namespace {

constexpr const size_t kSizeLoggingThreshold = 200;
constexpr const auto kSizeLoggingTimeThreshold = std::chrono::seconds(1);

constexpr const auto kAutoDeleteTimeThreshold = std::chrono::seconds(5);

template <typename T>
inline constexpr bool always_false_v = false;

}  // namespace

DeviceOpTracker::DeviceOpTracker(VkDevice device, VulkanDispatch* deviceDispatch)
    : mDevice(device), mDeviceDispatch(deviceDispatch) {}

void DeviceOpTracker::AddPendingGarbage(DeviceOpWaitable waitable, VkFence fence) {
    std::lock_guard<std::mutex> lock(mPendingGarbageMutex);

    mPendingGarbage.push_back(PendingGarbage{
        .waitable = std::move(waitable),
        .obj = fence,
        .timepoint = std::chrono::system_clock::now(),
    });

    if (mPendingGarbage.size() > kSizeLoggingThreshold) {
        GFXSTREAM_WARNING("VkDevice:%p has %d pending garbage objects.", mDevice,
                          mPendingGarbage.size());
    }
}

void DeviceOpTracker::AddPendingGarbage(DeviceOpWaitable waitable, VkSemaphore semaphore) {
    std::lock_guard<std::mutex> lock(mPendingGarbageMutex);

    mPendingGarbage.push_back(PendingGarbage{
        .waitable = std::move(waitable),
        .obj = semaphore,
        .timepoint = std::chrono::system_clock::now(),
    });

    if (mPendingGarbage.size() > kSizeLoggingThreshold) {
        GFXSTREAM_WARNING("VkDevice:%p has %d pending garbage objects.", mDevice,
                          mPendingGarbage.size());
    }
}

DeviceOpTracker::~DeviceOpTracker() { StopPollThread(); }

void DeviceOpTracker::StartPollThreadIfNeeded() {
    std::call_once(mPollThreadOnce, [this] {
        mPollThread = std::thread([this] {
            // Deliberately infrequent. A pass enters the driver, and on some drivers a fence
            // status query serialises against queue submits, so polling often enough to matter
            // for latency would only move the stall into vkQueueSubmit. All this reclaims is
            // fences and semaphores whose last use has already finished, so a lazy pass is fine.
            static const auto kInterval = std::chrono::milliseconds([] {
                const char* env = getenv("GFXSTREAM_DEVICE_OP_POLL_MS");
                if (env) {
                    char* end = nullptr;
                    const unsigned long v = strtoul(env, &end, 10);
                    if (end != env) return static_cast<long>(v);
                }
                return 500L;
            }());
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mPollThreadMutex);
                    mPollThreadCv.wait_for(lock, kInterval,
                                           [this] { return mPollThreadStopping.load(); });
                    if (mPollThreadStopping.load()) {
                        return;
                    }
                }
                PollAndProcessGarbage();
            }
        });
    });
}

void DeviceOpTracker::StopPollThread() {
    {
        std::lock_guard<std::mutex> lock(mPollThreadMutex);
        mPollThreadStopping = true;
    }
    mPollThreadCv.notify_all();
    if (mPollThread.joinable()) {
        mPollThread.join();
    }
}

void DeviceOpTracker::PollAndProcessGarbage() {
    // Take the operations to examine out of the queue before touching the driver. Polling can cost
    // milliseconds per call, and AddPendingDeviceOp -- which every queue submit runs -- needs this
    // same lock: holding it across the driver calls would only move the stall from the submit path
    // onto the lock.
    std::deque<PollFunction> examining;
    {
        std::lock_guard<std::mutex> pollFunctionsLock(mPollFunctionsMutex);
        examining.swap(mPollFunctions);
    }

    {
        const auto expiry = std::chrono::system_clock::now() - kAutoDeleteTimeThreshold;
        // Stop at the first operation still in flight: entries are in submission order, so the
        // ones behind it are almost certainly pending too, and every extra probe costs a driver
        // call.
        auto firstInFlightIt = examining.begin();
        for (; firstInFlightIt != examining.end(); ++firstInFlightIt) {
            if (firstInFlightIt->func() != DeviceOpStatus::kPending) {
                continue;
            }
            // An operation that never completes must not wedge everything queued behind it.
            if (firstInFlightIt->timepoint < expiry) {
                GFXSTREAM_WARNING(
                    "VkDevice:%p had an operation pending for over %d seconds; dropping it so "
                    "later completions can be reclaimed.",
                    mDevice,
                    std::chrono::duration_cast<std::chrono::seconds>(kAutoDeleteTimeThreshold)
                        .count());
                continue;
            }
            break;
        }
        examining.erase(examining.begin(), firstInFlightIt);
    }

    std::lock_guard<std::mutex> pollFunctionsLock(mPollFunctionsMutex);
    // Anything submitted while we were polling belongs after what we put back.
    examining.insert(examining.end(), std::make_move_iterator(mPollFunctions.begin()),
                     std::make_move_iterator(mPollFunctions.end()));
    mPollFunctions.swap(examining);

    if (mPollFunctions.size() > kSizeLoggingThreshold) {
        // Only report old-enough objects to avoid reporting lots of pending waitables
        // when many requests have been done in a small amount of time.
        const auto now = std::chrono::system_clock::now();
        const auto old = now - kSizeLoggingTimeThreshold;
        size_t numOldFuncs = std::count_if(
            mPollFunctions.begin(), mPollFunctions.end(), [old](const PollFunction& pollingFunc) {
                return (pollingFunc.timepoint < old);
            });
        if (numOldFuncs > kSizeLoggingThreshold) {
            GFXSTREAM_WARNING(
                "VkDevice:%p has %d pending waitables, %d taking more than %d milliseconds.",
                mDevice, mPollFunctions.size(), numOldFuncs,
                std::chrono::duration_cast<std::chrono::milliseconds>(kSizeLoggingTimeThreshold));
        }
    }

    const auto now = std::chrono::system_clock::now();
    const auto old = now - kAutoDeleteTimeThreshold;
    {
        std::lock_guard<std::mutex> pendingGarbageLock(mPendingGarbageMutex);

        // Assuming that pending garbage is added to the queue in the roughly the order
        // they are used, encountering an unsignaled/pending waitable likely means that
        // all pending garbage after is also still pending. This might not necessarily
        // always be the case but it is a simple heuristic to try to minimize the amount
        // of work performed here as it is expected that this function will be called
        // while processing other guest vulkan functions.
        auto firstPendingIt = std::find_if(mPendingGarbage.begin(), mPendingGarbage.end(),
                                           [old](const PendingGarbage& pendingGarbage) {
                                               if (pendingGarbage.timepoint < old) {
                                                   return /*still pending=*/false;
                                               }
                                               return !IsDone(pendingGarbage.waitable);
                                           });

        for (auto it = mPendingGarbage.begin(); it != firstPendingIt; it++) {
            PendingGarbage& pendingGarbage = *it;

            if (pendingGarbage.timepoint < old) {
                const auto difference = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - pendingGarbage.timepoint);
                GFXSTREAM_WARNING(
                    "VkDevice:%p had a waitable pending for %d milliseconds. Leaking object.",
                    mDevice, difference.count());
                continue;
            }

            std::visit(
                [this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, VkFence>) {
                        mDeviceDispatch->vkDestroyFence(mDevice, arg, nullptr);
                    } else if constexpr (std::is_same_v<T, VkSemaphore>) {
                        mDeviceDispatch->vkDestroySemaphore(mDevice, arg, nullptr);
                    } else {
                        static_assert(always_false_v<T>, "non-exhaustive visitor!");
                    }
                },
                pendingGarbage.obj);
        }

        mPendingGarbage.erase(mPendingGarbage.begin(), firstPendingIt);

        if (mPendingGarbage.size() > kSizeLoggingThreshold) {
            GFXSTREAM_WARNING("VkDevice:%p has %d pending garbage objects.", mDevice,
                              mPendingGarbage.size());
        }
    }
}

void DeviceOpTracker::OnDestroyDevice() {
    mDeviceDispatch->vkDeviceWaitIdle(mDevice);

    PollAndProcessGarbage();

    {
        std::lock_guard<std::mutex> pollFunctionsLock(mPollFunctionsMutex);
        if (mPollFunctions.size()) {
            // Should not keep polling fences after the device is destroyed
            GFXSTREAM_WARNING("VkDevice:%p has %d pending polling functions.", mDevice,
                                mPollFunctions.size());
            mPollFunctions.clear();
        }
    }

    {
        std::lock_guard<std::mutex> lock(mPendingGarbageMutex);
        if (!mPendingGarbage.empty()) {
            GFXSTREAM_WARNING("VkDevice:%p has %d leaking garbage objects on destruction.", mDevice,
                              mPendingGarbage.size());
        }
    }
}

void DeviceOpTracker::AddPendingDeviceOp(std::function<DeviceOpStatus()> pollFunction) {
    {
        std::lock_guard<std::mutex> lock(mPollFunctionsMutex);
        mPollFunctions.push_back(PollFunction{
            .func = std::move(pollFunction),
            .timepoint = std::chrono::system_clock::now(),
        });
    }
    // There is something to reclaim now, so there is a reason for the sweeper to exist.
    StartPollThreadIfNeeded();
}

DeviceOpBuilder::DeviceOpBuilder(DeviceOpTracker& tracker) : mTracker(tracker) {}

DeviceOpBuilder::~DeviceOpBuilder() {
    if (!mSubmittedFence) {
        GFXSTREAM_FATAL("Invalid usage: failed to call OnQueueSubmittedWithFence().");
    }
}

VkFence DeviceOpBuilder::CreateFenceForOp() {
    const VkFenceCreateInfo fenceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    VkFence fence = VK_NULL_HANDLE;
    VkResult result = mTracker.mDeviceDispatch->vkCreateFence(mTracker.mDevice, &fenceCreateInfo,
                                                              nullptr, &fence);

    mCreatedFence = fence;
    if (result != VK_SUCCESS) {
        GFXSTREAM_ERROR("DeviceOpBuilder failed to create VkFence!");
        return VK_NULL_HANDLE;
    }
    return fence;
}

DeviceOpWaitable DeviceOpBuilder::OnQueueSubmittedWithFence(VkFence fence) {
    if (mCreatedFence.has_value() && fence != mCreatedFence) {
        GFXSTREAM_FATAL(
            "Invalid usage: failed to call OnQueueSubmittedWithFence() with the fence "
            "requested from CreateFenceForOp.");
    }
    mSubmittedFence = fence;

    const bool destroyFenceOnCompletion = mCreatedFence.has_value();

    std::shared_ptr<std::promise<void>> promise = std::make_shared<std::promise<void>>();
    DeviceOpWaitable future = promise->get_future().share();

    mTracker.AddPendingDeviceOp([device = mTracker.mDevice,
                                 deviceDispatch = mTracker.mDeviceDispatch, fence,
                                 promise = std::move(promise), destroyFenceOnCompletion] {
        if (fence == VK_NULL_HANDLE) {
            return DeviceOpStatus::kDone;
        }

        VkResult result = deviceDispatch->vkGetFenceStatus(device, fence);
        if (result == VK_NOT_READY) {
            return DeviceOpStatus::kPending;
        }

        if (destroyFenceOnCompletion) {
            deviceDispatch->vkDestroyFence(device, fence, nullptr);
        }
        promise->set_value();

        return result == VK_SUCCESS ? DeviceOpStatus::kDone : DeviceOpStatus::kFailure;
    });

    return future;
}

void DeviceOpBuilder::OnQueueSubmissionAborted(VkFence fence) {
    if (mCreatedFence.has_value() && fence != mCreatedFence) {
        GFXSTREAM_FATAL(
            "Invalid usage: failed to call OnQueueSubmittedWithFence() with the fence "
            "requested from CreateFenceForOp.");
    }

    mSubmittedFence = fence;

    // Can be destroyed immediately as it's not used
    const bool destroyFenceOnCompletion = mCreatedFence.has_value();
    mTracker.AddPendingDeviceOp([device = mTracker.mDevice,
                                 deviceDispatch = mTracker.mDeviceDispatch, fence,
                                 destroyFenceOnCompletion] {
        if (destroyFenceOnCompletion) {
            deviceDispatch->vkDestroyFence(device, fence, nullptr);
        }
        return DeviceOpStatus::kDone;
    });
}

}  // namespace vk
}  // namespace host
}  // namespace gfxstream
