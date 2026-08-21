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


// Where a sweep's time goes, and how deep the queue it walks is.
//
// The two questions that decide whether a sweep is worth avoiding: how many operations it has to
// ask the driver about, and what one of those questions costs. vkGetFenceStatus is not a cheap
// read on this driver, so the cost of a sweep is essentially the queue depth times that -- which
// means a queue that grows over time turns an occasional sweep into a multi-millisecond stall
// while every average stays where it was.
//
// GFXSTREAM_SUBMIT_TRACE=1 turns it on; off, it costs one load and a branch. Reports every 100
// sweeps, and prints the depth left behind so a queue that is growing is visible as a trend.
struct SweepProfile {
    static bool Enabled() {
        static const bool on = getenv("GFXSTREAM_SUBMIT_TRACE") != nullptr;
        return on;
    }
    static uint64_t NowNs() {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }
    static void Record(uint64_t lockNs, uint64_t pollNs, size_t polled, size_t pollLeft,
                       uint64_t garbageNs, size_t destroyed, size_t garbageLeft) {
        static std::atomic<uint64_t> sPoll{0}, sGarbage{0}, sPolled{0}, sDestroyed{0}, sCount{0};
        static std::atomic<uint64_t> sLock{0}, sPollMax{0};
        sLock += lockNs;
        if (polled) {
            uint64_t per = pollNs / polled, prev = sPollMax.load();
            while (per > prev && !sPollMax.compare_exchange_weak(prev, per)) {}
        }
        sPoll += pollNs;
        sGarbage += garbageNs;
        sPolled += polled;
        sDestroyed += destroyed;
        const uint64_t n = ++sCount;
        constexpr uint64_t kEvery = 100;
        if (n % kEvery) return;
        GFXSTREAM_WARNING(
            "SWEEPPROF n=%llu avg/sweep(us): lock-wait=%llu poll=%llu (worst single "
            "vkGetFenceStatus=%llu) over %llu ops, %d queued | destroy=%llu over %llu objs, %d "
            "queued",
            (unsigned long long)n, (unsigned long long)(sLock.exchange(0) / kEvery / 1000),
            (unsigned long long)(sPoll.exchange(0) / kEvery / 1000),
            (unsigned long long)(sPollMax.exchange(0) / 1000),
            (unsigned long long)(sPolled.exchange(0) / kEvery), (int)pollLeft,
            (unsigned long long)(sGarbage.exchange(0) / kEvery / 1000),
            (unsigned long long)(sDestroyed.exchange(0) / kEvery), (int)garbageLeft);
    }
};

void DeviceOpTracker::PollAndProcessGarbage() {
    const bool prof = SweepProfile::Enabled();
    const uint64_t profT0 = prof ? SweepProfile::NowNs() : 0;
    size_t profPolled = 0, profDestroyed = 0;
    uint64_t profPollNs = 0;

    // Take the operations to examine out of the queue before touching the driver. Polling can cost
    // milliseconds per call, and AddPendingDeviceOp -- which every queue submit runs -- needs this
    // same lock: holding it across the driver calls would only move the stall from the submit path
    // onto the lock.
    std::deque<PendingOp> examining;
    {
        std::lock_guard<std::mutex> pollFunctionsLock(mPollFunctionsMutex);
        examining.swap(mPollFunctions);
    }
    const uint64_t profLockedNs = prof ? SweepProfile::NowNs() : 0;

    {
        const auto expiry = std::chrono::system_clock::now() - kAutoDeleteTimeThreshold;
        // Stop at the first operation still in flight: entries are in submission order, so the
        // ones behind it are almost certainly pending too, and every extra probe costs a driver
        // call.
        auto firstInFlightIt = examining.begin();
        for (; firstInFlightIt != examining.end(); ++firstInFlightIt) {
            ++profPolled;
            if (PollOne(*firstInFlightIt, /* assumeSignalled */ false) != DeviceOpStatus::kPending) {
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
    if (prof) profPollNs = SweepProfile::NowNs() - profLockedNs;

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
            mPollFunctions.begin(), mPollFunctions.end(), [old](const PendingOp& pollingFunc) {
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
            ++profDestroyed;

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

        if (prof) {
            SweepProfile::Record(profLockedNs - profT0, profPollNs, profPolled,
                                 mPollFunctions.size(),
                                 SweepProfile::NowNs() - profLockedNs - profPollNs, profDestroyed,
                                 mPendingGarbage.size());
        }
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

DeviceOpStatus DeviceOpTracker::PollOne(const PendingOp& op, bool assumeSignalled) {
    if (op.fence == VK_NULL_HANDLE) {
        if (op.promise) op.promise->set_value();
        return DeviceOpStatus::kDone;
    }
    VkResult result = VK_SUCCESS;
    if (!assumeSignalled) {
        result = mDeviceDispatch->vkGetFenceStatus(mDevice, op.fence);
        if (result == VK_NOT_READY) {
            return DeviceOpStatus::kPending;
        }
    }
    if (op.destroyFenceOnCompletion) {
        mDeviceDispatch->vkDestroyFence(mDevice, op.fence, nullptr);
    }
    if (op.promise) op.promise->set_value();
    return result == VK_SUCCESS ? DeviceOpStatus::kDone : DeviceOpStatus::kFailure;
}

void DeviceOpTracker::CompleteOpsForSignalledFences(const VkFence* fences, uint32_t fenceCount) {
    if (!fenceCount) return;

    std::deque<PendingOp> completed;
    {
        std::lock_guard<std::mutex> pollFunctionsLock(mPollFunctionsMutex);
        // One pass, partitioning into survivors and matches, rather than erasing matches where
        // they sit. Erasing from the middle of a deque is itself linear, so the obvious loop is
        // quadratic in the number of matches -- which was fine under the assumption this code was
        // written with, that the queue holds single digits because a sweep ran from every submit
        // and every destroy. It no longer does: sweeping moved to a thread that runs twice a
        // second, so the queue holds whatever half a second of submissions amounts to, and this
        // runs on a per-frame path with the guest blocked on the answer.
        //
        // Order is preserved and no driver state is touched, exactly as before.
        std::deque<PendingOp> survivors;
        for (PendingOp& op : mPollFunctions) {
            const bool matches = op.fence != VK_NULL_HANDLE &&
                                 std::find(fences, fences + fenceCount, op.fence) !=
                                     fences + fenceCount;
            if (matches) {
                completed.push_back(std::move(op));
            } else {
                survivors.push_back(std::move(op));
            }
        }
        mPollFunctions.swap(survivors);
    }

    // Fulfil off the lock: a promise's continuation runs on whoever fulfils it, and holding the
    // queue lock across that would put arbitrary work under it.
    for (const PendingOp& op : completed) {
        PollOne(op, /* assumeSignalled */ true);
    }
}

void DeviceOpTracker::AddPendingDeviceOp(VkFence fence,
                                         std::shared_ptr<std::promise<void>> promise,
                                         bool destroyFenceOnCompletion) {
    {
        std::lock_guard<std::mutex> lock(mPollFunctionsMutex);
        mPollFunctions.push_back(PendingOp{fence, std::move(promise), destroyFenceOnCompletion,
                                           std::chrono::system_clock::now()});
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

    mTracker.AddPendingDeviceOp(fence, std::move(promise), destroyFenceOnCompletion);

    return future;
}

void DeviceOpBuilder::OnQueueSubmissionAborted(VkFence fence) {
    if (mCreatedFence.has_value() && fence != mCreatedFence) {
        GFXSTREAM_FATAL(
            "Invalid usage: failed to call OnQueueSubmittedWithFence() with the fence "
            "requested from CreateFenceForOp.");
    }

    mSubmittedFence = fence;

    // Destroyed here rather than queued. The submission was abandoned, so this fence was never
    // handed to the driver and nothing can be waiting on it -- and a queue entry now carries a
    // fence the sweeper would poll, which for an unsubmitted fence never signals: it would sit
    // pending until the expiry threshold dropped it, leaking the fence it was queued to free.
    if (mCreatedFence.has_value()) {
        mTracker.mDeviceDispatch->vkDestroyFence(mTracker.mDevice, fence, nullptr);
    }
}

}  // namespace vk
}  // namespace host
}  // namespace gfxstream
