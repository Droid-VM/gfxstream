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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>

#include "gfxstream/testing/TestUtils.h"

namespace gfxstream {
namespace host {
namespace vk {
namespace {

using ::testing::_;
using ::MatchesStdRegex;
using ::testing::Return;
using ::testing::Test;

class DeviceOpTrackerTest : public Test {
protected:
    class MockDispatch {
    public:
        MOCK_METHOD(VkResult, vkCreateFence,
                    (VkDevice device, const VkFenceCreateInfo* pCreateInfo,
                     const VkAllocationCallbacks* pAllocator, VkFence* pFence));
        MOCK_METHOD(void, vkDestroyFence,
                    (VkDevice device, VkFence fence,
                     const VkAllocationCallbacks* pAllocator));
        MOCK_METHOD(VkResult, vkGetFenceStatus,
                    (VkDevice device, VkFence fence));
        MOCK_METHOD(VkResult, vkDeviceWaitIdle, (VkDevice device));
    };

    inline static MockDispatch* sMockDispatch = nullptr;

    DeviceOpTrackerTest() : mDevice(reinterpret_cast<VkDevice>(0x2222'0000)) {
        sMockDispatch = &mMockDispatch;
        std::memset(&mDispatch, 0, sizeof(mDispatch));
        mDispatch.vkCreateFence = [](VkDevice device, const VkFenceCreateInfo* pCreateInfo,
                                     const VkAllocationCallbacks* pAllocator,
                                     VkFence* pFence) -> VkResult {
            return sMockDispatch ? sMockDispatch->vkCreateFence(device, pCreateInfo, pAllocator, pFence)
                                 : VK_ERROR_INITIALIZATION_FAILED;
        };
        mDispatch.vkDestroyFence = [](VkDevice device, VkFence fence,
                                      const VkAllocationCallbacks* pAllocator) {
            if (sMockDispatch) {
                sMockDispatch->vkDestroyFence(device, fence, pAllocator);
            }
        };
        mDispatch.vkGetFenceStatus = [](VkDevice device, VkFence fence) -> VkResult {
            return sMockDispatch ? sMockDispatch->vkGetFenceStatus(device, fence)
                                 : VK_ERROR_INITIALIZATION_FAILED;
        };
        mDispatch.vkDeviceWaitIdle = [](VkDevice device) -> VkResult {
            return sMockDispatch ? sMockDispatch->vkDeviceWaitIdle(device) : VK_SUCCESS;
        };

        ON_CALL(mMockDispatch, vkCreateFence(mDevice, _, _, _))
            .WillByDefault([](VkDevice, const VkFenceCreateInfo*,
                             const VkAllocationCallbacks*, VkFence* pFence) {
                if (pFence) {
                    *pFence = reinterpret_cast<VkFence>(0x1234'0000);
                }
                return VK_SUCCESS;
            });
    }

    ~DeviceOpTrackerTest() {
        sMockDispatch = nullptr;
    }

    ::testing::NiceMock<MockDispatch> mMockDispatch;
    VulkanDispatch mDispatch;
    VkDevice mDevice;
};

using DeviceOpTrackerDeathTest = DeviceOpTrackerTest;

TEST_F(DeviceOpTrackerTest, OnQueueSubmissionAbortedDestroysCreatedFence) {
    VkFence createdFence = reinterpret_cast<VkFence>(0x1234'0000);
    EXPECT_CALL(mMockDispatch, vkCreateFence(mDevice, _, _, _)).Times(1);

    DeviceOpTracker tracker(mDevice, &mDispatch);
    {
        DeviceOpBuilder builder(tracker);
        VkFence usedFence = builder.CreateFenceForOp();
        ASSERT_EQ(usedFence, createdFence);

        // Simulate submission failure (e.g. dispatchVkQueueSubmit or queuePendingSubmission failed)
        builder.OnQueueSubmissionAborted(usedFence);
    }

    // Since submission was aborted, the created fence must be destroyed when garbage is polled,
    // and vkGetFenceStatus should NOT be polled.
    EXPECT_CALL(mMockDispatch, vkGetFenceStatus(_, _)).Times(0);
    EXPECT_CALL(mMockDispatch, vkDestroyFence(mDevice, createdFence, nullptr)).Times(1);

    tracker.PollAndProcessGarbage();
}

TEST_F(DeviceOpTrackerTest, OnQueueSubmissionAbortedGuestFenceDoesNotDestroy) {
    VkFence guestFence = reinterpret_cast<VkFence>(0x5678'0000);

    DeviceOpTracker tracker(mDevice, &mDispatch);
    {
        DeviceOpBuilder builder(tracker);
        // Guest provided a fence, so CreateFenceForOp() is not called.
        // Simulate submission failure
        builder.OnQueueSubmissionAborted(guestFence);
    }

    // The guest fence is not owned by the tracker, so vkDestroyFence must not be called.
    EXPECT_CALL(mMockDispatch, vkDestroyFence(_, _, _)).Times(0);
    EXPECT_CALL(mMockDispatch, vkGetFenceStatus(_, _)).Times(0);

    tracker.PollAndProcessGarbage();
}

TEST_F(DeviceOpTrackerTest, OnQueueSubmissionAbortedNullFence) {
    DeviceOpTracker tracker(mDevice, &mDispatch);
    {
        DeviceOpBuilder builder(tracker);
        // Simulate submission failure with null fence
        builder.OnQueueSubmissionAborted(VK_NULL_HANDLE);
    }

    EXPECT_CALL(mMockDispatch, vkDestroyFence(_, _, _)).Times(0);
    EXPECT_CALL(mMockDispatch, vkGetFenceStatus(_, _)).Times(0);

    tracker.PollAndProcessGarbage();
}

TEST_F(DeviceOpTrackerTest, OnQueueSubmittedWithFencePollsAndDestroysCreatedFence) {
    VkFence createdFence = reinterpret_cast<VkFence>(0x1234'0000);
    EXPECT_CALL(mMockDispatch, vkCreateFence(mDevice, _, _, _)).Times(1);

    DeviceOpTracker tracker(mDevice, &mDispatch);
    DeviceOpWaitable waitable;
    {
        DeviceOpBuilder builder(tracker);
        VkFence usedFence = builder.CreateFenceForOp();
        waitable = builder.OnQueueSubmittedWithFence(usedFence);
    }

    // 1st poll: fence not ready yet
    EXPECT_CALL(mMockDispatch, vkGetFenceStatus(mDevice, createdFence))
        .WillOnce(Return(VK_NOT_READY));
    EXPECT_CALL(mMockDispatch, vkDestroyFence(_, _, _)).Times(0);
    tracker.PollAndProcessGarbage();
    EXPECT_FALSE(IsDone(waitable));

    // 2nd poll: fence ready -> should destroy created fence and mark waitable done
    EXPECT_CALL(mMockDispatch, vkGetFenceStatus(mDevice, createdFence))
        .WillOnce(Return(VK_SUCCESS));
    EXPECT_CALL(mMockDispatch, vkDestroyFence(mDevice, createdFence, nullptr)).Times(1);
    tracker.PollAndProcessGarbage();
    EXPECT_TRUE(IsDone(waitable));
}

TEST_F(DeviceOpTrackerDeathTest, OnQueueSubmissionAbortedMismatchedFence) {
    VkFence wrongFence = reinterpret_cast<VkFence>(0x9999'0000);

    ASSERT_DEATH(
        {
            DeviceOpTracker tracker(mDevice, &mDispatch);
            DeviceOpBuilder builder(tracker);
            builder.CreateFenceForOp();
            builder.OnQueueSubmissionAborted(wrongFence);
        },
        MatchesStdRegex(".*Invalid usage.*"));
}

TEST_F(DeviceOpTrackerDeathTest, UnfinishedBuilderTriggersFatal) {
    ASSERT_DEATH(
        {
            DeviceOpTracker tracker(mDevice, &mDispatch);
            DeviceOpBuilder builder(tracker);
            builder.CreateFenceForOp();
            // Neither OnQueueSubmittedWithFence nor OnQueueSubmissionAborted is called
        },
        MatchesStdRegex(".*Invalid usage.*"));
}

}  // namespace
}  // namespace vk
}  // namespace host
}  // namespace gfxstream
