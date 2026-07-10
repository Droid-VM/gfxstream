// Copyright (C) 2024 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vulkan/vulkan.h>

#include <optional>

#include "gfxstream/host/Features.h"

namespace gfxstream {
namespace vk {

// A physical device may have memory types that are not desirable or are not
// supportable by the host renderer. This class helps to track the original
// host memory types, helps to track the emulated memory types shared with the
// guest, and helps to convert between both.
class EmulatedPhysicalDeviceMemoryProperties {
   public:
    EmulatedPhysicalDeviceMemoryProperties(const VkPhysicalDeviceMemoryProperties& host,
                                           uint32_t hostColorBufferMemoryTypeIndex,
                                           const gfxstream::host::FeatureSet& features);

    struct HostMemoryInfo {
        uint32_t index;
        VkMemoryType memoryType;
    };
    std::optional<HostMemoryInfo> getHostMemoryInfoFromHostMemoryTypeIndex(
        uint32_t hostMemoryTypeIndex) const;
    std::optional<HostMemoryInfo> getHostMemoryInfoFromGuestMemoryTypeIndex(
        uint32_t guestMemoryTypeIndex) const;

    const VkPhysicalDeviceMemoryProperties& getGuestMemoryProperties() const {
        return mGuestMemoryProperties;
    }
    const VkPhysicalDeviceMemoryProperties& getHostMemoryProperties() const {
        return mHostMemoryProperties;
    }

    uint32_t getGuestColorBufferMemoryTypeIndex() const {
        return mGuestColorBufferMemoryTypeIndex;
    }

    void transformToGuestMemoryRequirements(VkMemoryRequirements* hostMemoryRequirements) const;

    // True if the guest memory type is one of the synthesized device-local-only "shadow"
    // types (see the constructor). Allocations of those types stay entirely in host GPU
    // memory: the guest never maps them, so no host-visible blob / PCI BAR space is used.
    bool isDeviceLocalOnlyShadow(uint32_t guestMemoryTypeIndex) const;

   private:
    VkPhysicalDeviceMemoryProperties mGuestMemoryProperties;
    VkPhysicalDeviceMemoryProperties mHostMemoryProperties;
    uint32_t mGuestToHostMemoryTypeIndexMap[VK_MAX_MEMORY_TYPES];
    uint32_t mHostToGuestMemoryTypeIndexMap[VK_MAX_MEMORY_TYPES];

    // hostMemoryTypeIndex -> guest index of its device-local-only shadow (or kInvalid).
    uint32_t mHostToGuestDeviceLocalShadowMap[VK_MAX_MEMORY_TYPES];
    // Guest memory type indices that are device-local-only shadows.
    bool mGuestMemoryTypeIsShadow[VK_MAX_MEMORY_TYPES];

    // The memory type index reported to the guest for VkDeviceMemory requirements which would
    // try to import host ColorBuffer allocations
    // (e.g. vkGetAndroidHardwareBufferPropertiesANDROID()).
    uint32_t mGuestColorBufferMemoryTypeIndex;
};

}  // namespace vk
}  // namespace gfxstream
