// Copyright 2025 The Android Open Source Project
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

#include "external_memory.h"

#include "vk_utils.h"

namespace gfxstream {
namespace host {
namespace vk {

const char* ExternalMemory::to_string(const ExternalMemory::Mode mode) {
    switch (mode) {
        case Mode::Unknown:
            return "Unknown";
        case Mode::NotSupported:
            return "NotSupported";
        case Mode::OpaqueFd:
            return "OpaqueFd";
        case Mode::OpaqueWin32:
            return "OpaqueWin32";
        case Mode::Metal:
            return "Metal";
        case Mode::AndroidAHB:
            return "AndroidAHB";
        case Mode::QnxScreenBuffer:
            return "QnxScreenBuffer";
        case Mode::HostAllocation:
            return "HostAllocation";
        case Mode::DmaBuf:
            return "DmaBuf";
    }
    return "Unhandled";
}

std::optional<ExternalMemory::Mode> ExternalMemory::getMode(std::string modeStr) {
    for (const auto currMode : kAllValidModes) {
        if (to_string(currMode) == modeStr) {
            return currMode;
        }
    }
    return std::nullopt;
}

bool ExternalMemory::modeSupported(
    const ExternalMemory::Mode mode, const std::vector<VkExtensionProperties>& deviceExts,
    const VkPhysicalDeviceMemoryProperties& memoryProps, std::string_view driverVendor,
    VkPhysicalDevice physicalDevice,
    PFN_vkGetPhysicalDeviceImageFormatProperties2KHR getImageFormatProperties2Func) {
    std::vector<const char*> extRequired;
    getDeviceExtensionsForMode(mode, extRequired);

    if (!vk_util::extensionsSupported(deviceExts, extRequired)) {
        return false;
    }
    if (mode == Mode::HostAllocation) {
        // TODO(b/469094646): Check this during the initial gpu selection
        // Host allocation mode is designed for software renderers and only supported
        // if there is only a single type of memory. Some drivers may still report
        // invalid memory indices due to bugs in the extension's implementation, but
        // this check ensures that we can safely keep using the memory in such cases.
        if (memoryProps.memoryHeapCount != 1 || memoryProps.memoryTypeCount != 1) {
            GFXSTREAM_INFO(
                "Cannot use external memory mode HostAllocation with multiple memory types");
            return false;
        }
    }

    if (mode == Mode::DmaBuf) {
#if defined(__QNX__)
        // TODO(aruby@qnx.com): Remove once dmabuf extension support has been flushed out on QNX
        GFXSTREAM_INFO("External memory mode DmaBuf is not supported on QNX");
        return false;
#endif
        if (physicalDevice == VK_NULL_HANDLE || getImageFormatProperties2Func == nullptr) {
            // DmaBuf mode requires format support check, which needs valid device and function
            GFXSTREAM_INFO("Cannot use external memory mode DmaBuf without valid device and function");
            return false;
        }

        // Check if must-support image formats are supported, to avoid runtime crashes
        // This check is only done for DmaBuf mode for now to avoid regressions.
        const std::vector<VkFormat> kFormatsToCheck = {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R8G8B8A8_SRGB,
        };
        for (const auto& format : kFormatsToCheck) {
            VkPhysicalDeviceImageFormatInfo2 formatInfo2 = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                .pNext = nullptr,
                .format = format,
                .type = VK_IMAGE_TYPE_2D,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
            };

            VkPhysicalDeviceExternalImageFormatInfo extInfo = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                .pNext = nullptr,
                .handleType = getHandleType(mode),
            };

            formatInfo2.pNext = &extInfo;

            VkImageFormatProperties2 outProps2 = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                .pNext = nullptr,
            };

            VkExternalImageFormatProperties outExternalProps = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                .pNext = nullptr,
            };

            outProps2.pNext = &outExternalProps;

            VkResult res = getImageFormatProperties2Func(physicalDevice, &formatInfo2, &outProps2);
            if (res == VK_ERROR_FORMAT_NOT_SUPPORTED) {
                GFXSTREAM_INFO(
                    "%s: Mandatory format %s is not supported for external memory mode %s",
                    __func__, string_VkFormat(format), to_string(mode));
                return false;
            } else if (res != VK_SUCCESS) {
                GFXSTREAM_WARNING(
                    "%s: vkGetPhysicalDeviceImageFormatProperties2 failed for mode %s: %s",
                    __func__, to_string(mode), string_VkResult(res));
                return false;
            }

            VkExternalMemoryFeatureFlags featureFlags =
                outExternalProps.externalMemoryProperties.externalMemoryFeatures;
            if (!(featureFlags & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) ||
                !(featureFlags & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
                GFXSTREAM_INFO(
                    "%s: format %s does not support required export/import features for mode %s",
                    __func__, string_VkFormat(format), to_string(mode));
                return false;
            }
        }

        // Lastly, check for some known problematic drivers
        // This list should be removed once all issues are found through format support checks.
        bool dmaBufBlockList = (driverVendor == "NVIDIA (Vendor 0x10de)");
#ifdef CONFIG_AEMU
        // TODO(b/400999642): dma_buf support should be checked with image format support
        dmaBufBlockList |= (driverVendor == "radv (Vendor 0x1002)");
#endif
        if (dmaBufBlockList) {
            GFXSTREAM_INFO("External memory mode DmaBuf is not supported on this device");
            return false;
        }
    }

    // All checks passed, the mode can be used.
    return true;
}

ExternalMemory::Mode ExternalMemory::calculateMode(
    const std::vector<VkExtensionProperties>& deviceExts,
    const VkPhysicalDeviceMemoryProperties& memoryProps, std::optional<std::string> modeStrOpt,
    std::string_view driverVendor, VkPhysicalDevice physicalDevice,
    PFN_vkGetPhysicalDeviceImageFormatProperties2KHR getImageFormatProperties2Func) {
    if (modeStrOpt) {
        auto mode = getMode(*modeStrOpt);
        if (!mode) {
            GFXSTREAM_ERROR(
                "%s(): Could not find an ExternalMemoryMode matching the provided "
                "VulkanExternalMemoryMode string: %s ",
                __func__, modeStrOpt->c_str());

            return Mode::Unknown;
        }

        if (!modeSupported(*mode, deviceExts, memoryProps, driverVendor, physicalDevice,
                           getImageFormatProperties2Func)) {
            GFXSTREAM_ERROR(
                "%s(): Vulkan driver does not support the memory mode provided by the "
                "VulkanExternalMemoryMode string: %s",
                __func__, modeStrOpt->c_str());

            return Mode::NotSupported;
        }

        return *mode;
    }

#if defined(_WIN32)
    std::array<Mode, 2> supportedModes = {
        Mode::OpaqueWin32,
        Mode::HostAllocation,
    };
#elif defined(__ANDROID__)
    std::array<Mode, 1> supportedModes = {
        Mode::AndroidAHB,
    };
#elif defined(__QNX__)
    std::array<Mode, 2> supportedModes = {
        Mode::QnxScreenBuffer,
        Mode::OpaqueFd,
    };
#elif defined(__APPLE__)
    std::array<Mode, 2> supportedModes = {
        Mode::Metal,
        Mode::OpaqueFd,
    };
#else
    std::array<Mode, 2> supportedModes = {
        Mode::DmaBuf,
        Mode::OpaqueFd,
    };
#endif

    for (auto mode : supportedModes) {
        if (modeSupported(mode, deviceExts, memoryProps, driverVendor, physicalDevice,
                          getImageFormatProperties2Func)) {
            // Supported modes are in-order of preference, return the first one supported
            return mode;
        }
    }

    GFXSTREAM_ERROR("%s: Vulkan driver doesn't support any external memory modes!", __func__);
    return Mode::NotSupported;
}

VkExternalMemoryHandleTypeFlagBits ExternalMemory::getHandleType(const ExternalMemory::Mode mode) {
    switch (mode) {
        case Mode::OpaqueFd:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        case Mode::DmaBuf:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        case Mode::OpaqueWin32:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        case Mode::Metal:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT;
        case Mode::AndroidAHB:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        case Mode::QnxScreenBuffer:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_SCREEN_BUFFER_BIT_QNX;
        case Mode::HostAllocation:
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        default:
            // Should not call this function with Unknown. Not a fatal error as the
            // value retrieved might be used with external memory support check
            GFXSTREAM_ERROR("%s: Unhandled external memory mode '%s'", __func__, to_string(mode));
    }

    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_FLAG_BITS_MAX_ENUM;
}

void ExternalMemory::getDeviceExtensionsForMode(const ExternalMemory::Mode mode,
                                                std::vector<const char*>& outDeviceExtensions) {
    // These will always be necessary
    outDeviceExtensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    outDeviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    outDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);

    switch (mode) {
        case Mode::OpaqueFd:
            outDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            break;
        case Mode::DmaBuf:
            // A dma-buf is a Linux kernel construct, commonly used with open-source DRM drivers.
            // See https://docs.kernel.org/driver-api/dma-buf.html for details.
            outDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            outDeviceExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
            break;
#ifdef VK_USE_PLATFORM_WIN32_KHR
        case Mode::OpaqueWin32:
            outDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            break;
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
        case Mode::Metal:
            outDeviceExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_METAL_EXTENSION_NAME);
            break;
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
        case Mode::AndroidAHB:
            outDeviceExtensions.push_back(
                VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
            break;
#endif
#ifdef VK_USE_PLATFORM_SCREEN_QNX
        case Mode::QnxScreenBuffer:
            outDeviceExtensions.push_back(VK_QNX_EXTERNAL_MEMORY_SCREEN_BUFFER_EXTENSION_NAME);
            // EXT_queue_family_foreign is an extension dependency of
            // VK_QNX_external_memory_screen_buffer
            outDeviceExtensions.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
            break;
#endif
        case Mode::HostAllocation:
            outDeviceExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
            break;
        default:
            GFXSTREAM_FATAL("%s: Invalid external memory mode %s!", __func__, to_string(mode));
    }
}

}  // namespace vk
}  // namespace host
}  // namespace gfxstream