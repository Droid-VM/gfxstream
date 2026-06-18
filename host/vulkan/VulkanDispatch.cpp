// Copyright (C) 2018 The Android Open Source Project
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

#include "VulkanDispatch.h"

#include <cstdio>

#include "gfxstream/SharedLibrary.h"
#include "gfxstream/files/PathUtils.h"
#include "gfxstream/synchronization/Lock.h"
#include "gfxstream/system/System.h"
#include "gfxstream/common/logging.h"

using gfxstream::base::AutoLock;
using gfxstream::base::Lock;
using gfxstream::base::pj;
using gfxstream::base::pathExists;

namespace gfxstream {
namespace vk {

static std::string icdJsonNameToProgramAndLauncherPaths(const std::string& icdFilename) {
    std::string suffix = pj({"lib64", "vulkan", icdFilename});
    std::string fullpath = pj({gfxstream::base::getProgramDirectory(), suffix});
    if (!pathExists(fullpath.c_str())) {
        fullpath = pj({gfxstream::base::getLauncherDirectory(), suffix});
    }
    if (!pathExists(fullpath.c_str())) {
        fullpath = pj({gfxstream::base::getLauncherDirectory(), "lib", "qemu", suffix});
    }
    return fullpath;
}

static void setIcdPaths(const std::string& icdFilename) {
    const std::string paths = icdJsonNameToProgramAndLauncherPaths(icdFilename);
    GFXSTREAM_INFO("Setting ICD filenames for the loader = %s", paths.c_str());
    // Set both for backwards compatibility
    gfxstream::base::setEnvironmentVariable("VK_DRIVER_FILES", paths);
    gfxstream::base::setEnvironmentVariable("VK_ICD_FILENAMES", paths);
}

static void initIcdPaths(bool forTesting) {
    auto androidIcd = gfxstream::base::getEnvironmentVariable("ANDROID_EMU_VK_ICD");
    if (androidIcd == "") {
        // Rely on user to set VK_DRIVER_FILES
        return;
    }

    if (forTesting) {
        const char* testingICD = "swiftshader";
        GFXSTREAM_INFO("%s: In test environment, enforcing %s ICD.", __func__, testingICD);
        gfxstream::base::setEnvironmentVariable("ANDROID_EMU_VK_ICD", testingICD);
        androidIcd = testingICD;
    }
    if (androidIcd == "lavapipe") {
        GFXSTREAM_INFO("%s: ICD set to 'lavapipe', using Lavapipe ICD", __func__);
        setIcdPaths("lvp_icd.x86_64.json");
    } else if (androidIcd == "swiftshader") {
        GFXSTREAM_INFO("%s: ICD set to 'swiftshader', using Swiftshader ICD", __func__);
        setIcdPaths("vk_swiftshader_icd.json");
    } else {
#ifdef __APPLE__
        // Mac: Use MoltenVK by default unless GPU mode is set to swiftshader
        if (androidIcd == "kosmickrisp") {
            gfxstream::base::setEnvironmentVariable("ANDROID_EMU_VK_ICD", "kosmickrisp");
            setIcdPaths("kosmickrisp_icd.json");
        } else {
            if (androidIcd != "moltenvk") {
                GFXSTREAM_WARNING("%s: Unknown ICD (%s), resetting to MoltenVK", __func__,
                                  androidIcd.c_str());
            }
            gfxstream::base::setEnvironmentVariable("ANDROID_EMU_VK_ICD", "moltenvk");
            setIcdPaths("MoltenVK_icd.json");
        }

        // Configure MoltenVK library with environment variables
        // 0: No logging.
        // 1: Log errors only.
        // 2: Log errors and warning messages.
        // 3: Log errors, warnings and informational messages.
        // 4: Log errors, warnings, infos and debug messages.
        const bool verboseLogs =
            (gfxstream::base::getEnvironmentVariable("ANDROID_EMUGL_VERBOSE") == "1");
        const char* logLevelValue = verboseLogs ? "4" : "1";
        gfxstream::base::setEnvironmentVariable("MVK_CONFIG_LOG_LEVEL", logLevelValue);

        //  Limit MoltenVK to use single queue, as some older ANGLE versions
        //  expect this for -guest-angle to work.
        //  0: Limit Vulkan to a single queue, with no explicit semaphore
        //  synchronization, and use Metal's implicit guarantees that all operations
        //  submitted to a queue will give the same result as if they had been run in
        //  submission order.
        gfxstream::base::setEnvironmentVariable("MVK_CONFIG_VK_SEMAPHORE_SUPPORT_STYLE", "0");

        // TODO(b/364055067)
        // MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS is not working correctly
        gfxstream::base::setEnvironmentVariable("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0");

        // MVK_CONFIG_USE_MTLHEAP is required for VK_EXT_external_memory_metal
        gfxstream::base::setEnvironmentVariable("MVK_CONFIG_USE_MTLHEAP", "1");

        // TODO(b/351765838): VVL won't work with MoltenVK due to the current
        //  way of external memory handling, add it into disable list to
        //  avoid users enabling it implicitly (i.e. via vkconfig).
        //  It can be enabled with VK_LOADER_LAYERS_ALLOW=VK_LAYER_KHRONOS_validation
        GFXSTREAM_INFO("Vulkan Validation Layers won't be enabled with MoltenVK");
        gfxstream::base::setEnvironmentVariable("VK_LOADER_LAYERS_DISABLE",
                                              "VK_LAYER_KHRONOS_validation");
#else
        // By default, on other platforms, just use whatever the system
        // is packing.
#endif
    }
}

class SharedLibraries {
   public:
    explicit SharedLibraries(size_t sizeLimit = 1) : mSizeLimit(sizeLimit) {}

    size_t size() const { return mLibs.size(); }

    bool addLibrary(const std::string& path) {
        if (size() >= mSizeLimit) {
            GFXSTREAM_WARNING("Cannot add library %s due to size limit(%d)", path.c_str(),
                              mSizeLimit);
            return false;
        }

        auto library = gfxstream::base::SharedLibrary::open(path.c_str());
        if (library) {
            mLibs.push_back(library);
            fprintf(stderr, "VK loader: opened library: %s\n", path.c_str());
            return true;
        } else {
            // This is expected when searching for a valid library path
            fprintf(stderr, "VK loader: open FAILED: %s\n", path.c_str());
            return false;
        }
    }

    bool addFirstAvailableLibrary(const std::vector<std::string>& possiblePaths) {
        for (const std::string& possiblePath : possiblePaths) {
            if (addLibrary(possiblePath)) {
                return true;
            }
        }
        return false;
    }

    ~SharedLibraries() = default;

    void* dlsym(const char* name) {
        for (const auto& lib : mLibs) {
            void* funcPtr = reinterpret_cast<void*>(lib->findSymbol(name));
            if (funcPtr) {
                return funcPtr;
            }
        }
        return nullptr;
    }

   private:
    size_t mSizeLimit;
    std::vector<gfxstream::base::SharedLibrary*> mLibs;
};

static constexpr size_t getVulkanLibraryNumLimits() {
    return 1;
}

class VulkanDispatchImpl {
   public:
    VulkanDispatchImpl() : mVulkanLibs(getVulkanLibraryNumLimits()) {}

    void initialize(bool forTesting);

    static std::vector<std::string> getPossibleLoaderPathBasenames() {
#if defined(__APPLE__)
        return std::vector<std::string>{"libvulkan.dylib"};
#elif defined(__linux__)
        return std::vector<std::string>{
            "libvulkan.so",
            "libvulkan.so.1",
        };
#elif defined(_WIN32)
        return std::vector<std::string>{"vulkan-1.dll"};
#elif defined(__QNX__)
        return std::vector<std::string>{
            "libvulkan.so",
            "libvulkan.so.1",
        };
#else
#error "Unhandled platform in VulkanDispatchImpl."
#endif
    }

    std::vector<std::string> getPossibleLoaderPaths() {
        const std::string explicitPath =
            gfxstream::base::getEnvironmentVariable("ANDROID_EMU_VK_LOADER_PATH");
        if (!explicitPath.empty()) {
            return {
                explicitPath,
            };
        }

        const std::vector<std::string> possibleBasenames = getPossibleLoaderPathBasenames();

        const std::string explicitIcd = gfxstream::base::getEnvironmentVariable("ANDROID_EMU_VK_ICD");

#ifdef _WIN32
        constexpr const bool isWindows = true;
#else
        constexpr const bool isWindows = false;
#endif
        if (explicitIcd.empty() || isWindows) {
            return possibleBasenames;
        }

        std::vector<std::string> possibleDirectories;

        if (mForTesting || explicitIcd == "mock") {
            possibleDirectories = {
                pj({gfxstream::base::getProgramDirectory(), "testlib64"}),
                pj({gfxstream::base::getLauncherDirectory(), "testlib64"}),
            };
        }

        possibleDirectories.push_back(
            pj({gfxstream::base::getProgramDirectory(), "lib64", "vulkan"}));
        possibleDirectories.push_back(
            pj({gfxstream::base::getLauncherDirectory(), "lib64", "vulkan"}));

        std::vector<std::string> possiblePaths;
        for (const std::string& possibleDirectory : possibleDirectories) {
            for (const std::string& possibleBasename : possibleBasenames) {
                possiblePaths.push_back(pj({possibleDirectory, possibleBasename}));
            }
        }
        return possiblePaths;
    }

    void* dlopen() {
        if (mVulkanLibs.size() == 0) {
            const std::vector<std::string> possiblePaths = getPossibleLoaderPaths();
            // Diag (Gunyah/crosvm host Vulkan bring-up): use fprintf(stderr) — GFXSTREAM_INFO is
            // buffered and lost before abort. Show env + candidate paths so we can tell whether the
            // host Vulkan loader is even being found, vs found-but-missing-symbols.
            fprintf(stderr,
                    "VK loader bring-up: ANDROID_EMU_VK_LOADER_PATH='%s' ANDROID_EMU_VK_ICD='%s' "
                    "VK_ICD_FILENAMES='%s' LD_LIBRARY_PATH='%s'\n",
                    gfxstream::base::getEnvironmentVariable("ANDROID_EMU_VK_LOADER_PATH").c_str(),
                    gfxstream::base::getEnvironmentVariable("ANDROID_EMU_VK_ICD").c_str(),
                    gfxstream::base::getEnvironmentVariable("VK_ICD_FILENAMES").c_str(),
                    gfxstream::base::getEnvironmentVariable("LD_LIBRARY_PATH").c_str());
            for (const std::string& p : possiblePaths) {
                fprintf(stderr, "VK loader candidate: %s\n", p.c_str());
            }
            if (!mVulkanLibs.addFirstAvailableLibrary(possiblePaths)) {
                fprintf(stderr, "VK loader: cannot open any of %d candidate libraries\n",
                        (int)possiblePaths.size());
            }
            fprintf(stderr, "VK loader: %d library(ies) loaded\n", (int)mVulkanLibs.size());
        }
        return static_cast<void*>(&mVulkanLibs);
    }

    void* dlsym(void* lib, const char* name) {
        return (void*)((SharedLibraries*)(lib))->dlsym(name);
    }

    VulkanDispatch* dispatch() { return &mDispatch; }

   private:
    Lock mLock;
    bool mForTesting = false;
    bool mInitialized = false;
    VulkanDispatch mDispatch;
    SharedLibraries mVulkanLibs;
};

VulkanDispatchImpl* sVulkanDispatchImpl() {
    static VulkanDispatchImpl* impl = new VulkanDispatchImpl;
    return impl;
}

static void* sVulkanDispatchDlOpen() { return sVulkanDispatchImpl()->dlopen(); }

static void* sVulkanDispatchDlSym(void* lib, const char* sym) {
    return sVulkanDispatchImpl()->dlsym(lib, sym);
}

void VulkanDispatchImpl::initialize(bool forTesting) {
    AutoLock lock(mLock);

    if (mInitialized) {
        return;
    }

    mForTesting = forTesting;
    initIcdPaths(mForTesting);

    init_vulkan_dispatch_from_system_loader(sVulkanDispatchDlOpen, sVulkanDispatchDlSym,
                                            &mDispatch);

    // Android workaround: the Android Vulkan loader (system libvulkan.so) does not expose the
    // global commands (vkCreateInstance / vkEnumerateInstance*) as dlsym-able symbols; per the
    // Vulkan loader spec they must be queried via vkGetInstanceProcAddr(NULL, name). The generated
    // goldfish_vk_dispatch.cpp uses plain dlsym (and is autogenerated, so it can't be patched
    // there), so fix up the global commands in this (non-generated) file. Desktop loaders export
    // them directly, so these fallbacks only trigger on Android where dlsym returned null.
    if (mDispatch.vkGetInstanceProcAddr) {
        if (!mDispatch.vkCreateInstance) {
            mDispatch.vkCreateInstance = (PFN_vkCreateInstance)mDispatch.vkGetInstanceProcAddr(
                nullptr, "vkCreateInstance");
        }
        if (!mDispatch.vkEnumerateInstanceExtensionProperties) {
            mDispatch.vkEnumerateInstanceExtensionProperties =
                (PFN_vkEnumerateInstanceExtensionProperties)mDispatch.vkGetInstanceProcAddr(
                    nullptr, "vkEnumerateInstanceExtensionProperties");
        }
        if (!mDispatch.vkEnumerateInstanceLayerProperties) {
            mDispatch.vkEnumerateInstanceLayerProperties =
                (PFN_vkEnumerateInstanceLayerProperties)mDispatch.vkGetInstanceProcAddr(
                    nullptr, "vkEnumerateInstanceLayerProperties");
        }
        if (!mDispatch.vkEnumerateInstanceVersion) {
            mDispatch.vkEnumerateInstanceVersion =
                (PFN_vkEnumerateInstanceVersion)mDispatch.vkGetInstanceProcAddr(
                    nullptr, "vkEnumerateInstanceVersion");
        }
        fprintf(stderr, "VK loader(VulkanDispatch): patched global cmds, vkCreateInstance=%p\n",
                (void*)mDispatch.vkCreateInstance);
    }

    mInitialized = true;
}

VulkanDispatch* vkDispatch(bool forTesting) {
    sVulkanDispatchImpl()->initialize(forTesting);
    return sVulkanDispatchImpl()->dispatch();
}

bool vkDispatchValid(const VulkanDispatch* vk) {
    return vk->vkEnumerateInstanceExtensionProperties != nullptr ||
           vk->vkGetInstanceProcAddr != nullptr || vk->vkGetDeviceProcAddr != nullptr;
}

}  // namespace vk
}  // namespace gfxstream
