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

#include "virtio_gpu_resource.h"

#include <drm/drm_fourcc.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "frame_buffer.h"
#include "gfxstream_diag.h"
#include "virtio_gpu_format_utils.h"
#include "vulkan/host_visible_pool.h"

namespace gfxstream {
namespace host {
namespace {

using gfxstream::base::DescriptorType;
#ifdef GFXSTREAM_BUILD_WITH_SNAPSHOT_FRONTEND_SUPPORT
using gfxstream::host::snapshot::VirtioGpuResourceCreateArgs;
using gfxstream::host::snapshot::VirtioGpuResourceCreateBlobArgs;
using gfxstream::host::snapshot::VirtioGpuResourceSnapshot;
#endif


// Gunyah workaround: recycle RingBlob backing memory and never free it.
//
// On Gunyah, the VMM SHAREs the host-visible blob pages with the guest, and a
// SHARE is permanent: once a GPA is SHARE'd to a physical page, it cannot be
// re-pointed at a different page later. So when the guest re-maps a host-visible
// blob at a BAR offset it used before, the new blob MUST land on the same
// physical pages as the old one, or the guest reads stale data.
//
// We achieve this with a recycle pool: a RingBlob's backing memory is never
// freed, and when a same-size RingBlob is later needed we hand back a
// previously-freed one (its physical pages intact) instead of allocating new
// memory. gfxstream cannot see the guest BAR offset/GPA (it is only known on the
// crosvm side at map time), so we key the pool by size and reuse most-recently-
// freed first (LIFO) — guest address allocators tend to re-hand-out the most-
// recently-freed offset, and ASG ring blobs are per-context, fixed-size, and
// created/destroyed serially, so this matches "same GPA reuses same pages" in
// practice.
//
// Only enabled when the host VMM runs on Gunyah (gated by the
// GFXSTREAM_GUNYAH_PIN_RINGBLOB env var); other hosts (e.g. KVM-based
// crosvm/qemu) are unaffected and keep the normal allocate/free behavior.
bool ShouldPinRingBlobsForGunyah() {
    static const bool enabled = [] {
        const char* v = std::getenv("GFXSTREAM_GUNYAH_PIN_RINGBLOB");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
}

struct GunyahRingBlobPool {
    std::mutex mutex;
    // size -> stack of released RingBlobs available for reuse (LIFO), held WEAKLY.
    //
    // Weakly, because `pinned` below already owns every blob forever, and the recycle test is
    // "does anyone still hold this one" -- expressed as use_count() == 1, meaning `pinned` is the
    // only owner left. A strong reference here would add a second owner and make that test
    // permanently false: no blob would ever be handed back, every context would carve a fresh
    // one, and the whole recycling mechanism would be dead code that silently leaks a ring per
    // context. That is exactly what it was doing -- a 64 MB gfx-host pool ran out after ~63
    // contexts (RINGBLOB-POOL-MISS[exhausted] used=65244KB high=65244KB), after which every
    // further ring fell back to a fresh mlocked 2 MB memfd + a runtime Gunyah SHARE.
    std::unordered_map<uint64_t, std::vector<std::weak_ptr<RingBlob>>> freeBySize;
    // Permanent reference to every RingBlob ever created so its pages never free.
    std::vector<std::shared_ptr<RingBlob>> pinned;
};

GunyahRingBlobPool& GetGunyahRingBlobPool() {
    static GunyahRingBlobPool* pool = new GunyahRingBlobPool();  // intentional leak
    return *pool;
}

// Reuse a freed same-size RingBlob if available, else create one and pin it.
std::shared_ptr<RingBlob> AcquireGunyahRingBlob(uint32_t id, uint64_t size, uint64_t alignment,
                                                bool externalBlob) {
    auto& pool = GetGunyahRingBlobPool();
    std::lock_guard<std::mutex> lock(pool.mutex);

    // Round the backing up to a 2MB multiple so the shmem can be collapsed into
    // whole order-9 folios (the memfd is created sealed, so it cannot be grown
    // later with ftruncate). This is a requirement of the fresh-memfd path at the
    // bottom of this function and of nothing else -- see the pool branch below.
    constexpr uint64_t kPmdSize2 = 2ULL * 1024 * 1024;
    const uint64_t roundedSize = (size + kPmdSize2 - 1) & ~(kPmdSize2 - 1);

    // The recycle map is keyed by blob->size(), which is what Release() sees, so each backing
    // path has to look itself up under the size IT charges. Getting this wrong is not a missed
    // optimisation: a pooled blob is moved into pool.pinned and its chunks are never returned to
    // the allocator, so every recycle miss burns another slice of the pool permanently.
    auto takeFree = [&pool](uint64_t sz) -> std::shared_ptr<RingBlob> {
        auto it = pool.freeBySize.find(sz);
        if (it == pool.freeBySize.end() || it->second.empty()) return nullptr;

        // Only a blob nobody else still holds. A resource being dropped puts its RingBlob back
        // here immediately, but the consumer that was reading it is a separate thread with its own
        // reference and does not stop just because the resource did. Handing such a blob to a new
        // context puts two render threads on one ring: the old one goes on consuming, the new one
        // waits in its opening read for bytes the old one is taking, and neither ever finishes.
        // Seen directly -- two RING-VIEW lines with the same storage address, two RT-ENTERs, and
        // only the first HS-DONE.
        //
        // use_count()==1 means this vector is the last owner, so the reader is gone. Anything else
        // stays where it is and is looked at again next time.
        auto idle = std::find_if(it->second.begin(), it->second.end(),
                                 [](const std::weak_ptr<RingBlob>& b) { return b.use_count() == 1; });
        if (idle == it->second.end()) return nullptr;
        std::shared_ptr<RingBlob> blob = idle->lock();
        it->second.erase(idle);
        if (!blob) return nullptr;  // cannot happen while `pinned` owns it; not worth crashing on

        // Same physical pages as before, and that is the point -- but they still hold the previous
        // ring's header. A recycled blob arrives with the old write and read positions and the old
        // state byte, and whether the session survives then depends on a race: the guest
        // initialises the ring when it maps the blob, this side attaches when the guest pings, and
        // nothing orders those two. Attach first and the consumer is looking at a ring that says
        // it is empty and already consumed, so it waits in the opening four-byte read for bytes
        // that, as far as its view is concerned, were taken long ago -- and the guest waits for a
        // reply that will never come.
        //
        // Measured: every healthy attach reports write=0 read=0 state=0, the stuck one reported
        // write=16 read=16 state=1. Hand back pages that say nothing rather than pages that lie.
        if (void* mem = blob->map()) {
            memset(mem, 0, blob->size());
        }
        return blob;
    };

    // DroidVM gfxstream PRE-ALLOC: back the ASG RingBlob from the boot-blessed GpuPool so it is
    // pool-resident (guest maps the pool GPA directly) with ZERO runtime SHARE — the last thing
    // that still needed /dev/gunyah_share. This makes pre-alloc fully self-sufficient on phones
    // where the gunyah_host_share module can't be installed. The pool is already folio-backed and
    // mlocked at boot, so no per-blob collapse/mlock is needed.
    //
    // Charge the pool only what the ring actually needs. The 2MB round above is there so a
    // standalone memfd can be MADV_COLLAPSE'd into order-9 folios; a pool blob is never
    // collapsed, mlocked or mmap'd per-blob, so none of that applies to it. The ASG blob asks
    // for kAsgConsumerRingStorageSize + kAsgWriteBufferSize (1036KB), and rounding that to 2MB
    // charged nearly twice what it used -- with one ring per guest context, that was the pool's
    // entire footprint. `alignment` is the guest page size the caller already computed, which is
    // also the real floor: the guest maps the blob with io_remap_pfn_range at page granularity.
    if (auto* hvPool = vk::HostVisiblePool::get()) {
        const char* poolMiss = nullptr;
        const uint64_t align = alignment ? alignment : 4096;
        const uint64_t poolSize = (size + align - 1) & ~(align - 1);
        if (auto reused = takeFree(poolSize)) return reused;

        auto chunks = hvPool->alloc(poolSize, align);
        if (chunks.size() == 1) {
            // map-once: back the RingBlob with a pointer into the whole-pool mapping
            // (hvaForOffset = baseHva + offset), no per-blob mmap.
            auto pooled = RingBlob::CreateFromPool(id, poolSize,
                                                   hvPool->hvaForOffset(chunks[0].offset),
                                                   (int64_t)chunks[0].offset);
            if (pooled) {
                std::shared_ptr<RingBlob> pblob = std::move(pooled);
                if (void* addr = pblob->map()) std::memset(addr, 0, poolSize);
                pool.pinned.push_back(pblob);
                GFXSTREAM_DIAG_PRINT( "RINGBLOB-POOL: id=%u size=%llu -> pool offset=0x%llx (no SHARE)\n",
                        id, (unsigned long long)poolSize,
                        (unsigned long long)chunks[0].offset);
                return pblob;
            }
            hvPool->free(chunks);
            poolMiss = "create-from-pool-failed";
        } else if (!chunks.empty()) {
            hvPool->free(chunks);  // fragmented: fall back to fresh memfd + runtime SHARE
            poolMiss = "fragmented";
        } else {
            poolMiss = "exhausted";
        }
        // One ASG ring per guest context, and the pool is the only thing standing between that
        // ring and a runtime SHARE + guest MEM_ACCEPT -- which is where this route's accept
        // failures live. A miss used to leave no trace at all: the success line is behind
        // GFXSTREAM_DIAG and both failure branches just fell through silently, so the first
        // evidence of one was a blob map failing several layers down with no hint that the pool
        // had been asked and had said no. Say it once per kind, with the state that caused it:
        // "exhausted" wants a bigger gfx-host-mb, "fragmented" wants the multi-chunk path.
        if (poolMiss) {
            static std::atomic<uint64_t> sRingPoolMisses{0};
            const uint64_t n = ++sRingPoolMisses;
            if ((n & (n - 1)) == 0) {
                auto st = hvPool->stats();
                fprintf(stderr,
                        "RINGBLOB-POOL-MISS[%s] #%llu: req=%lluKB pool used=%lluKB high=%lluKB "
                        "free=%lluKB largest=%lluKB blocks=%zu -> fresh memfd + runtime SHARE\n",
                        poolMiss, (unsigned long long)n, (unsigned long long)(poolSize >> 10),
                        (unsigned long long)(st.used >> 10), (unsigned long long)(st.highWater >> 10),
                        (unsigned long long)(st.freeBytes >> 10),
                        (unsigned long long)(st.largest >> 10), st.blocks);
                fflush(stderr);
            }
        }
    }

    if (auto reused = takeFree(roundedSize)) return reused;

    // Gunyah persistent-BAR (fixed_blob_mapping) path: the VMM maps each blob into the shared BAR
    // via add_fd_mapping(), which requires an exportable fd. AlignedMemory (CreateWithHostMemory) is
    // NOT exportable, so always back the RingBlob with shmem (memfd) when pinning for Gunyah,
    // regardless of the ExternalBlob feature flag.
    (void)externalBlob;
    std::unique_ptr<RingBlob> created = RingBlob::CreateWithShmem(id, roundedSize);
    if (!created) {
        return nullptr;
    }
    std::shared_ptr<RingBlob> blob = std::move(created);

#ifndef _WIN32
    // Back the blob with 2MB (order-9) folios from the gh_hugepage_reserve pool.
    //
    // crosvm is a tracked gunyah-VM owner, so every order-9 allocation it makes is
    // intercepted and served from the module's reserve pool: the blob then consumes
    // VM reserve quota instead of competing with apps for system RAM, gunyah_share_66
    // coalesces each folio into a single 2MB mem_entry (instead of 512 4K entries),
    // and on ANY exit path — including SIGKILL/SIGSEGV — the final folio free reaches
    // the buddy allocator at order 9, where the module's free hook reclaims it back
    // into the pool. The folio must never be split (no partial unmap/hole-punch of a
    // 2MB chunk), or its 4K frees become invisible to the hook and the pages are lost
    // to the pool until the owner dies.
    //
    // Recipe mirrors crosvm's proven mthp guest-RAM path: round the memfd up to a 2MB
    // multiple, fault it in through a PMD-aligned temporary mapping, MADV_COLLAPSE.
    // Collapse allocates the order-9 folio in-process (madvise runs in crosvm's mm,
    // which is what the module's intercept matches on). Best-effort: on failure the
    // blob simply stays 4K-backed like before.
    {
        constexpr uint64_t kPmdSize = 2ULL * 1024 * 1024;
#ifndef MADV_COLLAPSE
        constexpr int kMadvCollapse = 25;
#else
        constexpr int kMadvCollapse = MADV_COLLAPSE;
#endif
        int hfd = blob->isExportable() ? blob->dupHandle() : -1;
        if (hfd >= 0) {
            int collapseRet = -1, collapseErrno = 0;
            void* rsv = mmap(nullptr, roundedSize + kPmdSize, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (rsv != MAP_FAILED) {
                uintptr_t alignedAddr =
                    (reinterpret_cast<uintptr_t>(rsv) + kPmdSize - 1) & ~(kPmdSize - 1);
                void* aligned = mmap(reinterpret_cast<void*>(alignedAddr), roundedSize,
                                     PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, hfd, 0);
                if (aligned != MAP_FAILED) {
                    madvise(aligned, roundedSize, MADV_HUGEPAGE);
                    std::memset(aligned, 0, roundedSize);
                    collapseRet = madvise(aligned, roundedSize, kMadvCollapse);
                    collapseErrno = collapseRet ? errno : 0;
                }
                munmap(rsv, roundedSize + kPmdSize);
            }
            close(hfd);
            if (collapseRet) {
                GFXSTREAM_DIAG_PRINT( "RINGBLOB-POOL: collapse failed id=%u rounded=%llu errno=%d\n",
                        id, (unsigned long long)roundedSize, collapseErrno);
            }
        }
    }
#endif

    // Gunyah SHARE (lend=false) does not fault in or lock the backing pages. A shmem/memfd-backed
    // RingBlob is demand-paged, so the guest's first access to a not-yet-present page would SIGBUS.
    // Touch every page to fault it in, then mlock to keep it resident for the VM's lifetime
    // (mirrors how qemu's pre-allocated hostmem backend is always resident).
#ifndef _WIN32
    if (void* addr = blob->map()) {
        std::memset(addr, 0, roundedSize);
        if (mlock(addr, roundedSize) != 0) {
            GFXSTREAM_DIAG_PRINT( "RINGBLOB-PIN: mlock failed id=%u size=%llu errno=%d\n", id,
                    (unsigned long long)size, errno);
        }
    }
#endif

    pool.pinned.push_back(blob);  // keep alive forever
    return blob;
}

void ReleaseGunyahRingBlob(const std::shared_ptr<RingBlob>& blob) {
    if (!blob) {
        return;
    }
    auto& pool = GetGunyahRingBlobPool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.freeBySize[blob->size()].push_back(std::weak_ptr<RingBlob>(blob));
}

enum pipe_texture_target {
    PIPE_BUFFER,
    PIPE_TEXTURE_1D,
    PIPE_TEXTURE_2D,
    PIPE_TEXTURE_3D,
    PIPE_TEXTURE_CUBE,
    PIPE_TEXTURE_RECT,
    PIPE_TEXTURE_1D_ARRAY,
    PIPE_TEXTURE_2D_ARRAY,
    PIPE_TEXTURE_CUBE_ARRAY,
    PIPE_MAX_TEXTURE_TYPES,
};

/**
 *  Resource binding flags -- state tracker must specify in advance all
 *  the ways a resource might be used.
 */
#define PIPE_BIND_DEPTH_STENCIL (1 << 0)        /* create_surface */
#define PIPE_BIND_RENDER_TARGET (1 << 1)        /* create_surface */
#define PIPE_BIND_BLENDABLE (1 << 2)            /* create_surface */
#define PIPE_BIND_SAMPLER_VIEW (1 << 3)         /* create_sampler_view */
#define PIPE_BIND_VERTEX_BUFFER (1 << 4)        /* set_vertex_buffers */
#define PIPE_BIND_INDEX_BUFFER (1 << 5)         /* draw_elements */
#define PIPE_BIND_CONSTANT_BUFFER (1 << 6)      /* set_constant_buffer */
#define PIPE_BIND_DISPLAY_TARGET (1 << 7)       /* flush_front_buffer */
#define PIPE_BIND_STREAM_OUTPUT (1 << 10)       /* set_stream_output_buffers */
#define PIPE_BIND_CURSOR (1 << 11)              /* mouse cursor */
#define PIPE_BIND_CUSTOM (1 << 12)              /* state-tracker/winsys usages */
#define PIPE_BIND_GLOBAL (1 << 13)              /* set_global_binding */
#define PIPE_BIND_SHADER_BUFFER (1 << 14)       /* set_shader_buffers */
#define PIPE_BIND_SHADER_IMAGE (1 << 15)        /* set_shader_images */
#define PIPE_BIND_COMPUTE_RESOURCE (1 << 16)    /* set_compute_resources */
#define PIPE_BIND_COMMAND_ARGS_BUFFER (1 << 17) /* pipe_draw_info.indirect */
#define PIPE_BIND_QUERY_BUFFER (1 << 18)        /* get_query_result_resource */

static inline uint32_t AlignUp(uint32_t n, uint32_t a) { return ((n + a - 1) / a) * a; }

struct ResourceFormatInfo {
    uint32_t drm_fourcc;
    int bpp;
};

static std::unordered_map<int, struct ResourceFormatInfo> virglFormatInfoMap = {
    {VIRGL_FORMAT_B8G8R8A8_UNORM, {DRM_FORMAT_ARGB8888, 4}},
    {VIRGL_FORMAT_B8G8R8X8_UNORM, {DRM_FORMAT_XRGB8888, 4}},
    {VIRGL_FORMAT_B5G6R5_UNORM, {DRM_FORMAT_RGB565, 2}},
    {VIRGL_FORMAT_R8G8B8A8_UNORM, {DRM_FORMAT_ABGR8888, 4}},
    {VIRGL_FORMAT_R8G8B8X8_UNORM, {DRM_FORMAT_XBGR8888, 4}},
    {VIRGL_FORMAT_R8_UNORM, {DRM_FORMAT_R8, 1}},
};

static std::optional<struct ResourceFormatInfo> VirglFormatInfo(uint32_t virglFormat) {
    auto it = virglFormatInfoMap.find(virglFormat);
    if (virglFormatInfoMap.end() != it) {
        return it->second;
    }
    return std::nullopt;
}

VirtioGpuResourceType GetResourceType(const struct stream_renderer_resource_create_args& args) {
    if (args.target == PIPE_BUFFER && args.bind == VIRGL_BIND_CUSTOM) {
        return VirtioGpuResourceType::PIPE;
    }

    if (args.format != VIRGL_FORMAT_R8_UNORM) {
        return VirtioGpuResourceType::COLOR_BUFFER;
    }
    if (args.bind & VIRGL_BIND_SAMPLER_VIEW) {
        return VirtioGpuResourceType::COLOR_BUFFER;
    }
    if (args.bind & VIRGL_BIND_RENDER_TARGET) {
        return VirtioGpuResourceType::COLOR_BUFFER;
    }
    if (args.bind & VIRGL_BIND_SCANOUT) {
        return VirtioGpuResourceType::COLOR_BUFFER;
    }
    if (args.bind & VIRGL_BIND_CURSOR) {
        return VirtioGpuResourceType::COLOR_BUFFER;
    }
    if (!(args.bind & VIRGL_BIND_LINEAR)) {
        // Always treat large single dimensional R8 requests as buffers, even
        // if they didn't request linear binding
        const uint32_t largeBufferLimit = 16000;
        bool shouldUseBuffer = args.width > largeBufferLimit && args.height == 1 &&
                               args.depth == 1 && args.array_size == 1;
        if (shouldUseBuffer) {
            return VirtioGpuResourceType::BUFFER;
        }
        return VirtioGpuResourceType::COLOR_BUFFER;
    }

    return VirtioGpuResourceType::BUFFER;
}

// Fills `outHandle` from an exported descriptor, consolidating the platform
// `#ifdef`s: on Android the descriptor is an opaque handle taken by value; on
// other platforms it is a ManagedDescriptor whose ownership is released to the
// VMM. Returns 0 on success, -EINVAL if the descriptor could not be obtained.
int fillExportHandle(struct stream_renderer_handle* outHandle, BlobDescriptorType& descriptorInfo) {
#ifdef __ANDROID__
    // Hand the VMM its own dup of an fd-type handle and keep ours, which Destroy() closes. Handing
    // out the stored value transferred ownership implicitly the moment the VMM closed it -- and
    // leaked it whenever the VMM never asked for an export at all.
    auto rawDescriptor = descriptorInfo.handle;
    switch (descriptorInfo.streamHandleType) {
        case STREAM_HANDLE_TYPE_MEM_OPAQUE_FD:
        case STREAM_HANDLE_TYPE_MEM_DMABUF:
        case STREAM_HANDLE_TYPE_MEM_SHM:
            rawDescriptor = dup(static_cast<int>(rawDescriptor));
            if (static_cast<int>(rawDescriptor) < 0) {
                return -EINVAL;
            }
            break;
        default:
            // Not an fd: the VMM takes a reference of its own kind, or none at all.
            break;
    }
#else
    auto rawDescriptorOpt = descriptorInfo.descriptor.release();
    if (!rawDescriptorOpt) {
        return -EINVAL;
    }
    auto rawDescriptor = *rawDescriptorOpt;
#endif
#ifdef _WIN32
    outHandle->os_handle = static_cast<int64_t>(reinterpret_cast<intptr_t>(rawDescriptor));
#else
    outHandle->os_handle = static_cast<int64_t>(rawDescriptor);
#endif
    outHandle->handle_type = descriptorInfo.streamHandleType;
    return 0;
}

}  // namespace

/*static*/
std::optional<VirtioGpuResource> VirtioGpuResource::Create(
    const struct stream_renderer_resource_create_args* args, struct iovec* iov, uint32_t num_iovs) {

    const auto resourceType = GetResourceType(*args);
    GFXSTREAM_DEBUG("resource id: %u, type: %d", args->handle, (int)resourceType);
    if (resourceType == VirtioGpuResourceType::BLOB) {
        GFXSTREAM_ERROR("Failed to create resource: encountered blob.");
        return std::nullopt;
    }

    if (resourceType == VirtioGpuResourceType::PIPE) {
        // Frontend only resource.
    } else if (resourceType == VirtioGpuResourceType::BUFFER) {
        if (!FrameBuffer::getFB()->createBufferWithResourceHandle(args->width * args->height,
                                                                  args->handle)) {
            GFXSTREAM_ERROR("Failed to create buffer with resource handle %d.", args->handle);
            return std::nullopt;
        }
    } else if (resourceType == VirtioGpuResourceType::COLOR_BUFFER) {
        auto formatOpt = ToGfxstreamFormat(args->format);
        if (!formatOpt) {
            GFXSTREAM_ERROR("Failed to create resource: unsupported format %d", args->format);
            return std::nullopt;
        }
        auto format = *formatOpt;

        // The virgl format the guest asked this resource to be, per resource. This is the other
        // half of the pairing that decides byte order: the guest's own image format says what it
        // writes, and this says what the host thinks it is reading back.
        if (::gfxstream::host::diagEnabled()) {
            static std::mutex sSeenMutex;
            static std::set<uint32_t> sSeenResources;
            bool firstTime = false;
            {
                std::lock_guard<std::mutex> lock(sSeenMutex);
                firstTime =
                    sSeenResources.size() < 24 && sSeenResources.insert(args->handle).second;
            }
            if (firstTime) {
                GFXSTREAM_DIAG_PRINT("CB-RESOURCE: %u %ux%u virgl %u -> %s\n", args->handle,
                                     args->width, args->height, args->format,
                                     ToString(format).c_str());
            }
        }

        if (!FrameBuffer::getFB()->createColorBufferWithResourceHandle(args->width, args->height,
                                                                       format, args->handle)) {
            const std::string formatString = ToString(format);
            GFXSTREAM_ERROR("Failed to create color buffer with resource handle %d. (%dx%d, format: %s)",
                            args->handle, args->width, args->height, formatString.c_str());
            return std::nullopt;
        }
        FrameBuffer::getFB()->setGuestManagedColorBufferLifetime(true /* guest manages lifetime */);
        FrameBuffer::getFB()->openColorBuffer(args->handle);
    } else {
        GFXSTREAM_ERROR("Failed to create resource: unhandled type.");
        return std::nullopt;
    }

    VirtioGpuResource resource;
    resource.mId = args->handle;
    resource.mResourceType = resourceType;
    resource.mCreateArgs = *args;

    resource.AttachIov(iov, num_iovs);

    return resource;
}

/*static*/ std::optional<VirtioGpuResource> VirtioGpuResource::Create(
    const gfxstream::host::FeatureSet& features, uint32_t pageSize, uint32_t contextId,
    uint32_t resourceId, const struct stream_renderer_resource_create_args* createArgs,
    const struct stream_renderer_create_blob* createBlobArgs,
    const struct stream_renderer_handle* handle) {
    VirtioGpuResource resource;

    std::optional<BlobDescriptorInfo> descriptorInfoOpt;

    if (createArgs != nullptr) {
        auto resourceType = GetResourceType(*createArgs);
        if (resourceType != VirtioGpuResourceType::BUFFER &&
            resourceType != VirtioGpuResourceType::COLOR_BUFFER) {
            GFXSTREAM_ERROR("failed to create blob resource: unhandled type.");
            return std::nullopt;
        }

        auto resourceOpt = Create(createArgs, nullptr, 0);
        if (!resourceOpt) {
            return std::nullopt;
        }

        if (resourceType == VirtioGpuResourceType::BUFFER) {
            descriptorInfoOpt = FrameBuffer::getFB()->exportBuffer(resourceId);
        } else if (resourceType == VirtioGpuResourceType::COLOR_BUFFER) {
            descriptorInfoOpt = FrameBuffer::getFB()->exportColorBuffer(resourceId);
        } else {
            GFXSTREAM_ERROR("failed to create blob resource: unhandled type.");
            return std::nullopt;
        }

        resource = std::move(*resourceOpt);
    } else {
        resource.mResourceType = VirtioGpuResourceType::BLOB;
    }

    resource.mId = resourceId;
    resource.mCreateBlobArgs = *createBlobArgs;

    if (createBlobArgs->blob_id == 0) {
        RingBlobMemory memory;
        if (ShouldPinRingBlobsForGunyah()) {
            // Reuse a pinned same-size RingBlob (the same physical pages) if one is free, else
            // carve one from the pool or allocate and pin a new one. Gunyah's SHARE is permanent,
            // so the pages have to stay put across unmap/remap.
            memory = AcquireGunyahRingBlob(resourceId, createBlobArgs->size, pageSize,
                                           features.ExternalBlob.enabled());
        } else if (features.ExternalBlob.enabled()) {
            memory = RingBlob::CreateWithShmem(resourceId, createBlobArgs->size);
        } else {
            memory = RingBlob::CreateWithHostMemory(resourceId, createBlobArgs->size, pageSize);
        }
        if (!memory) {
            GFXSTREAM_ERROR("Failed to create blob: failed to create ring blob.");
            return std::nullopt;
        }
        resource.mBlobMemory.emplace(std::move(memory));
    } else if (features.ExternalBlob.enabled()) {
        if (createBlobArgs->blob_mem == STREAM_BLOB_MEM_GUEST &&
            (createBlobArgs->blob_flags & STREAM_BLOB_FLAG_CREATE_GUEST_HANDLE)) {
            // Keyed by resource as well as by blob id. A compositor's scanout buffer is created
            // here by the gbm/gallium winsys on one virtio context and imported into Vulkan on
            // another, and the importer can only name the resource -- the blob id is private to
            // whoever created it. Without this second key the import has nothing to bind to and
            // falls through to a colour buffer the host never created.
#if defined(__linux__) || defined(__ANDROID__)
            GFXSTREAM_DIAG_PRINT("GUESTBLOB-CREATE: resource=%u blobId=%llu size=%llu\n",
                                 resourceId, (unsigned long long)createBlobArgs->blob_id,
                                 (unsigned long long)createBlobArgs->size);
            ExternalObjectManager::get()->addGuestBlobResourceDescriptor(resourceId,
                                                                        (int)handle->os_handle);
#endif
#if defined(__ANDROID__)
            ExternalObjectManager::get()->addBlobDescriptorInfo(
                contextId, createBlobArgs->blob_id, handle->os_handle, handle->handle_type, 0,
                std::nullopt);
#elif defined(__linux__) || defined(__QNX__)
            ManagedDescriptor managedHandle(handle->os_handle);
            ExternalObjectManager::get()->addBlobDescriptorInfo(
                contextId, createBlobArgs->blob_id, std::move(managedHandle), handle->handle_type,
                0, std::nullopt);
#else
            GFXSTREAM_ERROR("Failed to create blob: unimplemented external blob.");
            return std::nullopt;
#endif
        } else {
            if (!descriptorInfoOpt) {
                descriptorInfoOpt = ExternalObjectManager::get()->removeBlobDescriptorInfo(
                    contextId, createBlobArgs->blob_id);
            }
            if (!descriptorInfoOpt) {
                // Memory the host could not re-export registers a host-address MAPPING instead of
                // a descriptor -- vkGetBlobInternal falls back to vkMapMemory + addMapping when
                // the driver refuses to export imported memory. ExternalBlob mode used to reject
                // those outright, which left such memory guest-mappable in no mode at all.
                auto memoryMappingOpt =
                    ExternalObjectManager::get()->removeMapping(contextId, createBlobArgs->blob_id);
                if (memoryMappingOpt) {
                    resource.mBlobMemory.emplace(std::move(*memoryMappingOpt));
                    resource.mId = resourceId;
                    return resource;
                }
                GFXSTREAM_ERROR("Failed to create blob: no external blob descriptor.");
                return std::nullopt;
            }
            resource.mBlobMemory.emplace(
                std::make_shared<BlobDescriptorInfo>(std::move(*descriptorInfoOpt)));
        }
    } else {
        auto memoryMappingOpt =
            ExternalObjectManager::get()->removeMapping(contextId, createBlobArgs->blob_id);
        if (!memoryMappingOpt) {
            GFXSTREAM_ERROR("Failed to create blob: no external blob mapping.");
            return std::nullopt;
        }
        resource.mBlobMemory.emplace(std::move(*memoryMappingOpt));
    }

    return resource;
}

int VirtioGpuResource::Destroy() {
    if (mResourceType == VirtioGpuResourceType::BUFFER) {
        FrameBuffer::getFB()->closeBuffer(mId);
    } else if (mResourceType == VirtioGpuResourceType::COLOR_BUFFER) {
        FrameBuffer::getFB()->closeColorBuffer(mId);
    }
    // Close the descriptor consumed into mBlobMemory exactly once, here, where the resource dies.
    // On Android it is a raw handle, so every ColorBuffer- or Buffer-backed blob used to leak the
    // dma-buf fd dup'd at export time.
    if (mBlobMemory && std::holds_alternative<ExternalMemoryInfo>(*mBlobMemory)) {
        auto& memory = std::get<ExternalMemoryInfo>(*mBlobMemory);
        if (memory) {
            CloseBlobDescriptor(memory->descriptorInfo);
        }
    }
    return 0;
}

int VirtioGpuResource::ImportHandle(const struct stream_renderer_handle* handle,
                                    const struct stream_renderer_import_data* import_data) {
    if (mResourceType != VirtioGpuResourceType::COLOR_BUFFER) {
        GFXSTREAM_ERROR(
            "Failed to ImportResource: importing external handles to existing resources is only "
            "supported for ColorBuffer resources.");
        return -EINVAL;
    }

    auto colorBufferPtr = FrameBuffer::getFB()->findColorBuffer(mId);
    if (!colorBufferPtr) {
        GFXSTREAM_ERROR("Failed to ImportResource: could not find colorBuffer for res_handle: %d",
                        mId);
        return -EINVAL;
    }

    const bool preserveContent =
        (import_data->flags & STREAM_RENDERER_IMPORT_FLAG_PRESERVE_CONTENT);
    bool importSuccess = false;
    switch (handle->handle_type) {
#if GFXSTREAM_ENABLE_HOST_GLES
        case STREAM_HANDLE_TYPE_PLATFORM_EGL_NATIVE_PIXMAP:
            importSuccess = colorBufferPtr->glOpImportEglNativePixmap(
                reinterpret_cast<void*>(handle->os_handle), preserveContent);
            break;
#endif
        default:
            GFXSTREAM_ERROR(
                "Unsupported handle_type: 0x%x, specified for importing to resource: %d",
                handle->handle_type, mId);
            return -EINVAL;
    }

    return (importSuccess ? 0 : -EINVAL);
}

void VirtioGpuResource::AttachIov(struct iovec* iov, uint32_t num_iovs) {
    mIovs.clear();
    mLinear.clear();

    size_t linearSize = 0;
    if (num_iovs) {
        mIovs.reserve(num_iovs);
        for (uint32_t i = 0; i < num_iovs; ++i) {
            mIovs.push_back(iov[i]);
            linearSize += iov[i].iov_len;
        }
    }

    if (linearSize > 0) {
        mLinear.resize(linearSize, 0);
    }
}

void VirtioGpuResource::AttachToContext(VirtioGpuContextId contextId) {
    mAttachedToContexts.insert(contextId);
    mLatestAttachedContext = contextId;
}

void VirtioGpuResource::DetachFromContext(VirtioGpuContextId contextId) {
    mAttachedToContexts.erase(contextId);
    mLatestAttachedContext.reset();
    mHostPipe = nullptr;
}

std::unordered_set<VirtioGpuContextId> VirtioGpuResource::GetAttachedContexts() const {
    return mAttachedToContexts;
}

void VirtioGpuResource::DetachIov() {
    mIovs.clear();
    mLinear.clear();
}

int VirtioGpuResource::Map(void** outAddress, uint64_t* outSize) {
    if (!mBlobMemory) {
        GFXSTREAM_ERROR("Failed to map resource %d: no blob memory to map.", mId);
        return -EINVAL;
    }

    void* hva = nullptr;
    uint64_t hvaSize = 0;

    if (std::holds_alternative<RingBlobMemory>(*mBlobMemory)) {
        auto& memory = std::get<RingBlobMemory>(*mBlobMemory);
        hva = memory->map();
        hvaSize = memory->size();
    } else if (std::holds_alternative<ExternalMemoryMapping>(*mBlobMemory)) {
        if (!mCreateBlobArgs) {
            GFXSTREAM_ERROR("failed to map resource %d: missing args.", mId);
            return -EINVAL;
        }
        auto& memory = std::get<ExternalMemoryMapping>(*mBlobMemory);
        hva = memory.addr;
        hvaSize = mCreateBlobArgs->size;
    } else {
        GFXSTREAM_ERROR("failed to map resource %d: no mappable memory.", mId);
        return -EINVAL;
    }

    if (outAddress) {
        *outAddress = hva;
    }
    if (outSize) {
        *outSize = hvaSize;
    }
    return 0;
}

int VirtioGpuResource::GetInfo(struct stream_renderer_resource_info* outInfo) const {
    if (!mCreateArgs) {
        GFXSTREAM_ERROR("Failed to get info: resource %d missing args.", mId);
        return ENOENT;
    }

    auto formatInfo = VirglFormatInfo(mCreateArgs->format);
    if (!formatInfo) {
        return EINVAL;
    }

    outInfo->drm_fourcc = formatInfo->drm_fourcc;
    outInfo->stride = AlignUp(mCreateArgs->width * formatInfo->bpp, 16U);
    outInfo->virgl_format = mCreateArgs->format;
    outInfo->handle = mCreateArgs->handle;
    outInfo->height = mCreateArgs->height;
    outInfo->width = mCreateArgs->width;
    outInfo->depth = mCreateArgs->depth;
    outInfo->flags = mCreateArgs->flags;
    outInfo->tex_id = 0;
    return 0;
}

int VirtioGpuResource::GetVulkanInfo(struct stream_renderer_vulkan_info* outInfo) const {
    if (!mBlobMemory) {
        return -EINVAL;
    }
    if (!std::holds_alternative<ExternalMemoryInfo>(*mBlobMemory)) {
        return -EINVAL;
    }
    auto& memory = std::get<ExternalMemoryInfo>(*mBlobMemory);
    if (!memory->vulkanInfoOpt) {
        return -EINVAL;
    }
    auto& memoryVulkanInfo = *memory->vulkanInfoOpt;

    outInfo->memory_index = memoryVulkanInfo.memoryIndex;
    memcpy(outInfo->device_id.device_uuid, memoryVulkanInfo.deviceUUID,
           sizeof(outInfo->device_id.device_uuid));
    memcpy(outInfo->device_id.driver_uuid, memoryVulkanInfo.driverUUID,
           sizeof(outInfo->device_id.driver_uuid));
    return 0;
}

int VirtioGpuResource::GetCaching(uint32_t* outCaching) const {
    if (!mBlobMemory) {
        GFXSTREAM_ERROR("failed to get caching for resource %d: no blob memory", mId);
        return -EINVAL;
    }

    if (std::holds_alternative<ExternalMemoryMapping>(*mBlobMemory)) {
        auto& memory = std::get<ExternalMemoryMapping>(*mBlobMemory);
        *outCaching = memory.caching;
        return 0;
    } else if (std::holds_alternative<ExternalMemoryInfo>(*mBlobMemory)) {
        auto& memory = std::get<ExternalMemoryInfo>(*mBlobMemory);
        *outCaching = memory->caching;
        return 0;
    } else if (std::holds_alternative<RingBlobMemory>(*mBlobMemory)) {
        *outCaching = STREAM_RENDERER_MAP_CACHE_CACHED;
        return 0;
    }

    GFXSTREAM_ERROR("failed to get caching for resource %d: unhandled type?", mId);
    return -EINVAL;
}

// Corresponds to Virtio GPU "TransferFromHost" commands and VMM requests to
// copy into display buffers.
int VirtioGpuResource::TransferRead(uint64_t offset, stream_renderer_box* box,
                                    std::optional<std::vector<struct iovec>> iovs) {
    // A blob-backed resource never had iovs attached, so it has no staging buffer yet.
    EnsureLinearAllocated();

    // First, copy from the underlying backend resource to this resource's linear buffer:
    int ret = 0;
    if (mResourceType == VirtioGpuResourceType::BLOB) {
        GFXSTREAM_ERROR("Failed to transfer: unexpected blob.");
        return -EINVAL;
    } else if (mResourceType == VirtioGpuResourceType::PIPE) {
        ret = ReadFromPipeToLinear(offset, box);
    } else if (mResourceType == VirtioGpuResourceType::BUFFER) {
        ret = ReadFromBufferToLinear(offset, box);
    } else if (mResourceType == VirtioGpuResourceType::COLOR_BUFFER) {
        ret = ReadFromColorBufferToLinear(offset, box);
    } else {
        GFXSTREAM_ERROR("Failed to transfer: unhandled resource type.");
        return -EINVAL;
    }
    if (ret != 0) {
        GFXSTREAM_ERROR("Failed to transfer: failed to sync with backend resource.");
        return ret;
    }

    // Second, copy from this resource's linear buffer to the desired iov:
    if (iovs) {
        ret = TransferToIov(offset, box, *iovs);
    } else {
        ret = TransferToIov(offset, box, mIovs);
    }
    if (ret != 0) {
        GFXSTREAM_ERROR("Failed to transfer: failed to copy to iov.");
    }
    return ret;
}

// Corresponds to Virtio GPU "TransferToHost" commands.
int VirtioGpuResource::TransferWrite(uint64_t offset, stream_renderer_box* box,
                                     std::optional<std::vector<struct iovec>> iovs) {
    EnsureLinearAllocated();

    // First, copy from the desired iov to this resource's linear buffer:
    int ret = 0;
    if (iovs) {
        ret = TransferFromIov(offset, box, *iovs);
    } else {
        ret = TransferFromIov(offset, box, mIovs);
    }
    if (ret != 0) {
        GFXSTREAM_ERROR("Failed to transfer: failed to copy from iov.");
        return ret;
    }

    // Second, copy from this resource's linear buffer to the underlying backend resource:
    if (mResourceType == VirtioGpuResourceType::BLOB) {
        GFXSTREAM_ERROR("Failed to transfer: unexpected blob.");
        return -EINVAL;
    } else if (mResourceType == VirtioGpuResourceType::PIPE) {
        return WriteToPipeFromLinear(offset, box);
    } else if (mResourceType == VirtioGpuResourceType::BUFFER) {
        return WriteToBufferFromLinear(offset, box);
    } else if (mResourceType == VirtioGpuResourceType::COLOR_BUFFER) {
        return WriteToColorBufferFromLinear(offset, box);
    } else {
        GFXSTREAM_ERROR("Failed to transfer: unhandled resource type.");
        return -EINVAL;
    }
}

void VirtioGpuResource::EnsureLinearAllocated() {
    if (!mLinear.empty() || !mCreateArgs) {
        return;
    }

    // Must match what ReadFrom*ToLinear() and TransferWithIov() require, or those reject the
    // transfer with "mLinear is too small!" -- which is where this used to end up as a nullptr
    // dereference instead: a blob resource skips AttachIov entirely, so readColorBuffer wrote to
    // mLinear.data() == nullptr and took the host down on the first scanout readback.
    const size_t linearSize =
        GetTransferSize(mCreateArgs->format, mCreateArgs->width, mCreateArgs->height, 0, 0,
                        mCreateArgs->width, mCreateArgs->height);
    if (linearSize == 0) {
        // Unsupported format; leave mLinear empty and let the existing guards report it.
        return;
    }

    mLinear.resize(linearSize, 0);
}

int VirtioGpuResource::ReadFromPipeToLinear(uint64_t offset, stream_renderer_box* box) {
    if (mResourceType != VirtioGpuResourceType::PIPE) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d is not PIPE.", mId);
        return -EINVAL;
    }

    if (!mHostPipe) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d missing PIPE.", mId);
        return -EINVAL;
    }

    size_t requiredSize = box->x + box->w;
    if (mLinear.size() < requiredSize) {
        GFXSTREAM_ERROR("mLinear is too small! size: %zu, required: %zu", mLinear.size(), requiredSize);
        return -EINVAL;
    }

    return mHostPipe->TransferFromHost(mLinear.data() + box->x, box->w);
}

int VirtioGpuResource::WriteToPipeFromLinear(uint64_t offset, stream_renderer_box* box) {
    if (mResourceType != VirtioGpuResourceType::PIPE) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d is not PIPE.", mId);
        return -EINVAL;
    }

    if (!mCreateArgs) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d missing args.", mId);
        return -EINVAL;
    }

    auto hostPipe = mHostPipe;
    if (!mHostPipe) {
        GFXSTREAM_ERROR("No hostPipe");
        return -EINVAL;
    }

    return mHostPipe->TransferToHost(mLinear.data() + box->x, box->w);
}

int VirtioGpuResource::ReadFromBufferToLinear(uint64_t offset, stream_renderer_box* box) {
    if (mResourceType != VirtioGpuResourceType::BUFFER) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d is not BUFFER.", mId);
        return -EINVAL;
    }

    if (!mCreateArgs) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d missing args.", mId);
        return -EINVAL;
    }

    size_t requiredSize = mCreateArgs->width * mCreateArgs->height;
    if (mLinear.size() < requiredSize) {
        GFXSTREAM_ERROR("mLinear is too small! size: %zu, required: %zu", mLinear.size(), requiredSize);
        return -EINVAL;
    }

    FrameBuffer::getFB()->readBuffer(mCreateArgs->handle, 0,
                                     mCreateArgs->width * mCreateArgs->height, mLinear.data());
    return 0;
}

int VirtioGpuResource::WriteToBufferFromLinear(uint64_t offset, stream_renderer_box* box) {
    if (mResourceType != VirtioGpuResourceType::BUFFER) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d is not BUFFER.", mId);
        return -EINVAL;
    }

    if (!mCreateArgs) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d missing args.", mId);
        return -EINVAL;
    }

    FrameBuffer::getFB()->updateBuffer(mCreateArgs->handle, 0,
                                       mCreateArgs->width * mCreateArgs->height, mLinear.data());
    return 0;
}

int VirtioGpuResource::ReadFromColorBufferToLinear(uint64_t offset, stream_renderer_box* box) {
    if (mResourceType != VirtioGpuResourceType::COLOR_BUFFER) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d is not COLOR_BUFFER.", mId);
        return -EINVAL;
    }

    if (!mCreateArgs) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d missing args.", mId);
        return -EINVAL;
    }

    auto formatOpt = ToGfxstreamFormat(mCreateArgs->format);
    if (!formatOpt) {
        GFXSTREAM_ERROR("Failed to transfer: unsupported format %d", mCreateArgs->format);
        return -EINVAL;
    }
    auto format = *formatOpt;

    size_t requiredSize = GetTransferSize(mCreateArgs->format, mCreateArgs->width, mCreateArgs->height,
                                          0, 0, mCreateArgs->width, mCreateArgs->height);
    if (mLinear.size() < requiredSize) {
        GFXSTREAM_ERROR("mLinear is too small! size: %zu, required: %zu", mLinear.size(), requiredSize);
        return -EINVAL;
    }

    // We always xfer the whole thing again from GL
    // since it's fiddly to calc / copy-out subregions
    if (IsYuvFormat(format)) {
        FrameBuffer::getFB()->readColorBufferYUV(mCreateArgs->handle, 0, 0, mCreateArgs->width,
                                                 mCreateArgs->height, mLinear.data(),
                                                 mLinear.size());
    } else {
        FrameBuffer::getFB()->readColorBuffer(mCreateArgs->handle, 0, 0, mCreateArgs->width,
                                              mCreateArgs->height, format, mLinear.data(),
                                              mLinear.size());
    }

    return 0;
}

int VirtioGpuResource::WriteToColorBufferFromLinear(uint64_t offset, stream_renderer_box* box) {
    if (mResourceType != VirtioGpuResourceType::COLOR_BUFFER) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d is not COLOR_BUFFER.", mId);
        return -EINVAL;
    }

    if (!mCreateArgs) {
        GFXSTREAM_ERROR("Failed to transfer: resource %d missing args.", mId);
        return -EINVAL;
    }

    auto formatOpt = ToGfxstreamFormat(mCreateArgs->format);
    if (!formatOpt) {
        GFXSTREAM_ERROR("Failed to transfer: unsupported format %d", mCreateArgs->format);
        return -EINVAL;
    }
    auto format = *formatOpt;

    // We always xfer the whole thing again to GL
    // since it's fiddly to calc / copy-out subregions
    FrameBuffer::getFB()->updateColorBuffer(mCreateArgs->handle, 0, 0, mCreateArgs->width,
                                            mCreateArgs->height, format, mLinear.data());
    return 0;
}

int VirtioGpuResource::TransferToIov(uint64_t offset, const stream_renderer_box* box,
                                     std::optional<std::vector<struct iovec>> iovs) {
    if (iovs) {
        return TransferWithIov(offset, box, *iovs, TransferDirection::LINEAR_TO_IOV);
    } else {
        return TransferWithIov(offset, box, mIovs, TransferDirection::LINEAR_TO_IOV);
    }
}

int VirtioGpuResource::TransferFromIov(uint64_t offset, const stream_renderer_box* box,
                                       std::optional<std::vector<struct iovec>> iovs) {
    if (iovs) {
        return TransferWithIov(offset, box, *iovs, TransferDirection::IOV_TO_LINEAR);
    } else {
        return TransferWithIov(offset, box, mIovs, TransferDirection::IOV_TO_LINEAR);
    }
}

int VirtioGpuResource::TransferWithIov(uint64_t offset, const stream_renderer_box* box,
                                       const std::vector<struct iovec>& iovs,
                                       TransferDirection direction) {
    if (!mCreateArgs) {
        GFXSTREAM_ERROR("failed to transfer: missing resource args.");
        return -EINVAL;
    }
    if (box->x > mCreateArgs->width || box->y > mCreateArgs->height) {
        GFXSTREAM_ERROR("failed to transfer: box out of range of resource");
        return -EINVAL;
    }
    if (box->w == 0U || box->h == 0U) {
        GFXSTREAM_ERROR("failed to transfer: empty transfer");
        return -EINVAL;
    }
    if (box->x + box->w > mCreateArgs->width) {
        GFXSTREAM_ERROR("failed to transfer: box overflows resource width");
        return -EINVAL;
    }

    size_t linearBase =
        GetTransferOffset(mCreateArgs->format, mCreateArgs->width, mCreateArgs->height,
                          box->x, box->y, box->w, box->h);
    size_t start = linearBase;
    size_t length =
        GetTransferSize(mCreateArgs->format, mCreateArgs->width, mCreateArgs->height,
                                       box->x, box->y, box->w, box->h);
    size_t end = start + length;

    if (start == end) {
        GFXSTREAM_ERROR("failed to transfer: nothing to transfer");
        return -EINVAL;
    }

    if (end > mLinear.size()) {
        GFXSTREAM_ERROR("failed to transfer: start + length overflows!");
        return -EINVAL;
    }

    uint32_t iovIndex = 0;
    size_t iovOffset = 0;
    size_t written = 0;
    char* linear = static_cast<char*>(mLinear.data());

    while (written < length) {
        if (iovIndex >= iovs.size()) {
            GFXSTREAM_ERROR("failed to transfer: write request overflowed iovs");
            return -EINVAL;
        }

        const char* iovBase_const = static_cast<const char*>(iovs[iovIndex].iov_base);
        char* iovBase = static_cast<char*>(iovs[iovIndex].iov_base);
        size_t iovLen = iovs[iovIndex].iov_len;
        size_t iovOffsetEnd = iovOffset + iovLen;

        auto lower_intersect = std::max(iovOffset, start);
        auto upper_intersect = std::min(iovOffsetEnd, end);
        if (lower_intersect < upper_intersect) {
            size_t toWrite = upper_intersect - lower_intersect;
            switch (direction) {
                case TransferDirection::IOV_TO_LINEAR:
                    memcpy(linear + lower_intersect, iovBase_const + lower_intersect - iovOffset,
                           toWrite);
                    break;
                case TransferDirection::LINEAR_TO_IOV:
                    memcpy(iovBase + lower_intersect - iovOffset, linear + lower_intersect,
                           toWrite);
                    break;
                default:
                    GFXSTREAM_ERROR("failed to transfer: invalid synchronization dir");
                    return -EINVAL;
            }
            written += toWrite;
        }
        ++iovIndex;
        iovOffset += iovLen;
    }

    return 0;
}

int VirtioGpuResource::ExportBlob(struct stream_renderer_handle* outHandle) {
    // For non-blob COLOR_BUFFER resources (CREATE_3D path), there is no mBlobMemory
    // set. Attempt to export the ColorBuffer's external memory.
    if (!mBlobMemory && mResourceType == VirtioGpuResourceType::COLOR_BUFFER) {
        auto descriptorInfoOpt = FrameBuffer::getFB()->exportColorBuffer(mId);
        if (!descriptorInfoOpt) {
            return -EINVAL;
        }
        return fillExportHandle(outHandle, descriptorInfoOpt->descriptorInfo);
    }

    if (!mBlobMemory) {
        return -EINVAL;
    }

    if (std::holds_alternative<RingBlobMemory>(*mBlobMemory)) {
        auto& memory = std::get<RingBlobMemory>(*mBlobMemory);
        // A pool-resident ring blob: hand the VMM a dup of the pool memfd plus the offset so it
        // maps the pool GPA directly, with no runtime SHARE. This has to come before the
        // isExportable() check -- a pool ring blob is backed by borrowed memory and is not
        // exportable in the shmem sense.
        if (memory->poolOffset() >= 0) {
            auto* hvPool = vk::HostVisiblePool::get();
            // Returning 0 with os_handle -1 would be indistinguishable from a real handle except
            // by checking for a negative fd, and the VMM turns that into an error naming no cause,
            // leaving the display quietly falling back to a copy of memory nothing writes.
            if (!hvPool) {
                GFXSTREAM_ERROR(
                    "failed to export blob for resource %u: pool-resident ring blob at offset %lld "
                    "but there is no host-visible pool",
                    mId, (long long)memory->poolOffset());
                return -EINVAL;
            }
            int poolFd = dup(hvPool->memfd());
            if (poolFd < 0) {
                GFXSTREAM_ERROR(
                    "failed to export blob for resource %u: dup of the pool memfd failed", mId);
                return -EINVAL;
            }
            outHandle->os_handle = (int64_t)poolFd;
            outHandle->handle_type = STREAM_HANDLE_TYPE_MEM_POOL;
            outHandle->pool_offset = (uint64_t)memory->poolOffset();
            return 0;
        }
        if (!memory->isExportable()) {
            return -EINVAL;
        }

        // Handle ownership transferred to VMM, Gfxstream keeps the mapping. A recycled ring blob
        // is exported once per blob resource that reuses it, so hand out a dup instead of its only
        // handle when recycling is on.
        gfxstream::base::SharedMemory::handle_type handle =
            ShouldPinRingBlobsForGunyah() ? memory->dupHandle() : memory->releaseHandle();
#ifdef _WIN32
        outHandle->os_handle = static_cast<int64_t>(reinterpret_cast<intptr_t>(handle));
#else
        outHandle->os_handle = static_cast<int64_t>(handle);
#endif
        outHandle->handle_type = STREAM_HANDLE_TYPE_MEM_SHM;
        return 0;
    } else if (std::holds_alternative<ExternalMemoryInfo>(*mBlobMemory)) {
        auto& memory = std::get<ExternalMemoryInfo>(*mBlobMemory);
        // A GpuPool-resident blob: hand the VMM a dup of the pool memfd plus the offset, marked
        // MEM_POOL, so it maps the pool GPA directly instead of runtime-SHARE'ing this blob.
        if (memory->poolOffset >= 0) {
#ifdef __ANDROID__
            int rawDescriptor = dup(static_cast<int>(memory->descriptorInfo.handle));
#else
            auto rawDescriptorOpt = memory->descriptorInfo.descriptor.release();
            int rawDescriptor = rawDescriptorOpt ? static_cast<int>(*rawDescriptorOpt) : -1;
#endif
            // Reporting the failure rather than a success carrying no handle: a pool-resident blob
            // whose descriptor is not a live fd is a real condition -- guest-allocated memory
            // reaches the host as pages, not as something dup() can copy -- and returning 0 with
            // os_handle -1 leaves the caller unable to tell it from a real handle except by
            // checking for a negative fd, which the VMM turns into an error that names no cause.
            if (rawDescriptor < 0) {
                GFXSTREAM_ERROR(
                    "failed to export blob for resource %u: pool-resident at offset %lld but its "
                    "descriptor (type %u) is not exportable",
                    mId, (long long)memory->poolOffset, memory->descriptorInfo.streamHandleType);
                return -EINVAL;
            }
            outHandle->os_handle = static_cast<int64_t>(rawDescriptor);
            outHandle->handle_type = STREAM_HANDLE_TYPE_MEM_POOL;
            outHandle->pool_offset = static_cast<uint64_t>(memory->poolOffset);
            return 0;
        }
        int ret = fillExportHandle(outHandle, memory->descriptorInfo);
        if (ret != 0) {
            GFXSTREAM_ERROR("failed to export blob for resource %u: failed to get raw handle.",
                            mId);
        }
        return ret;
    }

    return -EINVAL;
}

void VirtioGpuResource::ReturnRingBlobToGunyahPool() {
    if (!ShouldPinRingBlobsForGunyah() || !mBlobMemory) {
        return;
    }
    if (!std::holds_alternative<RingBlobMemory>(*mBlobMemory)) {
        return;
    }
    // Back to the recycle pool -- kept alive and reusable by a later same-size blob -- instead of
    // being freed when this resource is dropped.
    ReleaseGunyahRingBlob(std::get<RingBlobMemory>(*mBlobMemory));
}

std::shared_ptr<RingBlob> VirtioGpuResource::ShareRingBlob() {
    if (!mBlobMemory) {
        return nullptr;
    }
    if (!std::holds_alternative<RingBlobMemory>(*mBlobMemory)) {
        return nullptr;
    }
    return std::get<RingBlobMemory>(*mBlobMemory);
}

#ifdef GFXSTREAM_BUILD_WITH_SNAPSHOT_FRONTEND_SUPPORT

std::optional<VirtioGpuResourceSnapshot> VirtioGpuResource::Snapshot() const {
    VirtioGpuResourceSnapshot resourceSnapshot;
    resourceSnapshot.set_id(mId);
    resourceSnapshot.set_type(static_cast<::gfxstream::host::snapshot::VirtioGpuResourceType>(mResourceType));

    if (mCreateArgs) {
        VirtioGpuResourceCreateArgs* snapshotCreateArgs = resourceSnapshot.mutable_create_args();
        snapshotCreateArgs->set_id(mCreateArgs->handle);
        snapshotCreateArgs->set_target(mCreateArgs->target);
        snapshotCreateArgs->set_format(mCreateArgs->format);
        snapshotCreateArgs->set_bind(mCreateArgs->bind);
        snapshotCreateArgs->set_width(mCreateArgs->width);
        snapshotCreateArgs->set_height(mCreateArgs->height);
        snapshotCreateArgs->set_depth(mCreateArgs->depth);
        snapshotCreateArgs->set_array_size(mCreateArgs->array_size);
        snapshotCreateArgs->set_last_level(mCreateArgs->last_level);
        snapshotCreateArgs->set_nr_samples(mCreateArgs->nr_samples);
        snapshotCreateArgs->set_flags(mCreateArgs->flags);
    }

    if (mCreateBlobArgs) {
        auto* snapshotCreateArgs = resourceSnapshot.mutable_create_blob_args();
        snapshotCreateArgs->set_mem(mCreateBlobArgs->blob_mem);
        snapshotCreateArgs->set_flags(mCreateBlobArgs->blob_flags);
        snapshotCreateArgs->set_id(mCreateBlobArgs->blob_id);
        snapshotCreateArgs->set_size(mCreateBlobArgs->size);
    }

    if (mBlobMemory) {
        if (std::holds_alternative<RingBlobMemory>(*mBlobMemory)) {
            auto& memory = std::get<RingBlobMemory>(*mBlobMemory);

            auto snapshotRingBlobOpt = memory->Snapshot();
            if (!snapshotRingBlobOpt) {
                GFXSTREAM_ERROR("Failed to snapshot ring blob for resource %d.", mId);
                return std::nullopt;
            }
            resourceSnapshot.mutable_ring_blob()->Swap(&*snapshotRingBlobOpt);
        } else if (std::holds_alternative<ExternalMemoryInfo>(*mBlobMemory)) {
            if (!mLatestAttachedContext) {
                GFXSTREAM_ERROR("Failed to snapshot resource %d: missing blob context?", mId);
                return std::nullopt;
            }
            if (!mCreateBlobArgs) {
                GFXSTREAM_ERROR("Failed to snapshot resource %d: missing blob args?", mId);
                return std::nullopt;
            }
            auto snapshotDescriptorInfo = resourceSnapshot.mutable_external_memory_descriptor();
            snapshotDescriptorInfo->set_context_id(*mLatestAttachedContext);
            snapshotDescriptorInfo->set_blob_id(mCreateBlobArgs->blob_id);
        } else if (std::holds_alternative<ExternalMemoryMapping>(*mBlobMemory)) {
            if (!mLatestAttachedContext) {
                GFXSTREAM_ERROR("Failed to snapshot resource %d: missing blob context?", mId);
                return std::nullopt;
            }
            if (!mCreateBlobArgs) {
                GFXSTREAM_ERROR("Failed to snapshot resource %d: missing blob args?", mId);
                return std::nullopt;
            }
            auto snapshotDescriptorInfo = resourceSnapshot.mutable_external_memory_mapping();
            snapshotDescriptorInfo->set_context_id(*mLatestAttachedContext);
            snapshotDescriptorInfo->set_blob_id(mCreateBlobArgs->blob_id);
        }
    }

    if (mLatestAttachedContext) {
        resourceSnapshot.set_latest_attached_context(*mLatestAttachedContext);
    }

    resourceSnapshot.mutable_attached_contexts()->Add(mAttachedToContexts.begin(),
                                                      mAttachedToContexts.end());

    return resourceSnapshot;
}

/*static*/ std::optional<VirtioGpuResource> VirtioGpuResource::Restore(
    const VirtioGpuResourceSnapshot& resourceSnapshot) {
    VirtioGpuResource resource = {};
    resource.mId = resourceSnapshot.id();
    resource.mResourceType = static_cast<VirtioGpuResourceType>(resourceSnapshot.type());

    if (resourceSnapshot.has_create_args()) {
        const auto& createArgsSnapshot = resourceSnapshot.create_args();
        resource.mCreateArgs = {
            .handle = createArgsSnapshot.id(),
            .target = createArgsSnapshot.target(),
            .format = createArgsSnapshot.format(),
            .bind = createArgsSnapshot.bind(),
            .width = createArgsSnapshot.width(),
            .height = createArgsSnapshot.height(),
            .depth = createArgsSnapshot.depth(),
            .array_size = createArgsSnapshot.array_size(),
            .last_level = createArgsSnapshot.last_level(),
            .nr_samples = createArgsSnapshot.nr_samples(),
            .flags = createArgsSnapshot.flags(),
        };
    }

    if (resourceSnapshot.has_create_blob_args()) {
        const auto& createArgsSnapshot = resourceSnapshot.create_blob_args();
        resource.mCreateBlobArgs = {
            .blob_mem = createArgsSnapshot.mem(),
            .blob_flags = createArgsSnapshot.flags(),
            .blob_id = createArgsSnapshot.id(),
            .size = createArgsSnapshot.size(),
        };
    }

    if (resourceSnapshot.has_ring_blob()) {
        auto resourceRingBlobOpt = RingBlob::Restore(resourceSnapshot.ring_blob());
        if (!resourceRingBlobOpt) {
            GFXSTREAM_ERROR("Failed to restore ring blob for resource %d", resource.mId);
            return std::nullopt;
        }
        resource.mBlobMemory.emplace(std::move(*resourceRingBlobOpt));
    } else if (resourceSnapshot.has_external_memory_descriptor()) {
        const auto& snapshotDescriptorInfo = resourceSnapshot.external_memory_descriptor();

        auto descriptorInfoOpt = ExternalObjectManager::get()->removeBlobDescriptorInfo(
            snapshotDescriptorInfo.context_id(), snapshotDescriptorInfo.blob_id());
        if (!descriptorInfoOpt) {
            GFXSTREAM_ERROR("Failed to restore resource: failed to find blob descriptor info.");
            return std::nullopt;
        }

        resource.mBlobMemory.emplace(
            std::make_shared<BlobDescriptorInfo>(std::move(*descriptorInfoOpt)));
    } else if (resourceSnapshot.has_external_memory_mapping()) {
        const auto& snapshotDescriptorInfo = resourceSnapshot.external_memory_mapping();

        auto memoryMappingOpt = ExternalObjectManager::get()->removeMapping(
            snapshotDescriptorInfo.context_id(), snapshotDescriptorInfo.blob_id());
        if (!memoryMappingOpt) {
            GFXSTREAM_ERROR("Failed to restore resource: failed to find mapping info.");
            return std::nullopt;
        }
        resource.mBlobMemory.emplace(std::move(*memoryMappingOpt));
    }

    if (resourceSnapshot.has_latest_attached_context()) {
        resource.mLatestAttachedContext = resourceSnapshot.latest_attached_context();
    }

    resource.mAttachedToContexts.insert(resourceSnapshot.attached_contexts().begin(),
                                        resourceSnapshot.attached_contexts().end());

    return resource;
}

#endif  // #ifdef GFXSTREAM_BUILD_WITH_SNAPSHOT_FRONTEND_SUPPORT

}  // namespace host
}  // namespace gfxstream
