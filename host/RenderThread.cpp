/*
* Copyright (C) 2011 The Android Open Source Project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "RenderThread.h"

#include <assert.h>
#include <cstring>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <unordered_map>

#include "ChannelStream.h"
#include "FrameBuffer.h"
#include "ReadBuffer.h"
#include "RenderChannelImpl.h"
#if GFXSTREAM_ENABLE_HOST_GLES
#include "RenderControl.h"
#endif
#include "RenderThreadInfo.h"
#include "VkDecoderContext.h"
#include "gfxstream/host/ChecksumCalculatorThreadInfo.h"
#include "gfxstream/HealthMonitor.h"
#include "gfxstream/Metrics.h"
#include "gfxstream/common/logging.h"
#include "gfxstream/host/stream_utils.h"
#include "gfxstream/synchronization/Lock.h"
#include "gfxstream/synchronization/MessageChannel.h"
#include "gfxstream/system/System.h"
#include "vulkan/VkCommonOperations.h"

namespace gfxstream {

using gfxstream::base::AutoLock;
using gfxstream::base::EventHangMetadata;
using gfxstream::host::GfxApiLogger;
using vk::VkDecoderContext;

struct RenderThread::SnapshotObjects {
    RenderThreadInfo* threadInfo;
    ChecksumCalculator* checksumCalc;
    ChannelStream* channelStream;
    RingStream* ringStream;
    ReadBuffer* readBuffer;
};

static bool getBenchmarkEnabledFromEnv() {
    auto threadEnabled = gfxstream::base::getEnvironmentVariable("ANDROID_EMUGL_RENDERTHREAD_STATS");
    if (threadEnabled == "1") return true;
    return false;
}

// Start with a smaller buffer to not waste memory on a low-used render threads.
static constexpr int kStreamBufferSize = 128 * 1024;

// Requires this many threads on the system available to run unlimited.
static constexpr int kMinThreadsToRunUnlimited = 5;

// A thread run limiter that limits render threads to run one slice at a time.
static gfxstream::base::Lock sThreadRunLimiter;

RenderThread::RenderThread(RenderChannelImpl* channel,
                           gfxstream::Stream* load,
                           uint32_t virtioGpuContextId)
    : gfxstream::base::Thread(gfxstream::base::ThreadFlags::MaskSignals, 2 * 1024 * 1024,
                            "RenderThread"),
      mChannel(channel),
      mRunInLimitedMode(gfxstream::base::getCpuCoreCount() < kMinThreadsToRunUnlimited),
      mContextId(virtioGpuContextId)
{
    if (load) {
        const bool success = load->getByte();
        if (success) {
            mStream.emplace(0);
            loadStream(load, &*mStream);
            mState = SnapshotState::StartLoading;
        } else {
            mFinished.store(true, std::memory_order_relaxed);
        }
    }
}

RenderThread::RenderThread(const AsgConsumerCreateInfo& info, Stream* load)
    : gfxstream::base::Thread(gfxstream::base::ThreadFlags::MaskSignals, 2 * 1024 * 1024,
                              info.virtioGpuContextName ? *info.virtioGpuContextName : ""),
      mRingStream(new RingStream(info, kStreamBufferSize)),
      mContextId(info.virtioGpuContextId ? *info.virtioGpuContextId : 0),
      mCapsetId(info.virtioGpuCapsetId ? *info.virtioGpuCapsetId : 0) {
    if (load) {
        const bool success = load->getByte();
        if (success) {
            mStream.emplace(0);
            loadStream(load, &*mStream);
            mState = SnapshotState::StartLoading;
        } else {
            mFinished.store(true, std::memory_order_relaxed);
        }
    }
}

// Note: the RenderThread destructor might be called from a different thread
// than from RenderThread::main() so thread specific cleanup likely belongs at
// the end of RenderThread::main().
RenderThread::~RenderThread() = default;

void RenderThread::pausePreSnapshot() {
    AutoLock lock(mLock);
    assert(mState == SnapshotState::Empty);
    mStream.emplace();
    mState = SnapshotState::StartSaving;
    if (mRingStream) {
        mRingStream->pausePreSnapshot();
        // mSnapshotSignal.broadcastAndUnlock(&lock);
    }
    if (mChannel) {
        mChannel->pausePreSnapshot();
        mSnapshotSignal.broadcastAndUnlock(&lock);
    }
}

void RenderThread::resume() {
    AutoLock lock(mLock);
    // This function can be called for a thread from pre-snapshot loading
    // state; it doesn't need to do anything.
    if (mState == SnapshotState::Empty) {
        return;
    }
    if (mRingStream) mRingStream->resume();
    waitForSnapshotCompletion(&lock);

    mNeedReloadProcessResources = true;
    mStream.reset();
    mState = SnapshotState::Empty;
    if (mChannel) mChannel->resume();
    if (mRingStream) mRingStream->resume();
    mSnapshotSignal.broadcastAndUnlock(&lock);
}

void RenderThread::save(gfxstream::Stream* stream) {
    bool success;
    {
        AutoLock lock(mLock);
        assert(mState == SnapshotState::StartSaving ||
               mState == SnapshotState::InProgress ||
               mState == SnapshotState::Finished);
        waitForSnapshotCompletion(&lock);
        success = mState == SnapshotState::Finished;
    }

    if (success) {
        assert(mStream);
        stream->putByte(1);
        saveStream(stream, *mStream);
    } else {
        stream->putByte(0);
    }
}

void RenderThread::waitForSnapshotCompletion(AutoLock* lock) {
    while (mState != SnapshotState::Finished &&
           !mFinished.load(std::memory_order_relaxed)) {
        mSnapshotSignal.wait(lock);
    }
}

bool RenderThread::isPausedForSnapshotLocked() const { return mState != SnapshotState::Empty; }

bool RenderThread::doSnapshotOp(const SnapshotObjects& objects, SnapshotState expectedState,
                                std::function<void()> op) {
    AutoLock lock(mLock);

    if (mState != expectedState) {
        return false;
    }
    mState = SnapshotState::InProgress;
    mSnapshotSignal.broadcastAndUnlock(&lock);

    op();

    lock.lock();

    mState = SnapshotState::Finished;
    mSnapshotSignal.broadcast();

    // Only return after we're allowed to proceed.
    while (isPausedForSnapshotLocked()) {
        mSnapshotSignal.wait(&lock);
    }

    return true;
}

bool RenderThread::loadSnapshot(const SnapshotObjects& objects) {
    return doSnapshotOp(objects, SnapshotState::StartLoading, [this, &objects] {
        objects.readBuffer->onLoad(&*mStream);
        if (objects.channelStream) objects.channelStream->load(&*mStream);
        if (objects.ringStream) objects.ringStream->load(&*mStream);
        objects.checksumCalc->load(&*mStream);
        objects.threadInfo->onLoad(&*mStream);
    });
}

bool RenderThread::saveSnapshot(const SnapshotObjects& objects) {
    return doSnapshotOp(objects, SnapshotState::StartSaving, [this, &objects] {
        objects.readBuffer->onSave(&*mStream);
        if (objects.channelStream) objects.channelStream->save(&*mStream);
        if (objects.ringStream) objects.ringStream->save(&*mStream);
        objects.checksumCalc->save(&*mStream);
        objects.threadInfo->onSave(&*mStream);
    });
}

void RenderThread::waitForFinished() {
    AutoLock lock(mLock);
    while (!mFinished.load(std::memory_order_relaxed)) {
        mFinishedSignal.wait(&lock);
    }
}

void RenderThread::sendExitSignal() {
    AutoLock lock(mLock);
    if (!mFinished.load(std::memory_order_relaxed)) {
        GFXSTREAM_FATAL("RenderThread exit signal sent before finished");
    }
    mCanExit.store(true, std::memory_order_relaxed);
    mExitSignal.broadcastAndUnlock(&lock);
}

void RenderThread::addressSpaceGraphicsReloadRingConfig() {
    if (mRingStream) {
        mRingStream->reloadRingConfig();
    }
}

void RenderThread::setFinished() {
    // Make sure it never happens that we wait forever for the thread to
    // save to snapshot while it was not even going to.
    {
        AutoLock lock(mLock);
        mFinished.store(true, std::memory_order_relaxed);
        if (mState != SnapshotState::Empty) {
            mSnapshotSignal.broadcastAndUnlock(&lock);
        }
    }
    {
        AutoLock lock(mLock);
        mFinishedSignal.broadcastAndUnlock(&lock);
    }
}

void RenderThread::waitForExitSignal() {
    AutoLock lock(mLock);
    GFXSTREAM_DEBUG("Waiting for exit signal RenderThread @%p", this);
    while (!mCanExit.load(std::memory_order_relaxed)) {
        mExitSignal.wait(&lock);
    }
}

// The two opcode ranges the decoders use, plus the renderControl range below them. Anything
// outside all three at a packet boundary means the stream is not where it thinks it is.
static bool plausibleOpcode(uint32_t word) {
    return (word >= 20000 && word < 30000) || (word >= 200000000 && word < 300000000) ||
           (word >= 1000 && word < 20000);
}

intptr_t RenderThread::main() {
    if (mFinished.load(std::memory_order_relaxed)) {
        GFXSTREAM_ERROR("Error: fail loading a RenderThread @%p", this);
        return 0;
    }

    std::unique_ptr<RenderThreadInfo> tInfo = std::make_unique<RenderThreadInfo>();
    fprintf(stderr, "RT-ENTER: thread=%p ring=%d\n", this, mRingStream ? 1 : 0);
    ChecksumCalculatorThreadInfo tChecksumInfo;
    ChecksumCalculator& checksumCalc = tChecksumInfo.get();
    bool needRestoreFromSnapshot = false;

    //
    // initialize decoders
#if GFXSTREAM_ENABLE_HOST_GLES
    if (FrameBuffer::getFB()->hasEmulationGl()) {
        tInfo->initGl();
    }

    initRenderControlContext(&(tInfo->m_rcDec));
#endif

    if (!mChannel && !mRingStream) {
        GFXSTREAM_DEBUG("Exited a loader RenderThread @%p", this);
        mFinished.store(true, std::memory_order_relaxed);
        return 0;
    }

    ChannelStream stream(mChannel, RenderChannel::Buffer::kSmallSize);
    IOStream* ioStream =
        mChannel ? (IOStream*)&stream : (IOStream*)mRingStream.get();

    ReadBuffer readBuf(kStreamBufferSize);
    if (mRingStream) {
        readBuf.setNeededFreeTailSize(0);
    }

    const SnapshotObjects snapshotObjects = {
        tInfo.get(), &checksumCalc, &stream, mRingStream.get(), &readBuf,
    };

    // Framebuffer initialization is asynchronous, so we need to make sure
    // it's completely initialized before running any GL commands.
    FrameBuffer::waitUntilInitialized();

    if (FrameBuffer::getFB()->hasEmulationVk()) {
        tInfo->m_vkInfo.emplace();
    }

    // This is the only place where we try loading from snapshot.
    // But the context bind / restoration will be delayed after receiving
    // the first GL command.
    if (loadSnapshot(snapshotObjects)) {
        GFXSTREAM_DEBUG("Loaded RenderThread @%p from snapshot", this);
        needRestoreFromSnapshot = true;
    } else {
        // Not loading from a snapshot: continue regular startup, read
        // the |flags|.
        // Accumulate. read() is "up to n bytes", so re-reading into &flags from the start after
        // a short read throws away what it already took out of the stream and leaves the decoder
        // at an arbitrary offset inside the first packet -- which is one of the two misframings
        // seen on this path (the other being these four bytes not consumed at all). Neither is
        // repairable downstream once the bytes are gone.
        uint32_t flags = 0;
        size_t got = 0;
        size_t flagsHave = 0;
        while (flagsHave < sizeof(flags)) {
            got = ioStream->read(reinterpret_cast<char*>(&flags) + flagsHave,
                                 sizeof(flags) - flagsHave);
            if (got > 0) {
                flagsHave += got;
                continue;
            }
            // Stream read may fail because of a pending snapshot.
            if (!saveSnapshot(snapshotObjects)) {
                // Giving up here means this thread never decodes anything and never replies, so
                // every guest thread that ever asks it a synchronous question waits forever --
                // with a drained ring and no error on either side. Say so, and say what the read
                // returned, because the guest cannot tell this apart from a host that is merely
                // slow.
                GFXSTREAM_ERROR(
                    "RTEXIT-EARLY: ctx=%u gave up on the opening handshake: read returned %zu, "
                    "have %zu of %zu bytes; this thread will never answer",
                    mContextId, got, flagsHave, sizeof(flags));
                setFinished();
                tInfo.reset();
                waitForExitSignal();
                fprintf(stderr, "RT-EXIT-EARLY: thread=%p\n", this);
                return 0;
            }
        }

        // |flags| used to mean something, now they're not used.
        (void)flags;
    }

    int stats_totalBytes = 0;
    uint64_t stats_progressTimeUs = 0;
    auto stats_t0 = gfxstream::base::getHighResTimeUs() / 1000;
    bool benchmarkEnabled = getBenchmarkEnabledFromEnv();

    GfxApiLogger gfxLogger;
    auto& metricsLogger = FrameBuffer::getFB()->getMetricsLogger();

    const ProcessResources* processResources = nullptr;
    bool anyProgress = false;
    uint32_t headerOpcode = 0;
    while (true) {
        // Let's make sure we read enough data for at least some processing.
        uint32_t packetSize;
        if (readBuf.validData() >= 8) {
            // We know that packet size is the second uint32_t from the start.
            std::memcpy(&packetSize, readBuf.buf() + 4, sizeof(uint32_t));
            std::memcpy(&headerOpcode, readBuf.buf(), sizeof(uint32_t));
            // Resynchronise once, on the first header of the stream. Every connection opens
            // with a four-byte clientFlags word that the guest sends unconditionally and that
            // nothing on this side reads; whether it reaches this decoder is a race with the ring
            // being initialised when the consumer is created. Usually it is discarded and the
            // stream starts on a packet boundary. When it is not, this decoder reads zero as an
            // opcode and the next packet's opcode as a length, waits for a body of some twenty
            // thousand bytes that will never come, and the guest thread that made the call waits
            // for its reply forever -- with an empty ring and no error on either side. That is a
            // desktop coming up whole, or as a bare wallpaper, or not at all, from one boot to the
            // next.
            //
            // Only the first header of a stream is eligible, and only when the word that follows
            // is itself a valid opcode, so a healthy stream can never take this path -- and past
            // that first header this costs nothing at all.
            if (readBuf.validData() >= 8) {
                // Every packet boundary, not just the first. A stream can pick up a stray
                // four-byte clientFlags word at any point -- the guest writes one per stream it
                // opens, and a decoder that is handed one mid-stream reads it as an opcode and
                // the real opcode as a length, then waits for a twenty-thousand-byte body that
                // will never come. Latching this check to the first header meant eleven such
                // stalls in one boot went unrepaired, which is a KDE session that never draws.
                //
                // Safe to run at every boundary: a healthy header always begins with a valid
                // opcode, so the repair below cannot trigger on one. Zero in particular is never
                // an opcode.
                const char* who = tInfo->m_processName ? tInfo->m_processName->c_str() : "null";
                if (plausibleOpcode(headerOpcode)) {
                    if (!mResyncChecked) {
                        mResyncChecked = true;
                        fprintf(stderr,
                                "STREAM-FIRST: process=%s opcode=%u len=%u valid=%u ring=%d\n",
                                who, headerOpcode, packetSize, (unsigned)readBuf.validData(),
                                mRingStream ? 1 : 0);
                    }
                } else if (plausibleOpcode(packetSize)) {
                    // What is being read as an opcode is the stray word; what is being read as a
                    // length is the real opcode. Drop four bytes and the stream is back on a
                    // packet boundary.
                    ++mResyncCount;
                    if (mResyncCount <= 3 || (mResyncCount % 64) == 0) {
                        // The opcode that was decoded immediately before this: a stray word at
                        // a packet boundary is most likely the tail of the packet in front of it,
                        // left behind because the guest and the host disagree about that packet's
                        // length. Naming it is how that disagreement gets found.
                        fprintf(stderr,
                                "STREAM-RESYNC: process=%s dropped a leading %u before opcode %u "
                                "(after opcode %u, occurrence %llu)\n",
                                who, headerOpcode, packetSize,
                                mRingStream ? mRingStream->lastDecoded() : 0,
                                (unsigned long long)mResyncCount);
                    }
                    readBuf.consume(sizeof(uint32_t));
                    mResynced = true;
                    anyProgress = true;
                    continue;
                } else if (readBuf.validData() >= 12 && !mMisalignReported) {
                    // Neither word is an opcode and more data will not help. Nothing here can
                    // repair that, but say so rather than let it present as a host that never
                    // answers.
                    mMisalignReported = true;
                    fprintf(stderr,
                            "STREAM-MISALIGN: process=%s opens with %u (len %u), which is not an "
                            "opcode, and the word after it is not one either\n",
                            who, headerOpcode, packetSize);
                }
            }
            if (!packetSize) {
                // Emulator will get live-stuck here if packet size is read to be zero;
                // crash right away so we can see these events.
                // emugl::emugl_crash_reporter(
                //     "Guest should never send a size-0 GL packet\n");
            }
        } else {
            // Read enough data to at least be able to get the packet size next
            // time.
            packetSize = 8;
        }
        if (!anyProgress) {
            // A whole packet already present and nothing decoded it is not the case this rule
            // was written for. Asking for one more byte then converts "I am holding a complete
            // packet the decoder declined" into "I am waiting for a byte the guest will never
            // send", because the guest is itself waiting for the reply to that very packet --
            // a park at `have = want - 1` with a correct opcode, and no error on either side.
            if (readBuf.validData() >= 8 && packetSize >= 8 &&
                readBuf.validData() >= packetSize && !mDecodeStallReported) {
                mDecodeStallReported = true;
                fprintf(stderr,
                        "DECODE-STALL: whole packet present and undecoded: opcode=%u len=%u "
                        "valid=%u ring=%d resynced=%d\n",
                        headerOpcode, packetSize, (unsigned)readBuf.validData(),
                        mRingStream ? 1 : 0, (int)mResynced);
            }
            // If we didn't make any progress last time, then make sure we read at least one
            // extra byte.
            packetSize = std::max(packetSize, static_cast<uint32_t>(readBuf.validData() + 1));
        }
        int stat = 0;
        if (packetSize > readBuf.validData()) {
            // Hand the wait to the stream before entering it. Everything below this line is
            // unreachable while it blocks, so this is the only chance to record what it is for.
            if (mRingStream) {
                // Always, including the case below eight bytes. Leaving the previous packet's
                // values in place made every park after a completed packet look like a decoder
                // stranded mid-packet -- the report fired on a healthy stream waiting for its
                // next header, which is the most ordinary thing a stream does.
                if (readBuf.validData() >= 8) {
                    uint32_t pendingOpcode;
                    std::memcpy(&pendingOpcode, readBuf.buf(), sizeof(uint32_t));
                    mRingStream->setPendingRead(pendingOpcode, packetSize,
                                                static_cast<uint32_t>(readBuf.validData()));
                } else {
                    mRingStream->setPendingRead(0, 0, 0);
                }
            }
            stat = readBuf.getData(ioStream, packetSize);
            if (stat <= 0) {
                if (saveSnapshot(snapshotObjects)) {
                    continue;
                } else {
                    break;
                }
            } else if (needRestoreFromSnapshot) {
                // If we're using RingStream that might load before FrameBuffer
                // restores the contexts from the handles, so check again here.

                tInfo->postLoadRefreshCurrentContextSurfacePtrs();
                needRestoreFromSnapshot = false;
            }
            if (mNeedReloadProcessResources) {
                processResources = nullptr;
                mNeedReloadProcessResources = false;
            }
        }

        //
        // log received bandwidth statistics
        //
        if (benchmarkEnabled) {
            stats_totalBytes += readBuf.validData();
            auto dt = gfxstream::base::getHighResTimeUs() / 1000 - stats_t0;
            if (dt > 1000) {
                float dts = (float)dt / 1000.0f;
                GFXSTREAM_INFO("Used Bandwidth %5.3f MB/s, time in progress %f ms total %f ms\n",
                               ((float)stats_totalBytes / dts) / (1024.0f * 1024.0f),
                               stats_progressTimeUs / 1000.0f, (float)dt);
                readBuf.printStats();
                stats_t0 = gfxstream::base::getHighResTimeUs() / 1000;
                stats_progressTimeUs = 0;
                stats_totalBytes = 0;
            }
        }

        bool progress = false;
        anyProgress = false;
        do {
            anyProgress |= progress;
            std::unique_ptr<EventHangMetadata::HangAnnotations> renderThreadData =
                std::make_unique<EventHangMetadata::HangAnnotations>();

            const char* contextName = nullptr;
            if (mNameOpt) {
                contextName = (*mNameOpt).c_str();
            }

            auto* healthMonitor = FrameBuffer::getFB()->getHealthMonitor();
            if (healthMonitor) {
                if (contextName) {
                    renderThreadData->insert(
                        {{"renderthread_guest_process", contextName}});
                }
                if (readBuf.validData() >= 4) {
                    renderThreadData->insert(
                        {{"first_opcode", std::to_string(*(uint32_t*)readBuf.buf())},
                         {"buffer_length", std::to_string(readBuf.validData())}});
                }
            }
            auto watchdog = WATCHDOG_BUILDER(healthMonitor, "RenderThread decode operation")
                                .setHangType(EventHangMetadata::HangType::kRenderThread)
                                .setAnnotations(std::move(renderThreadData))
                                .build();

            if (!tInfo->m_puid) {
                tInfo->m_puid = mContextId;
            }

            if (!processResources && tInfo->m_puid && tInfo->m_puid != INVALID_CONTEXT_ID) {
                processResources = FrameBuffer::getFB()->getProcessResources(tInfo->m_puid);
                // Decoding a sequenced Vulkan packet without these is not a degraded mode, it is
                // a broken one. Every such packet carries a sequence number and the decoder both
                // waits on and advances a counter that lives here; a thread that has no counter
                // still consumes the numbers but cannot advance it, so the count falls permanently
                // behind and any thread of the same process that does have one waits for a value
                // that will never arrive.
                //
                // Seen on KDE: a host render thread spinning at 100% of a core for minutes,
                // "seqno loop stuck: want=751 have=745 process=plasmashell", the guest's Qt Quick
                // render thread parked in vkCreateFence inside its present, and a desktop that
                // never draws. Before the decoder was taught not to dereference a null counter it
                // took the VMM down here instead, which is why this never presented as a hang.
                //
                // VirtioGpuContext::Create already does this for contexts it knows about; the
                // gap is an ASG consumer created without a context id, which leaves m_puid at
                // zero. Creating it here is idempotent -- try_emplace on the same puid is a
                // no-op -- so this only ever fills in one that is genuinely missing.
                if (!processResources) {
                    FrameBuffer::getFB()->createGraphicsProcessResources(tInfo->m_puid);
                    processResources = FrameBuffer::getFB()->getProcessResources(tInfo->m_puid);
                }
            }

            progress = false;
            size_t last;

            //
            // try to process some of the command buffer using the
            // Vulkan decoder
            //
            // Note: It's risky to limit Vulkan decoding to one thread,
            // so we do it outside the limiter
            if (tInfo->m_vkInfo) {
                if (tInfo->m_vkInfo->ctx_id == 0) {
                    tInfo->m_vkInfo->ctx_id = mContextId;
                }
                VkDecoderContext context = {
                    .processName = contextName,
                    .gfxApiLogger = &gfxLogger,
                    .healthMonitor = FrameBuffer::getFB()->getHealthMonitor(),
                    .metricsLogger = &metricsLogger,
                    .shouldExit = &(tInfo->m_shouldExit),
                };
                last = tInfo->m_vkInfo->m_vkDec.decode(readBuf.buf(), readBuf.validData(), ioStream,
                                                      processResources, context);
                // A complete packet the Vulkan decoder declined. Everything downstream of this
                // reads as "the host never answered", so name it here with the two things that
                // decide whether the decoder could have run at all.
                // Read the header here, not at the top of the loop: getData() refills the
                // buffer in between, so the values captured up there describe whatever was
                // present before the read and say nothing about what the decoder was handed.
                // (An earlier version of this probe printed those stale values and reported
                // "opcode=0 len=8" for every ordinary partial packet.)
                if (last == 0 && readBuf.validData() >= 8 && !mVkDeclineReported) {
                    uint32_t liveOp = 0, liveLen = 0;
                    std::memcpy(&liveOp, readBuf.buf(), sizeof(uint32_t));
                    std::memcpy(&liveLen, readBuf.buf() + 4, sizeof(uint32_t));
                    // Only a whole packet is interesting: short of that, returning zero is the
                    // decoder correctly asking for the rest of it.
                    if (liveLen >= 8 && readBuf.validData() >= liveLen) {
                        mVkDeclineReported = true;
                        fprintf(stderr,
                                "VKDEC-DECLINED: opcode=%u len=%u valid=%u procRes=%d puid=%u\n",
                                liveOp, liveLen, (unsigned)readBuf.validData(),
                                processResources ? 1 : 0, (unsigned)tInfo->m_puid);
                    }
                }
                if (last > 0) {
                    if (!processResources) {
                        GFXSTREAM_ERROR(
                            "Processed some Vulkan packets without process resources created. "
                            "That's problematic.");
                    }
                    if (mRingStream) mRingStream->noteDecoded(headerOpcode);
                    readBuf.consume(last);
                    progress = true;
                }
            }

            std::optional<gfxstream::base::AutoLock> limitedModeLock;
            if (mRunInLimitedMode) {
                limitedModeLock.emplace(sThreadRunLimiter);
            }

            // try to process some of the command buffer using the GLESv1
            // decoder
            //
            // DRIVER WORKAROUND:
            // On Linux with NVIDIA GPU's at least, we need to avoid performing
            // GLES ops while someone else holds the FrameBuffer write lock.
            //
            // To be more specific, on Linux with NVIDIA Quadro K2200 v361.xx,
            // we get a segfault in the NVIDIA driver when glTexSubImage2D
            // is called at the same time as glXMake(Context)Current.
            //
            // To fix, this driver workaround avoids calling
            // any sort of GLES call when we are creating/destroying EGL
            // contexts.
            {
                FrameBuffer::getFB()->lockContextStructureRead();
            }

#if GFXSTREAM_ENABLE_HOST_GLES
            if (tInfo->m_glInfo) {
                {
                    last = tInfo->m_glInfo->m_glDec.decode(
                            readBuf.buf(), readBuf.validData(), ioStream, &checksumCalc);
                    if (last > 0) {
                        progress = true;
                        readBuf.consume(last);
                    }
                }

                //
                // try to process some of the command buffer using the GLESv2
                // decoder
                //
                {
                    last = tInfo->m_glInfo->m_gl2Dec.decode(readBuf.buf(), readBuf.validData(),
                                                           ioStream, &checksumCalc);

                    if (last > 0) {
                        progress = true;
                        readBuf.consume(last);
                    }
                }
            }
#endif

            FrameBuffer::getFB()->unlockContextStructureRead();
            //
            // try to process some of the command buffer using the
            // renderControl decoder
            //
#if GFXSTREAM_ENABLE_HOST_GLES
            {
                last = tInfo->m_rcDec.decode(readBuf.buf(), readBuf.validData(),
                                            ioStream, &checksumCalc);
                if (last > 0) {
                    readBuf.consume(last);
                    progress = true;
                }
            }
#endif
        } while (progress);
    }

#if GFXSTREAM_ENABLE_HOST_GLES
    if (tInfo->m_glInfo) {
        FrameBuffer::getFB()->drainGlRenderThreadResources();
    }
#endif

    setFinished();
    // Since we now control when the thread exits, we must make sure the RenderThreadInfo is
    // destroyed after the RenderThread is finished, as the RenderThreadInfo cleanup thread is
    // waiting on the object to be destroyed.
    tInfo.reset();
    waitForExitSignal();

    fprintf(stderr, "RT-EXIT: thread=%p ring=%d\n", this, mRingStream ? 1 : 0);
    return 0;
}

}  // namespace gfxstream
