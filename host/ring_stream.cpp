// Copyright (C) 2019 The Android Open Source Project
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

#include "ring_stream.h"

#include <assert.h>
#include <memory.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <vector>

#include "gfxstream/host/address_space_graphics.h"
#include "gfxstream/host/dma_device.h"
#include "gfxstream/common/logging.h"
#include "gfxstream_diag.h"
#include "gfxstream/host/stream_utils.h"
#include "gfxstream/system/System.h"
#include "render-utils/dma_device.h"
#include "render-utils/stream.h"

namespace gfxstream {
namespace {

// Attach to the ring the guest has already set up, rather than setting it up again.
//
// Both sides used to call asg_context_create on the same shared memory, and that call ends with
// ring_buffer_init, which rewinds the ring's shared read and write counters. Nothing orders the
// two calls against each other: the guest maps the blob, initialises the ring and starts writing,
// while this side runs when the guest pings ASG_SET_VERSION. When that landed after the guest had
// already committed its first bytes -- the four-byte clientFlags word that opens every connection
// is usually the first -- the counters went back to zero underneath it, and from then on this side
// read from a position the guest had never written. Every packet after that was framed from the
// wrong offset: an opcode read as a length, a wait for a body of some twenty thousand bytes nobody
// would ever send, and a guest thread parked forever on a reply, with an empty ring and no error
// on either side. It presented as a desktop that came up whole, or as a bare wallpaper, or not at
// all, from one boot to the next, on the same binaries.
//
// So: the ring struct is shared and the guest owns its initialisation. The views are per-side --
// host and guest hold different pointers to the same bytes -- and this side owns its own, which is
// what ring_buffer_init_view_only is for. buffer_size and flush_interval stay the host's to
// publish, because the guest reads flush_interval to size its writes and cannot know it first.
struct asg_context CreateContext(const AsgConsumerCreateInfo& info) {
    struct asg_context context;

    context.to_host = reinterpret_cast<struct ring_buffer*>(
        info.ring_storage + offsetof(struct asg_ring_storage, to_host));
    context.to_host_large_xfer.ring = reinterpret_cast<struct ring_buffer*>(
        info.ring_storage + offsetof(struct asg_ring_storage, to_host_large_xfer));
    context.from_host_large_xfer.ring = reinterpret_cast<struct ring_buffer*>(
        info.ring_storage + offsetof(struct asg_ring_storage, from_host_large_xfer));

    context.buffer = info.buffer;
    // asg_context carries buffer_size and type1Read validates the guest-supplied offset/size
    // against it, so attaching instead of calling asg_context_create means setting it here.
    context.buffer_size = info.buffer_size;
    context.host_state = reinterpret_cast<asg_host_state*>(&context.to_host->state);
    context.ring_config = reinterpret_cast<asg_ring_config*>(context.to_host->config);

    ring_buffer_init_view_only(&context.to_host_large_xfer.view, (uint8_t*)context.buffer,
                               info.buffer_size);
    ring_buffer_init_view_only(&context.from_host_large_xfer.view, (uint8_t*)context.buffer,
                               info.buffer_size);

    context.ring_config->buffer_size = info.buffer_size;
    context.ring_config->flush_interval = info.buffer_flush_interval;


    // What the counters say at the moment this side attaches, which is the difference between a
    // session that works and one that hangs with no error on either end. A ring the guest has not
    // touched yet reads write=0 read=0 stateval=0; anything else means these pages still hold a
    // previous session's header and the guest's own initialisation has not landed yet, so this
    // side is about to frame packets from an offset the guest never wrote.
    GFXSTREAM_STALL_PRINT(
            "RING-VIEW: storage=%p to_host=%p state=%p write=%u read=%u stateval=%u "
            "buffer=%p size=%u\n",
            (void*)info.ring_storage, (void*)context.to_host, (void*)context.host_state,
            context.to_host->write_pos, context.to_host->read_pos,
            (unsigned)*(context.host_state), (void*)context.buffer, info.buffer_size);
    return context;
}

void SaveRingConfig(Stream* stream, const struct asg_ring_config& config) {
    stream->putBe32(config.buffer_size);
    stream->putBe32(config.flush_interval);
    stream->putBe32(config.host_consumed_pos);
    stream->putBe32(config.guest_write_pos);
    stream->putBe32(config.transfer_mode);
    stream->putBe32(config.transfer_size);
    stream->putBe32(config.in_error);
}

void LoadRingConfig(Stream* stream, struct asg_ring_config* config) {
    config->buffer_size = stream->getBe32();
    config->flush_interval = stream->getBe32();
    config->host_consumed_pos = stream->getBe32();
    config->guest_write_pos = stream->getBe32();
    config->transfer_mode = stream->getBe32();
    config->transfer_size = stream->getBe32();
    config->in_error = stream->getBe32();
}

void SaveAsgContext(Stream* stream, const struct asg_context& context) {
    stream->write(context.to_host, sizeof(struct ring_buffer));
    stream->write(context.to_host_large_xfer.ring, sizeof(struct ring_buffer));
    stream->write(context.from_host_large_xfer.ring, sizeof(struct ring_buffer));
    stream->putBe32(context.buffer_size);
    stream->write(context.buffer, context.buffer_size);
}

void LoadAsgContext(Stream* stream, struct asg_context* context) {
    stream->read(context->to_host, sizeof(struct ring_buffer));
    stream->read(context->to_host_large_xfer.ring, sizeof(struct ring_buffer));
    stream->read(context->from_host_large_xfer.ring, sizeof(struct ring_buffer));
    context->buffer_size = stream->getBe32();
    stream->read(context->buffer, context->buffer_size);
}

}  // namespace

RingStream::RingStream(const AsgConsumerCreateInfo& info, size_t bufsize) :
    IOStream(bufsize),
    mContext(CreateContext(info)),
    mSavedRingConfig(*mContext.ring_config),
    mCallbacks(info.callbacks),
    mBufSize(info.buffer_size) {}

RingStream::~RingStream() = default;


// How much of a consumer's time on its core is spinning rather than consuming.
//
// FPS on this device varies by more than two to one between boots of the same binary, which makes
// any throughput comparison of a spin setting unreadable. Counts do not: the per-frame structural
// numbers this session measured -- opcodes, batches, replays -- stayed within 4% across the same
// runs whose scores moved 2.1x, because they are normalised by the work rather than by the clock.
//
// So count the spinning instead of timing it. Reported against packets consumed, so "spins per
// packet" is comparable between two trees without either of them having to score the same.
namespace {
// Per consumer thread, because the aggregate is dominated by whichever thread has nothing to do.
//
// A consumer with no traffic sits inside one readRaw call forever: spin the budget, park on the
// message queue, get woken, spin the budget again. It never returns, so it contributes no reads
// and no packets, while producing far more spin events than a thread that is actually working --
// which is why the summed counters read as "zero reads, zero packets" even though the system is
// plainly running. Summed, the busy thread is invisible; split by thread, the two are obvious.
//
// Parking itself is cheap: onUnavailableRead blocks on a queue. What costs is the spin before it,
// and with a zero sleep each turn is a sched_yield -- a syscall that hands the core away and takes
// it back, thousands of times per wake, on a core every consumer shares.
struct SpinCount {
    static bool Enabled() {
        static const bool on = getenv("GFXSTREAM_ASG_SPIN_TRACE") != nullptr;
        return on;
    }
    struct Counts {
        uint64_t episodes = 0, spins = 0, parks = 0, reads = 0, packets = 0, events = 0;
        uint64_t gaps[7] = {};
        uint64_t lastPacketNs = 0;
        uint64_t stallStartNs = 0;
        uint64_t stallStartNotifies = 0;
        char name[20] = {};
    };
    static Counts& Mine() {
        static thread_local Counts c = [] {
            Counts init;
#if defined(__linux__)
            prctl(PR_GET_NAME, init.name, 0, 0, 0);
#endif
            return init;
        }();
        return c;
    }
    static uint64_t NowNs() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    }
    static void NoteEntry() { ++Mine().reads; }
    static void NoteEpisode() { ++Mine().episodes; }

    // A stall, caught while it is happening rather than inferred from a histogram afterwards.
    //
    // The gap distribution says most packets arrive faster than they used to and about one in a
    // hundred takes over five milliseconds, and that thin tail is where the time goes. An average
    // cannot say what those hundred-fold waits have in common; a line printed from inside one can.
    //
    // The threshold sits far above a normal wait (a handful of turns) and far below the park
    // budget, so a healthy consumer never reaches it and pays a compare per turn for the privilege.
    static constexpr uint32_t kStallSpins = 500;
    static void NoteStallBegin(uint32_t writePos, uint32_t readPos, uint32_t state, uint32_t avail,
                               uint32_t largeAvail, uint32_t lastOpcode) {
        Counts& c = Mine();
        c.stallStartNs = NowNs();
        c.stallStartNotifies = asgNotifyCount().load(std::memory_order_relaxed);
        GFXSTREAM_WARNING(
            "ASGSTALL[%s] begin: spins=%u write=%u read=%u state=%u avail=%u largeavail=%u "
            "lastop=%u",
            c.name[0] ? c.name : "?", kStallSpins, writePos, readPos, state, avail, largeAvail,
            lastOpcode);
    }
    static void NoteStallEnd(uint32_t spins, uint32_t writePos, uint32_t readPos, uint32_t state,
                             uint32_t avail, uint32_t largeAvail, bool parked) {
        Counts& c = Mine();
        if (!c.stallStartNs) return;
        const uint64_t us = (NowNs() - c.stallStartNs) / 1000;
        const uint64_t notifies =
            asgNotifyCount().load(std::memory_order_relaxed) - c.stallStartNotifies;
        c.stallStartNs = 0;
        GFXSTREAM_WARNING(
            "ASGSTALL[%s] end: %s after %uus, spins=%u write=%u read=%u state=%u avail=%u "
            "largeavail=%u wakeups=%llu",
            c.name[0] ? c.name : "?", parked ? "parked" : "got data", (unsigned)us, spins, writePos,
            readPos, state, avail, largeAvail, (unsigned long long)notifies);
    }
    static void NotePacket() {
        Counts& c = Mine();
        ++c.packets;
        const uint64_t now = NowNs();
        if (c.lastPacketNs) {
            const uint64_t gapUs = (now - c.lastPacketNs) / 1000;
            static constexpr uint64_t kEdges[] = {10, 50, 100, 500, 1000, 5000};
            size_t i = 0;
            while (i < 6 && gapUs >= kEdges[i]) ++i;
            ++c.gaps[i];
        }
        c.lastPacketNs = now;
    }
    static void NoteSpins(uint64_t n, bool parked) {
        Counts& c = Mine();
        c.spins += n;
        if (parked) ++c.parks;
        constexpr uint64_t kEvery = 2000;
        if (++c.events % kEvery) return;
        // Raw counts only. A ratio computed here hides how its parts were counted, which has
        // already produced three findings this session that were arithmetic rather than behaviour.
        GFXSTREAM_WARNING(
            "ASGSPIN[%s] raw: episodes=%llu spin_iters=%llu parks=%llu reads=%llu packets=%llu | "
            "gap us <10:%llu <50:%llu <100:%llu <500:%llu <1k:%llu <5k:%llu 5k+:%llu",
            c.name[0] ? c.name : "?", (unsigned long long)c.episodes, (unsigned long long)c.spins,
            (unsigned long long)c.parks, (unsigned long long)c.reads,
            (unsigned long long)c.packets, (unsigned long long)c.gaps[0],
            (unsigned long long)c.gaps[1], (unsigned long long)c.gaps[2],
            (unsigned long long)c.gaps[3], (unsigned long long)c.gaps[4],
            (unsigned long long)c.gaps[5], (unsigned long long)c.gaps[6]);
        c.episodes = c.spins = c.parks = c.reads = c.packets = 0;
        for (auto& g : c.gaps) g = 0;
    }
};
}  // namespace

void RingStream::reloadRingConfig() {
    *mContext.ring_config = mSavedRingConfig;
}

void* RingStream::allocBuffer(size_t minSize) {
    if (mWriteBuffer.size() < minSize) {
        mWriteBuffer.resize_noinit(minSize);
    }
    return mWriteBuffer.data();
}

int RingStream::commitBuffer(size_t size) {
    size_t sent = 0;
    auto data = mWriteBuffer.data();

    size_t iters = 0;
    size_t backedOffIters = 0;
    const size_t kBackoffIters = 10000000ULL;
    while (sent < size) {
        ++iters;
        auto avail = ring_buffer_available_write(
            mContext.from_host_large_xfer.ring,
            &mContext.from_host_large_xfer.view);

        // Check if the guest process crashed.
        if (!avail) {
            if (*(mContext.host_state) == ASG_HOST_STATE_EXIT) {
                return sent;
            } else {
                ring_buffer_yield();
                if (iters > kBackoffIters) {
                    gfxstream::base::sleepUs(10);
                    ++backedOffIters;
                }
            }
            continue;
        }

        auto remaining = size - sent;
        auto todo = remaining < avail ? remaining : avail;

        ring_buffer_view_write(
            mContext.from_host_large_xfer.ring,
            &mContext.from_host_large_xfer.view,
            data + sent, todo, 1);

        sent += todo;
    }

    if (backedOffIters > 0) {
        GFXSTREAM_WARNING(
            "Backed off %zu times to avoid overloading the guest system. This "
            "may indicate resource constraints or performance issues.",
            backedOffIters);
    }
    return sent;
}

const unsigned char* RingStream::readRaw(void* buf, size_t* inout_len) {
    if (SpinCount::Enabled()) SpinCount::NoteEntry();
    size_t wanted = *inout_len;
    size_t count = 0U;
    auto dst = static_cast<char*>(buf);

    uint32_t ringAvailable = 0;
    uint32_t ringLargeXferAvailable = 0;

    // What to do while the ring is empty. There are two costs, and a single flat spin count has to
    // trade them against each other blind: parking needs a guest doorbell -- a virtio round trip --
    // to come back, while yield-spinning contends with the guest's vCPU threads for the same
    // physical cores. Backing off in stages separates them.
    //
    // Measured with Minecraft, GPU pinned at 734 MHz, one fixed scene, as guest submits/second:
    //     flat 3000                        508
    //     3000:0                           499   (this code, one stage -- the ladder is free)
    //     500:0,5000:1,10000:10,30000:50   540
    //     3000:0,20000:1,40000:10,80000:50 554
    //     3000:0,50000:1                   558
    //     3000:0,150000:1                  567
    //     20000:0,150000:1                 494   (worse than no ladder at all)
    //
    // Two things fell out of that. Spinning past 3000 lands below the flat baseline, because that
    // is expensive waiting on cores the guest needs. And the sleeping stage looked like where all
    // the gain was -- parking seemed a loss, since the host consumer is ~99% idle anyway, so
    // waiting cheaply appeared to cost nobody anything while a park has to be paid back with a VM
    // exit.
    //
    // "Waiting cheaply" was the mistake, and one game was the wrong workload to find it on. Each
    // sleepUs(1) programs a timer and takes an interrupt on expiry; one guest process has a couple
    // of render threads doing that, and a desktop session has one per client. Measured on an idle
    // KDE desktop -- nothing running, nothing moving -- as timer interrupts across the phone, how
    // busy the phone was, and vkmark on the same boot:
    //
    //     3000:0,150000:1   289211 irq/s   100.0% busy   vkmark 1643
    //     3000:0              1822 irq/s    21.1% busy   vkmark 3632   <- default
    //     30:0                1133 irq/s    12.6% busy   vkmark  924   (the flat 30 this replaces)
    //
    // The sleeping stage does not merely spend CPU that was going spare: it saturates the phone
    // with interrupt work and takes throughput down with it, to less than half of what the same
    // build does without it. Spinning is what buys the handoff latency, and 30 iterations is too
    // short to buy any, so it parks constantly. Keep the yield stage, park after it, and let a
    // doorbell do the rest.
    //
    // GFXSTREAM_ASG_SPIN_LEVELS overrides the stages as ascending "iters:sleep_us" pairs; an
    // empty-ring iteration past the last stage parks. One stage reproduces flat behaviour, which
    // is what made the ladder measurable apart from the stage values.
    struct SpinLevel {
        uint32_t upToIter;
        uint32_t sleepUs;
    };
    static const std::vector<SpinLevel> spinLevels = [] {
        std::vector<SpinLevel> levels;
        const char* env = getenv("GFXSTREAM_ASG_SPIN_LEVELS");
        const char* spec = env ? env : "3000:0";
        uint32_t prev = 0;
        while (*spec) {
            char* end = nullptr;
            const unsigned long iters = strtoul(spec, &end, 10);
            if (end == spec || *end != ':') break;
            spec = end + 1;
            const unsigned long us = strtoul(spec, &end, 10);
            if (end == spec) break;
            spec = *end == ',' ? end + 1 : end;
            if (iters > prev) {
                levels.push_back(
                    SpinLevel{static_cast<uint32_t>(iters), static_cast<uint32_t>(us)});
                prev = static_cast<uint32_t>(iters);
            }
        }
        if (levels.empty()) levels.push_back(SpinLevel{3000, 0});
        return levels;
    }();
    // Sleeping stages are only distinct from one another if a 1 us sleep is near 1 us: the default
    // timer slack is 50 us, which collapses every stage below it into the same wait. Slack is
    // per-thread, so each consumer sets its own once.
    static thread_local const bool slackSet = [] {
#if defined(__linux__) && defined(PR_SET_TIMERSLACK)
        prctl(PR_SET_TIMERSLACK, 1000 /* ns */, 0, 0, 0);
#endif
        return true;
    }();
    (void)slackSet;
    uint32_t spins = 0;
    bool inLargeXfer = true;

    *(mContext.host_state) = ASG_HOST_STATE_CAN_CONSUME;

    while (count < wanted) {

        if (mReadBufferLeft) {
            size_t avail = std::min<size_t>(wanted - count, mReadBufferLeft);
            memcpy(dst + count,
                    mReadBuffer.data() + (mReadBuffer.size() - mReadBufferLeft),
                    avail);
            count += avail;
            mReadBufferLeft -= avail;
            continue;
        }

        mReadBuffer.clear();

        // no read buffer left...
        if (count > 0) {  // There is some data to return.
            break;
        }

        *(mContext.host_state) = ASG_HOST_STATE_CAN_CONSUME;

        if (mShouldExit) {
            return nullptr;
        }

        ringAvailable =
            ring_buffer_available_read(mContext.to_host, 0);
        ringLargeXferAvailable =
            ring_buffer_available_read(
                mContext.to_host_large_xfer.ring,
                &mContext.to_host_large_xfer.view);

        if (SpinCount::Enabled() && spins && (ringAvailable || ringLargeXferAvailable)) {
            SpinCount::NoteStallEnd(spins, mContext.to_host->write_pos, mContext.to_host->read_pos,
                                    (unsigned)*(mContext.host_state), ringAvailable,
                                    ringLargeXferAvailable, /* parked */ false);
        }

        auto current = dst + count;
        auto ptrEnd = dst + wanted;

        if (ringAvailable) {
            inLargeXfer = false;
            uint32_t transferMode =
                mContext.ring_config->transfer_mode;
            switch (transferMode) {
                case 1:
                    type1Read(ringAvailable, dst, &count, &current, ptrEnd);
                    break;
                case 2:
                    type2Read(ringAvailable, &count, &current, ptrEnd);
                    break;
                case 3:
                    // emugl::emugl_crash_reporter(
                    //     "Guest should never set to "
                    //     "transfer mode 3 with ringAvailable != 0\n");
                default:
                    // emugl::emugl_crash_reporter(
                    //     "Unknown transfer mode %u\n",
                    //     transferMode);
                    break;
            }
        } else if (ringLargeXferAvailable) {
            type3Read(ringLargeXferAvailable,
                      &count, &current, ptrEnd);
            inLargeXfer = true;
            if (0 == __atomic_load_n(&mContext.ring_config->transfer_size, __ATOMIC_ACQUIRE)) {
                inLargeXfer = false;
            }
        } else {
            if (inLargeXfer && 0 != __atomic_load_n(&mContext.ring_config->transfer_size, __ATOMIC_ACQUIRE)) {
                continue;
            }

            if (inLargeXfer && 0 == __atomic_load_n(&mContext.ring_config->transfer_size, __ATOMIC_ACQUIRE)) {
                inLargeXfer = false;
            }

            ++spins;
            if (SpinCount::Enabled()) {
                if (spins == 1) SpinCount::NoteEpisode();
                SpinCount::NoteSpins(1, /* parked */ false);
                if (spins == SpinCount::kStallSpins) {
                    SpinCount::NoteStallBegin(mContext.to_host->write_pos,
                                              mContext.to_host->read_pos,
                                              (unsigned)*(mContext.host_state),
                                              ring_buffer_available_read(mContext.to_host, 0),
                                              ring_buffer_available_read(
                                                  mContext.to_host_large_xfer.ring,
                                                  &mContext.to_host_large_xfer.view),
                                              mLastDecoded);
                }
            }
            uint32_t sleepUs = UINT32_MAX;  // past the last stage -> park
            for (const SpinLevel& level : spinLevels) {
                if (spins <= level.upToIter) {
                    sleepUs = level.sleepUs;
                    break;
                }
            }
            if (sleepUs != UINT32_MAX) {
                if (sleepUs == 0) {
                    ring_buffer_yield();
                } else {
                    gfxstream::base::sleepUs(sleepUs);
                }
                continue;
            }
            if (SpinCount::Enabled()) {
                SpinCount::NoteStallEnd(spins, mContext.to_host->write_pos,
                                        mContext.to_host->read_pos,
                                        (unsigned)*(mContext.host_state),
                                        ring_buffer_available_read(mContext.to_host, 0),
                                        ring_buffer_available_read(
                                            mContext.to_host_large_xfer.ring,
                                            &mContext.to_host_large_xfer.view),
                                        /* parked */ true);
            }
            spins = 0;
            if (SpinCount::Enabled()) SpinCount::NoteSpins(0, /* parked */ true);

            if (mShouldExit) {
                return nullptr;
            }

            if (mShouldExitForSnapshot && mInSnapshotOperation) {
                return nullptr;
            }

            // The ladder ran dry, so park now rather than running it kMaxUnavailableReads more
            // times: the stages already covered every latency worth spinning through.
            mUnavailableReadCount = kMaxUnavailableReads;
            if (mUnavailableReadCount >= kMaxUnavailableReads) {
                *(mContext.host_state) = ASG_HOST_STATE_NEED_NOTIFY;

                bool sleeping = false;
                do {
                    // Whatever this loop does next, it is parked until the guest rings: say so
                    // every time round. The state word is the only thing the guest has to decide
                    // whether to ring at all -- it pings when this is neither CAN_CONSUME nor
                    // RENDERING -- so a stale CAN_CONSUME here tells the guest the host is busy
                    // and needs no doorbell, while the host waits for exactly that doorbell.
                    *(mContext.host_state) = ASG_HOST_STATE_NEED_NOTIFY;
                    const AsgOnUnavailableReadStatus status = mCallbacks.onUnavailableRead();
                    switch (status) {
                        case AsgOnUnavailableReadStatus::kContinue: {
                            *(mContext.host_state) = ASG_HOST_STATE_CAN_CONSUME;
                            // And stop sleeping. Without this the loop calls onUnavailableRead
                            // forever once any call has returned kSleep: a later kContinue means
                            // "go read again", but sleeping stays set, so the read never happens
                            // and the state is left saying CAN_CONSUME.
                            sleeping = false;
                            break;
                        }
                        case AsgOnUnavailableReadStatus::kExit: {
                            *(mContext.host_state) = ASG_HOST_STATE_EXIT;
                            mShouldExit = true;
                            break;
                        }
                        case AsgOnUnavailableReadStatus::kSleep: {
                            sleeping = true;
                            break;
                        }
                        case AsgOnUnavailableReadStatus::kPauseForSnapshot: {
                            mShouldExitForSnapshot = true;
                            break;
                        }
                        case AsgOnUnavailableReadStatus::kResumeAfterSnapshot: {
                            mShouldExitForSnapshot = false;
                            break;
                        }
                    }
                } while (sleeping);
            }
            continue;
        }
    }

    *inout_len = count;
    if (SpinCount::Enabled()) SpinCount::NotePacket();
    ++mXmits;
    mTotalRecv += count;

    *(mContext.host_state) = ASG_HOST_STATE_RENDERING;
    return (const unsigned char*)buf;
}

void RingStream::type1Read(
    uint32_t available,
    char* begin,
    size_t* count, char** current, const char* ptrEnd) {

    uint32_t xferTotal = available / sizeof(struct asg_type1_xfer);

    if (mType1Xfers.size() < xferTotal) {
        mType1Xfers.resize(xferTotal * 2);
    }

    auto xfersPtr = mType1Xfers.data();

    ring_buffer_copy_contents(
        mContext.to_host, 0, xferTotal * sizeof(struct asg_type1_xfer), (uint8_t*)xfersPtr);

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code-loop-increment"
#endif // __clang__
    for (uint32_t i = 0; i < xferTotal; ++i) {
        const asg_type1_xfer& xfer = xfersPtr[i];

        const uint32_t buffer_size = mContext.buffer_size;

        // Guest controls offset/size via shared memory; validate against the
        // host-allocated auxiliary buffer before dereferencing.
        if (xfer.offset >= buffer_size || xfer.size > buffer_size - xfer.offset) {
            GFXSTREAM_ERROR("Invalid type1 xfer: offset %u, size %u, buffer_size %u\n",
                            xfer.offset, xfer.size, buffer_size);
            __atomic_store_n(&mContext.ring_config->in_error, 1, __ATOMIC_RELEASE);
            return;
        }

        if (*current + xfer.size > ptrEnd) {
            // Save in a temp buffer or we'll get stuck
            if (begin == *current && i == 0) {
                const char* src = mContext.buffer + xfer.offset;
                mReadBuffer.resize_noinit(xfer.size);
                memcpy(mReadBuffer.data(), src, xfer.size);
                mReadBufferLeft = xfer.size;
                ring_buffer_advance_read(
                        mContext.to_host, sizeof(struct asg_type1_xfer), 1);
                __atomic_fetch_add(&mContext.ring_config->host_consumed_pos, xfersPtr[i].size, __ATOMIC_RELEASE);
            }
            return;
        }
        const char* src = mContext.buffer + xfer.offset;
        memcpy(*current, src, xfer.size);
        ring_buffer_advance_read(
                mContext.to_host, sizeof(struct asg_type1_xfer), 1);
        __atomic_fetch_add(&mContext.ring_config->host_consumed_pos, xfer.size, __ATOMIC_RELEASE);
        *current += xfer.size;
        *count += xfer.size;

        // TODO: Figure out why running multiple xfers here can result in data
        // corruption and remove clang diagnostic block.
        return;
    }
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__
}

void RingStream::type2Read(
    uint32_t available,
    size_t* count, char** current,const char* ptrEnd) {

    GFXSTREAM_FATAL("nyi. abort");

    uint32_t xferTotal = available / sizeof(struct asg_type2_xfer);

    if (mType2Xfers.size() < xferTotal) {
        mType2Xfers.resize(xferTotal * 2);
    }

    auto xfersPtr = mType2Xfers.data();

    ring_buffer_copy_contents(
        mContext.to_host, 0, available, (uint8_t*)xfersPtr);

    for (uint32_t i = 0; i < xferTotal; ++i) {

        if (*current + xfersPtr[i].size > ptrEnd) return;

        const char* src =
            mCallbacks.getPtr(xfersPtr[i].physAddr);

        memcpy(*current, src, xfersPtr[i].size);

        ring_buffer_advance_read(
            mContext.to_host, sizeof(struct asg_type1_xfer), 1);

        *current += xfersPtr[i].size;
        *count += xfersPtr[i].size;
    }
}

void RingStream::type3Read(
    uint32_t available,
    size_t* count, char** current, const char* ptrEnd) {

    uint32_t xferTotal = __atomic_load_n(&mContext.ring_config->transfer_size, __ATOMIC_ACQUIRE);
    uint32_t maxCanRead = ptrEnd - *current;
    uint32_t ringAvail = available;
    uint32_t actuallyRead = std::min(ringAvail, std::min(xferTotal, maxCanRead));

    // Decrement transfer_size before letting the guest proceed in ring_buffer funcs or we will race
    // to the next time the guest sets transfer_size
    __atomic_fetch_sub(&mContext.ring_config->transfer_size, actuallyRead, __ATOMIC_RELEASE);

    ring_buffer_read_fully_with_abort(
            mContext.to_host_large_xfer.ring,
            &mContext.to_host_large_xfer.view,
            *current, actuallyRead,
            1, &mContext.ring_config->in_error);

    *current += actuallyRead;
    *count += actuallyRead;
}

void* RingStream::getDmaForReading(uint64_t guest_paddr) {
    return gfxstream::host::g_gfxstream_dma_get_host_addr(guest_paddr);
}

void RingStream::unlockDma(uint64_t guest_paddr) {
    gfxstream::host::g_gfxstream_dma_unlock(guest_paddr);
}

int RingStream::writeFully(const void* buf, size_t len) {
    void* dstBuf = alloc(len);
    memcpy(dstBuf, buf, len);
    flush();
    return 0;
}

const unsigned char *RingStream::readFully( void *buf, size_t len) {
    GFXSTREAM_FATAL("not intended for use with RingStream");
    return nullptr;
}

void RingStream::onSave(gfxstream::Stream* stream) {
    stream->putBe32(mReadBufferLeft);
    stream->write(mReadBuffer.data() + mReadBuffer.size() - mReadBufferLeft,
                  mReadBufferLeft);

    gfxstream::host::saveBuffer(stream, mWriteBuffer);

    stream->putBe32(mUnavailableReadCount);

    mSavedRingConfig = *mContext.ring_config;

    SaveRingConfig(stream, mSavedRingConfig);

    SaveAsgContext(stream, mContext);
}

unsigned char* RingStream::onLoad(gfxstream::Stream* stream) {
    gfxstream::host::loadBuffer(stream, &mReadBuffer);
    mReadBufferLeft = mReadBuffer.size();

    gfxstream::host::loadBuffer(stream, &mWriteBuffer);

    mUnavailableReadCount = stream->getBe32();

    LoadRingConfig(stream, &mSavedRingConfig);

    LoadAsgContext(stream, &mContext);

    return reinterpret_cast<unsigned char*>(mWriteBuffer.data());
}

}  // namespace gfxstream
