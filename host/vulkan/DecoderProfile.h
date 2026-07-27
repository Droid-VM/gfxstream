// Copyright 2026 Google LLC
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace gfxstream {
namespace vk {

// Where the wire time goes, per opcode.
//
// The question this exists to answer is not how many of each opcode cross the ring -- it is what
// the guest is waiting for. The guest blocks on any opcode that carries a return value, and its
// wait is the host's dispatch time for that opcode PLUS however much work was already queued ahead
// of it, so a count alone cannot tell a call that is slow from a call that is merely stuck behind
// other work. Timing every opcode, and recording how many packets arrive per batch, separates
// those two.
//
// Enabled by GFXSTREAM_DECODER_PROFILE=1; when off, Begin returns 0 without reading the clock and
// End does nothing. Reports every GFXSTREAM_DECODER_PROFILE_SEC seconds (default 10).
//
// Returns a start timestamp in ns, or 0 when profiling is off.
uint64_t decoderProfileBegin();

// Same clock, unconditionally -- for pairing with a Begin that is known to be non-zero.
uint64_t decoderProfileNow();

// Attributes the elapsed time to opcode. start==0 is ignored, so the disabled path costs one
// compare.
//
// Nested calls are accounted to the innermost opcode only: vkQueueFlushCommandsGOOGLE's timing
// spans the whole command-buffer replay, so without this its number would include every vkCmd*
// inside it and the two would double-count -- one flush measured 427us against ~268us of vkCmd*
// it contained. What is left after subtracting is the flush's own framing cost, which is the part
// that could actually be reduced.
void decoderProfileEnd(uint32_t opcode, uint64_t start);

// For a call decoded inside another timed call (the vkCmd* inside a command-buffer replay).
// Records its own time and adds it to what the enclosing call will subtract.
void decoderProfileEndInner(uint32_t opcode, uint64_t start);

// One call per decode() invocation: how many packets that buffer held and how long the whole batch
// took. A synchronous opcode at the end of a 200-packet batch waited for the other 199.
void decoderProfileBatch(uint64_t packets, uint64_t nanos);

// Same, for one subDecode() call -- the command-buffer replay inside a
// vkQueueFlushCommandsGOOGLE. Reported separately because the two say different things: a
// top-level batch is what the guest sent in one go, a sub batch is how many vkCmd* one flush
// carried.
void decoderProfileSubBatch(uint64_t packets, uint64_t nanos);

// Clears the per-thread nesting accumulator; call at the start of a top-level decode batch.
void decoderProfileResetNesting();

// Time inside a command-buffer flush that is not the replayed commands themselves. That residue is
// bimodal -- 88% of flushes spend under 25us on it, 12% spend 400-800us -- and the phases have
// different fixes, so they are counted apart.
enum class FlushPhase { kHandleLookup = 0, kPoolFree = 1, kCount = 2 };
void decoderProfileFlushPhase(FlushPhase phase, uint64_t nanos);

// Thread CPU clock, or 0 when profiling is off.
uint64_t decoderProfileThreadCpuNow();

// One flush's wall and CPU time, bucketed by whether it hit the slow mode.
void decoderProfileFlushWallCpu(uint64_t wallNanos, uint64_t cpuNanos);

// Counts queue submissions the host has finished handing to the driver. Independent of whether
// profiling is enabled: vkWaitSemaphores reads it to find out whether submissions complete while
// it is blocked, which is the difference between waiting on the GPU and waiting on this proxy.
void decoderNoteSubmitHandled();
uint64_t decoderSubmitsHandled();

}  // namespace vk
}  // namespace gfxstream
