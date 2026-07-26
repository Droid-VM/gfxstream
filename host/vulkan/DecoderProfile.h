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
void decoderProfileEnd(uint32_t opcode, uint64_t start);

// One call per decode() invocation: how many packets that buffer held and how long the whole batch
// took. A synchronous opcode at the end of a 200-packet batch waited for the other 199.
void decoderProfileBatch(uint64_t packets, uint64_t nanos);

}  // namespace vk
}  // namespace gfxstream
