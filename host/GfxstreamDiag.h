// Copyright 2026 The Android Open Source Project
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

#pragma once

#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace gfxstream {

// One switch for the diagnostics added while chasing stream stalls on this route.
//
// They print through fprintf(stderr) rather than the logger because crosvm raises gfxstream's log
// level high enough to drop GFXSTREAM_WARNING and GFXSTREAM_ERROR entirely, which produced three
// separate wrong conclusions before anyone noticed the messages were never reaching the log at
// all. Being unconditional then cost something else: one vkmark run emitted 6622 lines, every one
// of them a write on the pipe crosvm's output is streamed over.
//
// So they stay, and they stay off. GFXSTREAM_DIAG=1 turns them on. Read once, so a site that is
// off costs a load and a branch.
inline bool diagEnabled() {
    static const bool enabled = [] {
        const char* v = getenv("GFXSTREAM_DIAG");
        return v && v[0] && strcmp(v, "0") != 0;
    }();
    return enabled;
}

}  // namespace gfxstream

// Same arguments as fprintf(stderr, ...), minus the stream.
#define GFXSTREAM_DIAG_PRINT(...)                    \
    do {                                             \
        if (::gfxstream::diagEnabled()) {            \
            fprintf(stderr, __VA_ARGS__);            \
        }                                            \
    } while (0)

// The states that mean a stream is finished or stuck: an opcode with no case, a thread giving
// up, a consumer parked with a guest waiting on the other side. Every one of them is the answer
// to "why did that guest hang", and each is rate-limited at its call site, so it is tempting to
// print them unconditionally.
//
// It was tried, and measured: with these unconditional, vulkaninfo failed 0/6 on the gfxstream
// route; with them behind the switch again, 6/6, same guest and same host otherwise. They sit in
// the consumer's park path, and an fprintf+fflush there changes the timing of the very handshake
// they are meant to describe. A diagnostic that alters what it measures is worse than a silent
// one, so they stay behind GFXSTREAM_DIAG -- turn it on deliberately, and read the result knowing
// the timing is no longer the timing of a normal run.
#define GFXSTREAM_STALL_PRINT(...) GFXSTREAM_DIAG_PRINT(__VA_ARGS__)

// A defensive guard firing is evidence, not safety. Each of these was added to stop a crash whose
// cause was never established, so the tree cannot tell "the guard is insurance" from "the bug is
// still live and the guard is hiding it". Say so, once per site, whether or not diagnostics are
// on -- a run that prints none of these is the only version of "fixed" worth believing.
#define GFXSTREAM_GUARD_FIRED(name, ...)                                                 \
    do {                                                                                 \
        static std::atomic<bool> reported_{false};                                       \
        bool expected_ = false;                                                          \
        if (reported_.compare_exchange_strong(expected_, true)) {                         \
            fprintf(stderr, "GUARD-FIRED[%s]: ", name);                                  \
            fprintf(stderr, __VA_ARGS__);                                                \
            fflush(stderr);                                                              \
        }                                                                                \
    } while (0)
