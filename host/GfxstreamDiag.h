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
