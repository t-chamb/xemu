/*
 * In-process texture upscaling on the Apple Neural Engine
 *
 * A background worker runs textures through VideoToolbox's
 * super-resolution scaler (4x) the first time they are seen, writes the
 * result into the texrep replacement store, and signals the render thread
 * so the binding is recreated with the upscaled image — no batch step.
 *
 * Copyright (c) 2026 xemu contributors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HW_XBOX_NV2A_PGRAPH_VK_TEXREP_ANE_MACOS_H
#define HW_XBOX_NV2A_PGRAPH_VK_TEXREP_ANE_MACOS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when the super-resolution scaler exists on this system (Apple
 * Silicon, macOS 26+) and its model assets are usable. */
bool texrep_ane_available(void);

/* Queue one texture for background upscaling. Takes ownership of rgba
 * (g_malloc'd, width*height*4, straight-alpha canonical RGBA) on both
 * outcomes. png_path is copied. Returns false when the job is rejected
 * (queue saturated) so the caller can leave the texture eligible for a
 * later re-offer. On success the worker writes png_path and calls
 * texrep_ane_mark_ready(content_hash); retryable worker-side failures
 * (model still downloading, allocation) call texrep_ane_mark_dropped
 * instead. */
bool texrep_ane_submit(uint64_t content_hash, uint8_t *rgba, int width,
                       int height, const char *png_path);

/* Abandon queued jobs, wait for the in-flight one, release sessions. */
void texrep_ane_finalize(void);

/* Implemented by texrep.c: called from the worker thread when a
 * replacement PNG has been written and verified. */
void texrep_ane_mark_ready(uint64_t content_hash);

/* Implemented by texrep.c: called from the worker thread when a job was
 * abandoned for a retryable reason; the texture becomes eligible for
 * re-offer on a later upload. */
void texrep_ane_mark_dropped(uint64_t content_hash);

#ifdef __cplusplus
}
#endif

#endif
