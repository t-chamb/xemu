/*
 * Content-addressed texture dump and replacement pipeline
 *
 * Textures are identified solely by the content hash xemu already computes
 * over the guest's raw texture (and palette) bytes — the pipeline is
 * game-agnostic by construction. On-disk layout under the xemu base path:
 *
 *   textures/dump/<hash>.png      originals, canonical RGBA8, level 0
 *   textures/replace/<hash>.png   replacements, any resolution
 *
 * A replacement PNG is loaded once, converted to the texture's native
 * channel order, given a full box-filtered mip chain, and cached for the
 * session. Textures whose guest data is observed to change (render
 * targets are excluded upstream; CPU-animated textures are caught by
 * address) are denylisted from replacement for the session.
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

#ifndef HW_XBOX_NV2A_PGRAPH_VK_TEXREP_H
#define HW_XBOX_NV2A_PGRAPH_VK_TEXREP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native channel order of the target VkFormat; the canonical interchange
 * format (PNG side) is always RGBA8. */
typedef enum TexRepOrder {
    TEXREP_ORDER_RGBA8,
    TEXREP_ORDER_BGRA8,
} TexRepOrder;

/* Source pixel format for dumping (converted to canonical RGBA8). */
typedef enum TexRepDumpFormat {
    TEXREP_DUMP_RGBA8,
    TEXREP_DUMP_BGRA8,
    TEXREP_DUMP_R5G6B5,
    TEXREP_DUMP_A1R5G5B5,
    TEXREP_DUMP_A4R4G4B4,
} TexRepDumpFormat;

#define TEXREP_MAX_LEVELS 16

typedef struct TexRepImage {
    int width, height;
    int levels;                /* full mip chain length (per face) */
    int faces;                 /* 1 for 2D, 6 for cubemaps */
    uint8_t *data;             /* packed chain(s), native channel order */
    size_t data_size;
    size_t face_stride;        /* bytes per face chain (== data_size / faces) */
    size_t level_offset[TEXREP_MAX_LEVELS];   /* within a face chain */
    int level_width[TEXREP_MAX_LEVELS];
    int level_height[TEXREP_MAX_LEVELS];
} TexRepImage;

void texrep_init(void);
void texrep_finalize(void);

bool texrep_replace_enabled(void);
bool texrep_dump_enabled(void);

/* Returns a session-cached replacement image for the given content hash,
 * or NULL. The returned pointer stays valid until texrep_finalize(). */
const TexRepImage *texrep_lookup(uint64_t content_hash, TexRepOrder order);

/* Cubemap variant: requires all six <hash>_face0..5.png files with
 * identical square dimensions. */
const TexRepImage *texrep_lookup_cube(uint64_t content_hash,
                                      TexRepOrder order);

/* Mark a guest texture address as dynamic; its contents were observed to
 * change in place, so it must never be replaced this session. */
void texrep_mark_dynamic(uint64_t vram_offset);
bool texrep_is_dynamic(uint64_t vram_offset);

/* Write the (level 0) original out as a canonical RGBA8 PNG, once per
 * hash. No-op if the file already exists. force_opaque must be set for
 * formats whose alpha channel is ignored by the view swizzle (X8R8G8B8
 * style): their stored alpha bytes are junk (frequently zero), and
 * propagating them into the PNG makes every alpha-aware scaler
 * premultiply the color channels to black. */
/* face < 0 writes <hash>.png (2D); face 0..5 writes <hash>_faceN.png. */
void texrep_dump(uint64_t content_hash, TexRepDumpFormat fmt, int width,
                 int height, const void *level0_data, bool force_opaque,
                 int face);

/* Offer a texture to the in-process ANE upscaler (macOS; no-op elsewhere
 * or when display.texture_pipeline.auto_upscale is off). */
void texrep_auto_upscale(uint64_t content_hash, TexRepDumpFormat fmt,
                         int width, int height, const void *level0_data,
                         bool force_opaque);

/* Render thread: true once for a hash whose background upscale finished;
 * also invalidates the session's negative lookup so the fresh file loads. */
bool texrep_take_ready(uint64_t content_hash);

#ifdef __cplusplus
}
#endif

#endif
