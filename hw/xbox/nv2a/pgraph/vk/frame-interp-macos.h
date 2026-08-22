/*
 * Frame Interpolation for macOS via VideoToolbox VTFrameProcessor
 *
 * Copyright (c) 2025 xemu contributors
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

#ifndef FRAME_INTERP_MACOS_H
#define FRAME_INTERP_MACOS_H

#include <stdbool.h>
#include <IOSurface/IOSurfaceRef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Check if VTFrameRateConversion is supported on this system */
bool frame_interp_is_available(void);

/* Initialize frame interpolation for given dimensions.
 * Returns true on success. */
bool frame_interp_init(int width, int height);

/* Tear down frame interpolation resources */
void frame_interp_finalize(void);

/* Report the current display drawable size; caps interpolation resolution
 * (midpoints are never presented above it). Safe to call every frame. */
void frame_interp_set_display_size(int width, int height);

/* Pin interpolation resolution to the largest ladder rung <= max_dim and
 * disable adaptation; 0 restores adaptive behavior. */
void frame_interp_set_quality_cap(int max_dim);

/* Push a new real frame's IOSurface into the ring buffer */
void frame_interp_push_frame(IOSurfaceRef surface);

/* Number of frames pushed since init (0 if not initialized) */
int frame_interp_frame_count(void);

/* Current interpolation dimensions; false when not initialized. */
bool frame_interp_get_size(int *width, int *height);

/* Midpoint of the two most recently pushed frames, or NULL if it has not
 * finished computing yet (never returns an older pair's midpoint — showing
 * one would step motion backwards). On success *out_fc receives the frame
 * count identifying the pair. The returned IOSurface is owned by the
 * interpolator and remains valid until the next newer result publishes. */
IOSurfaceRef frame_interp_get_interpolated(int *out_fc);

#ifdef __cplusplus
}
#endif

#endif /* FRAME_INTERP_MACOS_H */
