/*
 * Frame Interpolation for macOS via VideoToolbox VTFrameProcessor
 *
 * Uses VTFrameRateConversion to generate synthetic intermediate frames
 * between real 30fps game frames, producing smooth 60fps output.
 *
 * All heavy work (vImageScale + VTFrameProcessor) runs asynchronously on a
 * GCD serial queue. The render loop never blocks on frame interpolation.
 * Double-buffered output ensures tear-free reads.
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

#import <Foundation/Foundation.h>
#import <VideoToolbox/VTFrameProcessor.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <stdatomic.h>

#include "frame-interp-macos.h"

// Ring of recent real frames. Three slots so the synchronous push-time
// copy never writes a slot the (at most one frame behind, coalesced)
// async VT job may still be reading.
#define RING_SIZE 3

// Cap interpolation resolution for performance. VTFrameProcessor has ~15ms
// of fixed neural engine overhead regardless of resolution.
#define INTERP_MAX_DIM 1280

static struct {
    bool initialized;
    bool available;
    int src_width;   // source IOSurface dimensions
    int src_height;
    int width;       // interpolation dimensions (may be smaller than source)
    int height;
    bool needs_scale;

    id processor; // VTFrameProcessor * (macOS 15.4+)
    id config;    // VTFrameRateConversionConfiguration * (macOS 15.4+)

    CVPixelBufferRef ring[RING_SIZE];
    atomic_int ring_head; // points to current (most recent) frame
    atomic_int frame_count;

    // Double-buffered async output
    CVPixelBufferRef interp_output[2];
    IOSurfaceRef interp_iosurface[2];
    // Packed publication: (result_fc << 1) | buffer_idx, or -1 for none.
    // A single atomic so the consumer can never pair a new index with a
    // stale frame count (or vice versa).
    atomic_int ready_state;
    dispatch_queue_t queue;    // serial queue for async processing
} g_interp;

static void do_vt_processing(int head, int fc);

static CVPixelBufferRef create_iosurface_pixel_buffer(int width, int height)
{
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferWidthKey: @(width),
        (NSString *)kCVPixelBufferHeightKey: @(height),
        (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
    };

    CVPixelBufferRef buffer = NULL;
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height,
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)attrs,
        &buffer);

    if (status != kCVReturnSuccess) {
        fprintf(stderr, "[frame-interp] Failed to create pixel buffer: %d\n",
                status);
        return NULL;
    }
    return buffer;
}

bool frame_interp_is_available(void)
{
    if (@available(macOS 15.4, *)) {
        return [VTFrameRateConversionConfiguration processorSupported];
    }
    return false;
}

bool frame_interp_init(int width, int height)
{
    // MRC: pool for the autoreleased literals created below; the render
    // thread calling this has no pool of its own.
    @autoreleasepool {
    if (g_interp.initialized) {
        frame_interp_finalize();
    }

    memset(&g_interp, 0, sizeof(g_interp));
    atomic_store(&g_interp.ready_state, -1);

    if (!frame_interp_is_available()) {
        fprintf(stderr, "[frame-interp] VTFrameRateConversion not available\n");
        return false;
    }

    if (@available(macOS 15.4, *)) {
        g_interp.src_width = width;
        g_interp.src_height = height;

        // Cap interpolation resolution for performance
        int interp_w = width, interp_h = height;
        int max_dim = MAX(interp_w, interp_h);
        if (max_dim > INTERP_MAX_DIM) {
            float scale = (float)INTERP_MAX_DIM / max_dim;
            interp_w = (int)(interp_w * scale) & ~1; // keep even
            interp_h = (int)(interp_h * scale) & ~1;
        }
        g_interp.width = interp_w;
        g_interp.height = interp_h;
        g_interp.needs_scale = (interp_w != width || interp_h != height);

        // Create frame rate conversion configuration
        VTFrameRateConversionConfiguration *config =
            [[VTFrameRateConversionConfiguration alloc]
                initWithFrameWidth:interp_w
                       frameHeight:interp_h
                usePrecomputedFlow:NO
             qualityPrioritization:VTFrameRateConversionConfigurationQualityPrioritizationNormal
                          revision:VTFrameRateConversionConfigurationRevision1];
        g_interp.config = config;
        if (!config) {
            fprintf(stderr, "[frame-interp] Failed to create FRC config\n");
            return false;
        }

        // Create processor and start session
        VTFrameProcessor *processor = [[VTFrameProcessor alloc] init];
        g_interp.processor = processor;
        NSError *error = nil;
        BOOL ok = [processor
            startSessionWithConfiguration:config
                                    error:&error];
        if (!ok) {
            fprintf(stderr, "[frame-interp] Failed to start session: %s\n",
                    error.localizedDescription.UTF8String);
            [processor release];
            g_interp.processor = nil;
            [config release];
            g_interp.config = nil;
            return false;
        }

        // Allocate ring buffer CVPixelBuffers at interpolation resolution
        for (int i = 0; i < RING_SIZE; i++) {
            g_interp.ring[i] =
                create_iosurface_pixel_buffer(interp_w, interp_h);
            if (!g_interp.ring[i]) {
                frame_interp_finalize();
                return false;
            }
        }

        // Allocate double-buffered output
        for (int i = 0; i < 2; i++) {
            g_interp.interp_output[i] =
                create_iosurface_pixel_buffer(interp_w, interp_h);
            if (!g_interp.interp_output[i]) {
                frame_interp_finalize();
                return false;
            }
            g_interp.interp_iosurface[i] =
                CVPixelBufferGetIOSurface(g_interp.interp_output[i]);
        }

        // Create serial dispatch queue for async processing
        g_interp.queue = dispatch_queue_create(
            "org.xemu.frame-interp", DISPATCH_QUEUE_SERIAL);

        g_interp.initialized = true;
        g_interp.available = true;
        atomic_store(&g_interp.frame_count, 0);
        atomic_store(&g_interp.ring_head, 0);

        fprintf(stderr,
                "[frame-interp] session started: %dx%d source, %dx%d "
                "interpolation%s\n",
                width, height, interp_w, interp_h,
                g_interp.needs_scale ? " (downscaled)" : "");
        return true;
    }

    return false;
    } // @autoreleasepool
}

void frame_interp_finalize(void)
{
    // Wait for any pending async work to complete
    if (g_interp.queue) {
        dispatch_sync(g_interp.queue, ^{});
    }

    if (@available(macOS 15.4, *)) {
        if (g_interp.processor) {
            [(VTFrameProcessor *)g_interp.processor endSession];
            [(VTFrameProcessor *)g_interp.processor release];
            g_interp.processor = nil;
        }
        [(id)g_interp.config release];
        g_interp.config = nil;
    }

    for (int i = 0; i < 2; i++) {
        if (g_interp.interp_output[i]) {
            CVPixelBufferRelease(g_interp.interp_output[i]);
            g_interp.interp_output[i] = NULL;
        }
        g_interp.interp_iosurface[i] = NULL;
    }

    for (int i = 0; i < RING_SIZE; i++) {
        if (g_interp.ring[i]) {
            CVPixelBufferRelease(g_interp.ring[i]);
            g_interp.ring[i] = NULL;
        }
    }

    if (g_interp.queue) {
        dispatch_release(g_interp.queue);
        g_interp.queue = nil;
    }

    atomic_store(&g_interp.ready_state, -1);
    g_interp.initialized = false;
    g_interp.available = false;
}

// Interpolate the midpoint of the frame pair identified by the (head, fc)
// snapshot captured at push time. Must be called from the dispatch queue;
// the snapshot guarantees we only pair ring slots whose copies have
// completed (the serial queue orders copies ahead of this call).
static void do_vt_processing(int head, int fc)
{
    @autoreleasepool {
    if (@available(macOS 15.4, *)) {
        int prev_idx = (head + RING_SIZE - 1) % RING_SIZE;
        int curr_idx = head;

        // Write to the buffer NOT currently being displayed
        int ready = atomic_load(&g_interp.ready_state);
        int write_idx = (ready >= 0 && (ready & 1) == 0) ? 1 : 0;

        CMTime prev_time = CMTimeMake(fc - 1, 30);
        CMTime curr_time = CMTimeMake(fc, 30);

        // MRC: everything created here is autoreleased and drained by the
        // enclosing pool when this (synchronous) VT pass returns.
        VTFrameProcessorFrame *src_frame =
            [[[VTFrameProcessorFrame alloc]
                initWithBuffer:g_interp.ring[prev_idx]
                presentationTimeStamp:prev_time] autorelease];
        VTFrameProcessorFrame *next_frame =
            [[[VTFrameProcessorFrame alloc]
                initWithBuffer:g_interp.ring[curr_idx]
                presentationTimeStamp:curr_time] autorelease];

        if (!src_frame || !next_frame) {
            return;
        }

        VTFrameProcessorFrame *dest_frame =
            [[[VTFrameProcessorFrame alloc]
                initWithBuffer:g_interp.interp_output[write_idx]
                presentationTimeStamp:CMTimeMake(fc * 2 - 1, 60)] autorelease];

        if (!dest_frame) {
            return;
        }

        VTFrameRateConversionParameters *params =
            [[[VTFrameRateConversionParameters alloc]
                initWithSourceFrame:src_frame
                          nextFrame:next_frame
                        opticalFlow:nil
                 interpolationPhase:@[@0.5]
                     submissionMode:
                      VTFrameRateConversionParametersSubmissionModeSequential
                  destinationFrames:@[dest_frame]] autorelease];

        if (!params) {
            return;
        }

        NSError *error = nil;
        VTFrameProcessor *processor =
            (VTFrameProcessor *)g_interp.processor;
        BOOL ok = [processor processWithParameters:params
                                             error:&error];
        if (ok) {
            // Atomically publish the new result with its frame count
            atomic_store(&g_interp.ready_state, (fc << 1) | write_idx);
        }
    }
    } // @autoreleasepool
}

// Copy IOSurface data into the ring buffer slot (with optional downscale).
// Must be called from the dispatch queue.
static void do_push_copy(IOSurfaceRef surface, int head)
{
    CVPixelBufferRef dst = g_interp.ring[head];

    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    CVPixelBufferLockBaseAddress(dst, 0);

    if (g_interp.needs_scale) {
        vImage_Buffer src_buf = {
            .data = IOSurfaceGetBaseAddress(surface),
            .width = (vImagePixelCount)IOSurfaceGetWidth(surface),
            .height = (vImagePixelCount)IOSurfaceGetHeight(surface),
            .rowBytes = IOSurfaceGetBytesPerRow(surface),
        };
        vImage_Buffer dst_buf = {
            .data = CVPixelBufferGetBaseAddress(dst),
            .width = (vImagePixelCount)g_interp.width,
            .height = (vImagePixelCount)g_interp.height,
            .rowBytes = CVPixelBufferGetBytesPerRow(dst),
        };
        vImageScale_ARGB8888(&src_buf, &dst_buf, NULL, kvImageNoFlags);
    } else {
        void *src_ptr = IOSurfaceGetBaseAddress(surface);
        void *dst_ptr = CVPixelBufferGetBaseAddress(dst);
        size_t src_stride = IOSurfaceGetBytesPerRow(surface);
        size_t dst_stride = CVPixelBufferGetBytesPerRow(dst);
        size_t copy_height = MIN((size_t)g_interp.height,
                                 IOSurfaceGetHeight(surface));
        size_t copy_width_bytes = MIN(src_stride, dst_stride);

        for (size_t y = 0; y < copy_height; y++) {
            memcpy((uint8_t *)dst_ptr + y * dst_stride,
                   (uint8_t *)src_ptr + y * src_stride,
                   copy_width_bytes);
        }
    }

    CVPixelBufferUnlockBaseAddress(dst, 0);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

void frame_interp_push_frame(IOSurfaceRef surface)
{
    if (!g_interp.initialized || !surface) {
        return;
    }

    // Advance ring head and frame count atomically (main thread only)
    int head = (atomic_load(&g_interp.ring_head) + 1) % RING_SIZE;
    atomic_store(&g_interp.ring_head, head);
    int fc = atomic_fetch_add(&g_interp.frame_count, 1) + 1;

    // Copy the pixels NOW, while the caller still holds the display
    // surface between nv2a_get_framebuffer_surface and its release —
    // a deferred copy could read a surface Vulkan has already
    // overwritten with the next frame. ~1ms at 1280x960 on Apple
    // Silicon. The 3-slot ring guarantees this write never lands in a
    // slot the (at most one pair behind, coalesced) VT job still reads.
    do_push_copy(surface, head);

    if (fc < 2) {
        return;
    }

    // Only the ANE inference runs async. Capture head and fc by value —
    // the queue may run this block after further pushes have advanced
    // the live counters.
    dispatch_async(g_interp.queue, ^{
        if (!g_interp.initialized) {
            return;
        }

        // Coalesce: if a newer frame has already been pushed, skip this
        // pair — the newer push's block will interpolate the newer pair.
        if (atomic_load(&g_interp.frame_count) != fc) {
            return;
        }

        do_vt_processing(head, fc);
    });
}

int frame_interp_frame_count(void)
{
    if (!g_interp.initialized) {
        return 0;
    }
    return atomic_load(&g_interp.frame_count);
}

IOSurfaceRef frame_interp_get_interpolated(int *out_fc)
{
    if (!g_interp.initialized || atomic_load(&g_interp.frame_count) < 2) {
        return NULL;
    }

    // Non-blocking. Only return the midpoint of the newest pushed pair:
    // any older midpoint lies temporally behind a frame the caller has
    // already displayed and would step motion backwards.
    int state = atomic_load(&g_interp.ready_state);
    if (state < 0) {
        return NULL;
    }
    int result_fc = state >> 1;
    if (result_fc != atomic_load(&g_interp.frame_count)) {
        return NULL;
    }
    if (out_fc) {
        *out_fc = result_fc;
    }
    return g_interp.interp_iosurface[state & 1];
}
