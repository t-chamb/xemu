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

/* Interpolation resolution adapts at runtime. Two caps apply on top of the
 * source dimensions:
 *
 *  - The display drawable size: midpoints are only ever presented at window
 *    resolution, so interpolating above it is wasted inference.
 *  - A performance rung, moved along the ladder below. The consumer
 *    queries results just before pushing the next real frame, so a
 *    midpoint has one full 30fps period (~33ms) from push to consumption;
 *    past that it goes stale and cadence degrades to 30fps. Inference has
 *    a large fixed ANE overhead, so cost is only weakly elastic in
 *    resolution — stepping down on inference time alone destroys quality
 *    without buying anything. The rung therefore steps down only when
 *    midpoints measurably go stale (miss-rate EMA), and steps up only
 *    when misses are absent and the inference-time EMA, scaled by the
 *    area ratio of the next rung (a conservative overestimate given the
 *    fixed overhead), still fits the budget.
 *
 * Adaptation state deliberately lives outside g_interp: session re-inits
 * memset g_interp, and the learned rung must survive them.
 */
static const int adapt_ladder[] = { 640, 960, 1280, 1600, 1920, 2560, 3200 };
#define ADAPT_LADDER_LEN ((int)(sizeof(adapt_ladder) / sizeof(adapt_ladder[0])))
#define ADAPT_START_IDX 2            /* 1280: known safe on every ANE */
#define ADAPT_MIN_SAMPLES 60         /* pairs (~2s) between rung decisions */
#define ADAPT_AUTO_FLOOR_IDX 1       /* auto never drops below 960 */
#define ADAPT_MISS_STEP_DOWN 0.30    /* miss-rate EMA to step down */
#define ADAPT_MISS_STEP_UP 0.02      /* miss-rate EMA to allow step up */
#define ADAPT_BUDGET_MS 30.0         /* usable slice of the ~33ms period */

static struct {
    _Atomic int perf_idx;            /* ladder index; render thread writes */
    _Atomic int display_w;           /* last known drawable size */
    _Atomic int display_h;
    _Atomic int ema_us;              /* inference EMA, written on VT queue */
    double miss_ema;                 /* render thread only */
    int samples;                     /* render thread only */
    _Atomic bool pinned;             /* UI pinned a rung; stop adapting */
} g_adapt = { .perf_idx = ADAPT_START_IDX };

/* Round a display dimension up to a ladder rung so live window resizing
 * only changes the target when it crosses a rung boundary. */
static int ladder_ceil(int dim)
{
    for (int i = 0; i < ADAPT_LADDER_LEN; i++) {
        if (adapt_ladder[i] >= dim) {
            return adapt_ladder[i];
        }
    }
    return adapt_ladder[ADAPT_LADDER_LEN - 1];
}

static void compute_interp_dims(int src_w, int src_h, int *out_w, int *out_h)
{
    int cap = adapt_ladder[atomic_load(&g_adapt.perf_idx)];
    int disp_w = atomic_load(&g_adapt.display_w);
    int disp_h = atomic_load(&g_adapt.display_h);
    if (disp_w > 0 && disp_h > 0) {
        cap = MIN(cap, ladder_ceil(MAX(disp_w, disp_h)));
    }

    int w = src_w, h = src_h;
    int max_dim = MAX(w, h);
    if (max_dim > cap) {
        float scale = (float)cap / max_dim;
        w = (int)(w * scale) & ~1; /* keep even */
        h = (int)(h * scale) & ~1;
    }
    *out_w = w;
    *out_h = h;
}

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

/* VT queue thread: record inference wall time. */
static void adapt_note_sample(double ms)
{
    int prev = atomic_load(&g_adapt.ema_us);
    int cur = (int)(ms * 1000.0);
    atomic_store(&g_adapt.ema_us,
                 prev == 0 ? cur : prev + (cur - prev) / 5);
}

/* Render thread: record whether the previous pair's midpoint was ready in
 * time to be consumed (miss = it went stale), and move the rung. */
static void adapt_note_consumption(bool miss)
{
    if (atomic_load(&g_adapt.pinned)) {
        return;
    }
    g_adapt.miss_ema += 0.1 * ((miss ? 1.0 : 0.0) - g_adapt.miss_ema);
    if (++g_adapt.samples < ADAPT_MIN_SAMPLES) {
        return;
    }

    int idx = atomic_load(&g_adapt.perf_idx);
    double ema_ms = atomic_load(&g_adapt.ema_us) / 1000.0;
    int session_dim = MAX(g_interp.width, g_interp.height);

    if (g_adapt.miss_ema > ADAPT_MISS_STEP_DOWN &&
        idx > ADAPT_AUTO_FLOOR_IDX) {
        atomic_store(&g_adapt.perf_idx, idx - 1);
        fprintf(stderr,
                "[frame-interp] midpoints stale (%.0f%%, %.1fms): "
                "cap %d -> %d\n",
                g_adapt.miss_ema * 100.0, ema_ms, adapt_ladder[idx],
                adapt_ladder[idx - 1]);
        g_adapt.miss_ema = 0.0;
    } else if (g_adapt.miss_ema < ADAPT_MISS_STEP_UP &&
               idx + 1 < ADAPT_LADDER_LEN &&
               session_dim >= adapt_ladder[idx]) {
        /* Only raise while the rung is the binding constraint, and only
         * when the next rung's estimated cost (area-scaled — conservative,
         * since much of the cost is fixed overhead) still fits the budget.
         */
        double ratio = (double)adapt_ladder[idx + 1] * adapt_ladder[idx + 1] /
                       ((double)adapt_ladder[idx] * adapt_ladder[idx]);
        if (ema_ms > 0.0 && ema_ms * ratio < ADAPT_BUDGET_MS) {
            atomic_store(&g_adapt.perf_idx, idx + 1);
            fprintf(stderr,
                    "[frame-interp] headroom (%.1fms): cap %d -> %d\n",
                    ema_ms, adapt_ladder[idx], adapt_ladder[idx + 1]);
            g_adapt.miss_ema = 0.0;
        }
    }
    g_adapt.samples = 0;
}

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

        int interp_w, interp_h;
        compute_interp_dims(width, height, &interp_w, &interp_h);
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
        uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        BOOL ok = [processor processWithParameters:params
                                             error:&error];
        if (ok) {
            adapt_note_sample(
                (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) / 1e6);
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

void frame_interp_set_display_size(int width, int height)
{
    if (width > 0 && height > 0) {
        atomic_store(&g_adapt.display_w, width);
        atomic_store(&g_adapt.display_h, height);
    }
}

/* 0 pins nothing (adaptive); otherwise clamp the ladder to the largest rung
 * not exceeding max_dim and stop adapting. Called from the settings UI.
 */
void frame_interp_set_quality_cap(int max_dim)
{
    static int applied = -1;
    if (max_dim == applied) {
        return;
    }
    applied = max_dim;
    g_adapt.pinned = max_dim > 0;
    if (g_adapt.pinned) {
        int idx = 0;
        while (idx + 1 < ADAPT_LADDER_LEN && adapt_ladder[idx + 1] <= max_dim) {
            idx++;
        }
        atomic_store(&g_adapt.perf_idx, idx);
    }
}

void frame_interp_push_frame(IOSurfaceRef surface)
{
    if (!g_interp.initialized || !surface) {
        return;
    }

    // Re-init the session when the adaptive target (perf rung, display
    // size, or a UI-pinned cap) no longer matches the session dimensions.
    // Rate-limited so a live window resize cannot thrash sessions; each
    // re-init costs one pair of warm-up frames.
    int want_w, want_h;
    compute_interp_dims(g_interp.src_width, g_interp.src_height, &want_w,
                        &want_h);
    if (want_w != g_interp.width || want_h != g_interp.height) {
        static uint64_t last_reinit_ns;
        uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if (now - last_reinit_ns > 1000000000ull) {
            last_reinit_ns = now;
            if (!frame_interp_init(g_interp.src_width,
                                   g_interp.src_height)) {
                return;
            }
        }
    }

    // Did the previous pair's midpoint complete in time to be consumed?
    // (The consumer queried it just before this call.)
    int fc_old = atomic_load(&g_interp.frame_count);
    if (fc_old >= 2) {
        int state = atomic_load(&g_interp.ready_state);
        adapt_note_consumption(state < 0 || (state >> 1) != fc_old);
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
