/*
 * In-process texture upscaling on the Apple Neural Engine
 *
 * See texrep-ane-macos.h. The pixel path deliberately avoids
 * CoreGraphics entirely: input pixel buffers are filled with byte
 * shuffles from the canonical straight-alpha RGBA the renderer already
 * produced, the alpha channel is scaled with vImage, and the output PNG
 * is encoded with miniz — none of the premultiplication behaviors that
 * repeatedly bit the CG-based paths can occur here. Outputs are still
 * gated on the same statistical invariants as the batch tool.
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

#import <Foundation/Foundation.h>
#import <VideoToolbox/VTFrameProcessor.h>
#import <VideoToolbox/VTFrameProcessor_SuperResolutionScaler.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include <stdatomic.h>

#include "qemu/osdep.h"
#include "util/miniz/miniz.h"

#include "texrep-ane-macos.h"

#define ANE_SCALE 4
#define ANE_MAX_PENDING 64

typedef struct AneJob {
    uint64_t hash;
    uint8_t *rgba;
    int width, height;
    char *png_path;
} AneJob;

static struct {
    bool checked;
    bool available;
    bool model_ready;
    bool model_download_started;
    dispatch_queue_t queue;
    NSMutableDictionary *sessions; /* "WxH" -> VTFrameProcessor */
    atomic_int pending;
    int done;
} g_ane;

bool texrep_ane_available(void)
{
    if (@available(macOS 26.0, *)) {
        if (!g_ane.checked) {
            g_ane.checked = true;
            g_ane.available =
                [VTSuperResolutionScalerConfiguration isSupported];
        }
        return g_ane.available;
    }
    return false;
}

static CVPixelBufferRef create_bgra_buffer(int width, int height)
{
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{},
    };
    CVPixelBufferRef pb = NULL;
    CVReturn ret = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                       kCVPixelFormatType_32BGRA,
                                       (__bridge CFDictionaryRef)attrs, &pb);
    return ret == kCVReturnSuccess ? pb : NULL;
}

/* Straight canonical RGBA -> opaque BGRA pixel buffer (RGB only; the
 * scaler never sees alpha, so nothing can premultiply). */
static void fill_input(CVPixelBufferRef pb, const uint8_t *rgba, int width,
                       int height)
{
    CVPixelBufferLockBaseAddress(pb, 0);
    uint8_t *base = CVPixelBufferGetBaseAddress(pb);
    size_t stride = CVPixelBufferGetBytesPerRow(pb);
    for (int y = 0; y < height; y++) {
        uint8_t *row = base + y * stride;
        const uint8_t *src = rgba + (size_t)y * width * 4;
        for (int x = 0; x < width; x++) {
            row[x * 4 + 0] = src[x * 4 + 2];
            row[x * 4 + 1] = src[x * 4 + 1];
            row[x * 4 + 2] = src[x * 4 + 0];
            row[x * 4 + 3] = 255;
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
}

API_AVAILABLE(macos(26.0))
static VTFrameProcessor *session_for_size(int width, int height)
{
    NSString *key = [NSString stringWithFormat:@"%dx%d", width, height];
    VTFrameProcessor *proc = g_ane.sessions[key];
    if (proc) {
        return proc;
    }

    VTSuperResolutionScalerConfiguration *config =
        [[[VTSuperResolutionScalerConfiguration alloc]
              initWithFrameWidth:width
                     frameHeight:height
                     scaleFactor:ANE_SCALE
                       inputType:
                           VTSuperResolutionScalerConfigurationInputTypeImage
              usePrecomputedFlow:NO
           qualityPrioritization:
               VTSuperResolutionScalerConfigurationQualityPrioritizationNormal
                        revision:VTSuperResolutionScalerConfigurationRevision1]
            autorelease];
    if (!config) {
        return nil;
    }

    if (config.configurationModelStatus !=
        VTSuperResolutionScalerConfigurationModelStatusReady) {
        if (!g_ane.model_download_started) {
            g_ane.model_download_started = true;
            fprintf(stderr, "[texrep-ane] downloading model assets\n");
            [config downloadConfigurationModelWithCompletionHandler:^(
                        NSError *error) {
                if (error) {
                    fprintf(stderr, "[texrep-ane] model download failed\n");
                } else {
                    g_ane.model_ready = true;
                }
            }];
        }
        return nil; /* jobs are dropped until the model is present */
    }

    VTFrameProcessor *processor = [[[VTFrameProcessor alloc] init] autorelease];
    NSError *error = nil;
    if (![processor startSessionWithConfiguration:config error:&error]) {
        fprintf(stderr, "[texrep-ane] startSession %dx%d failed\n", width,
                height);
        return nil;
    }
    g_ane.sessions[key] = processor; /* dictionary retains */
    return processor;
}

/* Same invariant gate as the batch tool: an upscale must not materially
 * change channel statistics. */
static bool verify_stats(const uint8_t *src, size_t src_px,
                         const uint8_t *dst, size_t dst_px)
{
    double smean[4] = { 0 }, dmean[4] = { 0 };
    for (size_t i = 0; i < src_px; i++) {
        for (int c = 0; c < 4; c++) {
            smean[c] += src[i * 4 + c];
        }
    }
    for (size_t i = 0; i < dst_px; i++) {
        for (int c = 0; c < 4; c++) {
            dmean[c] += dst[i * 4 + c];
        }
    }
    for (int c = 0; c < 4; c++) {
        if (fabs(smean[c] / src_px - dmean[c] / dst_px) > 16.0) {
            return false;
        }
    }
    return true;
}

API_AVAILABLE(macos(26.0))
static void process_job(AneJob *job)
{
    int w = job->width, h = job->height;
    int ow = w * ANE_SCALE, oh = h * ANE_SCALE;

    VTFrameProcessor *proc = session_for_size(w, h);
    if (!proc) {
        return;
    }

    CVPixelBufferRef src = create_bgra_buffer(w, h);
    CVPixelBufferRef dst = create_bgra_buffer(ow, oh);
    if (!src || !dst) {
        if (src) CVPixelBufferRelease(src);
        if (dst) CVPixelBufferRelease(dst);
        return;
    }
    fill_input(src, job->rgba, w, h);

    VTFrameProcessorFrame *src_frame =
        [[[VTFrameProcessorFrame alloc] initWithBuffer:src
                                 presentationTimeStamp:kCMTimeZero]
            autorelease];
    VTFrameProcessorFrame *dst_frame =
        [[[VTFrameProcessorFrame alloc] initWithBuffer:dst
                                 presentationTimeStamp:kCMTimeZero]
            autorelease];
    VTSuperResolutionScalerParameters *params =
        [[[VTSuperResolutionScalerParameters alloc]
             initWithSourceFrame:src_frame
                   previousFrame:nil
             previousOutputFrame:nil
                     opticalFlow:nil
                  submissionMode:
                      VTSuperResolutionScalerParametersSubmissionModeSequential
                destinationFrame:dst_frame] autorelease];

    NSError *error = nil;
    bool ok = src_frame && dst_frame && params &&
              [proc processWithParameters:params error:&error];

    if (ok) {
        /* Upscale alpha separately with vImage (Lanczos). */
        g_autofree uint8_t *alpha_src = g_malloc((size_t)w * h);
        g_autofree uint8_t *alpha_dst = g_malloc((size_t)ow * oh);
        for (size_t i = 0; i < (size_t)w * h; i++) {
            alpha_src[i] = job->rgba[i * 4 + 3];
        }
        vImage_Buffer va_src = { alpha_src, (vImagePixelCount)h,
                                 (vImagePixelCount)w, (size_t)w };
        vImage_Buffer va_dst = { alpha_dst, (vImagePixelCount)oh,
                                 (vImagePixelCount)ow, (size_t)ow };
        vImageScale_Planar8(&va_src, &va_dst, NULL, kvImageNoFlags);

        g_autofree uint8_t *out = g_malloc((size_t)ow * oh * 4);
        CVPixelBufferLockBaseAddress(dst, kCVPixelBufferLock_ReadOnly);
        const uint8_t *base = CVPixelBufferGetBaseAddress(dst);
        size_t stride = CVPixelBufferGetBytesPerRow(dst);
        for (int y = 0; y < oh; y++) {
            const uint8_t *row = base + y * stride;
            uint8_t *o = out + (size_t)y * ow * 4;
            for (int x = 0; x < ow; x++) {
                o[x * 4 + 0] = row[x * 4 + 2];
                o[x * 4 + 1] = row[x * 4 + 1];
                o[x * 4 + 2] = row[x * 4 + 0];
                o[x * 4 + 3] = alpha_dst[(size_t)y * ow + x];
            }
        }
        CVPixelBufferUnlockBaseAddress(dst, kCVPixelBufferLock_ReadOnly);

        if (verify_stats(job->rgba, (size_t)w * h, out, (size_t)ow * oh)) {
            size_t png_size = 0;
            void *png = tdefl_write_image_to_png_file_in_memory_ex(
                out, ow, oh, 4, &png_size, MZ_DEFAULT_LEVEL, MZ_FALSE);
            if (png) {
                if (g_file_set_contents(job->png_path, png, png_size, NULL)) {
                    texrep_ane_mark_ready(job->hash);
                    g_ane.done++;
                    if (g_ane.done == 1 || g_ane.done % 50 == 0) {
                        fprintf(stderr, "[texrep-ane] %d textures upscaled\n",
                                g_ane.done);
                    }
                }
                mz_free(png);
            }
        } else {
            fprintf(stderr,
                    "[texrep-ane] %016" PRIx64 ": output failed statistical "
                    "verification, discarded\n", job->hash);
        }
    }

    CVPixelBufferRelease(src);
    CVPixelBufferRelease(dst);
}

void texrep_ane_submit(uint64_t content_hash, uint8_t *rgba, int width,
                       int height, const char *png_path)
{
    if (!texrep_ane_available() ||
        atomic_load(&g_ane.pending) >= ANE_MAX_PENDING) {
        g_free(rgba);
        return;
    }

    if (!g_ane.queue) {
        g_ane.queue =
            dispatch_queue_create("org.xemu.texrep-ane",
                                  DISPATCH_QUEUE_SERIAL);
        g_ane.sessions = [[NSMutableDictionary alloc] init];
    }

    AneJob *job = g_malloc0(sizeof(*job));
    job->hash = content_hash;
    job->rgba = rgba;
    job->width = width;
    job->height = height;
    job->png_path = g_strdup(png_path);

    atomic_fetch_add(&g_ane.pending, 1);
    dispatch_async(g_ane.queue, ^{
        @autoreleasepool {
            if (@available(macOS 26.0, *)) {
                process_job(job);
            }
        }
        g_free(job->rgba);
        g_free(job->png_path);
        g_free(job);
        atomic_fetch_sub(&g_ane.pending, 1);
    });
}

void texrep_ane_finalize(void)
{
    if (g_ane.queue) {
        dispatch_sync(g_ane.queue, ^{});
        dispatch_release(g_ane.queue);
        g_ane.queue = nil;
    }
    if (@available(macOS 26.0, *)) {
        for (NSString *key in g_ane.sessions) {
            [(VTFrameProcessor *)g_ane.sessions[key] endSession];
        }
    }
    [g_ane.sessions release];
    g_ane.sessions = nil;
}
