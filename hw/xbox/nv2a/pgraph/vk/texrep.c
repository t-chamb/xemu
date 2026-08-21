/*
 * Content-addressed texture dump and replacement pipeline
 *
 * See texrep.h for the design overview.
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

#include "qemu/osdep.h"
#include "ui/xemu-settings.h"

#include "texrep.h"

#include "ui/thirdparty/stb_image/stb_image.h"
#include "util/miniz/miniz.h"

/* Keep the packed mip chain comfortably inside the 64MB staging buffer. */
#define TEXREP_MAX_REPLACEMENT_DIM 2048

static struct {
    bool initialized;
    char *dump_dir;
    char *replace_dir;
    GHashTable *cache;      /* hash -> TexRepImage* (NULL = known absent) */
    GHashTable *dynamic;    /* vram_offset -> present */
    int num_replaced;
    int num_dumped;
} g_texrep;

static char *hash_path(const char *dir, uint64_t hash)
{
    char name[32];
    snprintf(name, sizeof(name), "%016" PRIx64 ".png", hash);
    return g_build_filename(dir, name, NULL);
}

void texrep_init(void)
{
    if (g_texrep.initialized) {
        return;
    }
    const char *base = xemu_settings_get_base_path();
    if (!base) {
        return;
    }
    char *root = g_build_filename(base, "textures", NULL);
    g_texrep.dump_dir = g_build_filename(root, "dump", NULL);
    g_texrep.replace_dir = g_build_filename(root, "replace", NULL);
    g_free(root);

    g_texrep.cache = g_hash_table_new(g_int64_hash, g_int64_equal);
    g_texrep.dynamic = g_hash_table_new(g_int64_hash, g_int64_equal);
    g_texrep.initialized = true;
}

static void free_cache_entry(gpointer key, gpointer value, gpointer opaque)
{
    g_free(key);
    TexRepImage *img = value;
    if (img) {
        g_free(img->data);
        g_free(img);
    }
}

static void free_dynamic_entry(gpointer key, gpointer value, gpointer opaque)
{
    g_free(key);
}

void texrep_finalize(void)
{
    if (!g_texrep.initialized) {
        return;
    }
    g_hash_table_foreach(g_texrep.cache, free_cache_entry, NULL);
    g_hash_table_destroy(g_texrep.cache);
    g_hash_table_foreach(g_texrep.dynamic, free_dynamic_entry, NULL);
    g_hash_table_destroy(g_texrep.dynamic);
    g_free(g_texrep.dump_dir);
    g_free(g_texrep.replace_dir);
    memset(&g_texrep, 0, sizeof(g_texrep));
}

bool texrep_replace_enabled(void)
{
    return g_texrep.initialized && g_config.display.texture_pipeline.replace;
}

bool texrep_dump_enabled(void)
{
    return g_texrep.initialized && g_config.display.texture_pipeline.dump;
}

void texrep_mark_dynamic(uint64_t vram_offset)
{
    if (!g_texrep.initialized ||
        g_hash_table_contains(g_texrep.dynamic, &vram_offset)) {
        return;
    }
    uint64_t *key = g_memdup2(&vram_offset, sizeof(vram_offset));
    g_hash_table_add(g_texrep.dynamic, key);
}

bool texrep_is_dynamic(uint64_t vram_offset)
{
    return g_texrep.initialized &&
           g_hash_table_contains(g_texrep.dynamic, &vram_offset);
}

/* Box-filter one RGBA8 mip level into the next (floor dimension halving,
 * clamping so odd and 1-wide levels stay defined). */
static void downsample_level(const uint8_t *src, int sw, int sh, uint8_t *dst,
                             int dw, int dh)
{
    for (int y = 0; y < dh; y++) {
        int sy0 = MIN(y * 2, sh - 1), sy1 = MIN(y * 2 + 1, sh - 1);
        for (int x = 0; x < dw; x++) {
            int sx0 = MIN(x * 2, sw - 1), sx1 = MIN(x * 2 + 1, sw - 1);
            const uint8_t *p00 = src + (sy0 * sw + sx0) * 4;
            const uint8_t *p01 = src + (sy0 * sw + sx1) * 4;
            const uint8_t *p10 = src + (sy1 * sw + sx0) * 4;
            const uint8_t *p11 = src + (sy1 * sw + sx1) * 4;
            uint8_t *d = dst + (y * dw + x) * 4;
            for (int c = 0; c < 4; c++) {
                d[c] = (p00[c] + p01[c] + p10[c] + p11[c] + 2) / 4;
            }
        }
    }
}

static TexRepImage *build_image(uint8_t *rgba, int w, int h, TexRepOrder order)
{
    if (order == TEXREP_ORDER_BGRA8) {
        for (size_t i = 0; i < (size_t)w * h * 4; i += 4) {
            uint8_t t = rgba[i];
            rgba[i] = rgba[i + 2];
            rgba[i + 2] = t;
        }
    }

    TexRepImage *img = g_malloc0(sizeof(*img));
    img->width = w;
    img->height = h;

    int lw = w, lh = h;
    size_t total = 0;
    while (img->levels < TEXREP_MAX_LEVELS) {
        img->level_offset[img->levels] = total;
        img->level_width[img->levels] = lw;
        img->level_height[img->levels] = lh;
        total += (size_t)lw * lh * 4;
        img->levels++;
        if (lw == 1 && lh == 1) {
            break;
        }
        lw = MAX(lw / 2, 1);
        lh = MAX(lh / 2, 1);
    }

    img->data = g_malloc(total);
    img->data_size = total;
    memcpy(img->data, rgba, (size_t)w * h * 4);
    for (int i = 1; i < img->levels; i++) {
        downsample_level(img->data + img->level_offset[i - 1],
                         img->level_width[i - 1], img->level_height[i - 1],
                         img->data + img->level_offset[i],
                         img->level_width[i], img->level_height[i]);
    }
    return img;
}

const TexRepImage *texrep_lookup(uint64_t content_hash, TexRepOrder order)
{
    if (!texrep_replace_enabled()) {
        return NULL;
    }

    gpointer value;
    if (g_hash_table_lookup_extended(g_texrep.cache, &content_hash, NULL,
                                     &value)) {
        return value; /* may be the cached-negative NULL */
    }

    TexRepImage *img = NULL;
    char *path = hash_path(g_texrep.replace_dir, content_hash);
    int w = 0, h = 0, channels = 0;
    uint8_t *rgba = stbi_load(path, &w, &h, &channels, 4);
    if (rgba) {
        if (w > 0 && h > 0 && w <= TEXREP_MAX_REPLACEMENT_DIM &&
            h <= TEXREP_MAX_REPLACEMENT_DIM) {
            img = build_image(rgba, w, h, order);
            g_texrep.num_replaced++;
            if (g_texrep.num_replaced == 1) {
                fprintf(stderr, "[texrep] replacements active (%s)\n",
                        g_texrep.replace_dir);
            }
        } else {
            fprintf(stderr,
                    "[texrep] %s: %dx%d exceeds max dimension %d, ignored\n",
                    path, w, h, TEXREP_MAX_REPLACEMENT_DIM);
        }
        stbi_image_free(rgba);
    }
    g_free(path);

    uint64_t *key = g_memdup2(&content_hash, sizeof(content_hash));
    g_hash_table_insert(g_texrep.cache, key, img);
    return img;
}

static void convert_to_rgba(TexRepDumpFormat fmt, const void *src, int count,
                            uint8_t *dst)
{
    const uint8_t *s8 = src;
    const uint16_t *s16 = src;
    switch (fmt) {
    case TEXREP_DUMP_RGBA8:
        memcpy(dst, src, (size_t)count * 4);
        break;
    case TEXREP_DUMP_BGRA8:
        for (int i = 0; i < count; i++) {
            dst[i * 4 + 0] = s8[i * 4 + 2];
            dst[i * 4 + 1] = s8[i * 4 + 1];
            dst[i * 4 + 2] = s8[i * 4 + 0];
            dst[i * 4 + 3] = s8[i * 4 + 3];
        }
        break;
    case TEXREP_DUMP_R5G6B5:
        for (int i = 0; i < count; i++) {
            uint16_t p = s16[i];
            dst[i * 4 + 0] = ((p >> 11) & 0x1f) * 255 / 31;
            dst[i * 4 + 1] = ((p >> 5) & 0x3f) * 255 / 63;
            dst[i * 4 + 2] = (p & 0x1f) * 255 / 31;
            dst[i * 4 + 3] = 255;
        }
        break;
    case TEXREP_DUMP_A1R5G5B5:
        for (int i = 0; i < count; i++) {
            uint16_t p = s16[i];
            dst[i * 4 + 0] = ((p >> 10) & 0x1f) * 255 / 31;
            dst[i * 4 + 1] = ((p >> 5) & 0x1f) * 255 / 31;
            dst[i * 4 + 2] = (p & 0x1f) * 255 / 31;
            dst[i * 4 + 3] = (p & 0x8000) ? 255 : 0;
        }
        break;
    case TEXREP_DUMP_A4R4G4B4:
        for (int i = 0; i < count; i++) {
            uint16_t p = s16[i];
            dst[i * 4 + 0] = ((p >> 8) & 0xf) * 17;
            dst[i * 4 + 1] = ((p >> 4) & 0xf) * 17;
            dst[i * 4 + 2] = (p & 0xf) * 17;
            dst[i * 4 + 3] = ((p >> 12) & 0xf) * 17;
        }
        break;
    }
}

void texrep_dump(uint64_t content_hash, TexRepDumpFormat fmt, int width,
                 int height, const void *level0_data)
{
    if (!texrep_dump_enabled() || width <= 0 || height <= 0) {
        return;
    }

    char *path = hash_path(g_texrep.dump_dir, content_hash);
    if (g_file_test(path, G_FILE_TEST_EXISTS)) {
        g_free(path);
        return;
    }
    if (g_mkdir_with_parents(g_texrep.dump_dir, 0755) != 0) {
        g_free(path);
        return;
    }

    g_autofree uint8_t *rgba = g_malloc((size_t)width * height * 4);
    convert_to_rgba(fmt, level0_data, width * height, rgba);

    size_t png_size = 0;
    void *png = tdefl_write_image_to_png_file_in_memory_ex(
        rgba, width, height, 4, &png_size, MZ_DEFAULT_LEVEL, MZ_FALSE);
    if (png) {
        g_file_set_contents(path, png, png_size, NULL);
        mz_free(png);
        g_texrep.num_dumped++;
        if (g_texrep.num_dumped == 1) {
            fprintf(stderr, "[texrep] dumping textures to %s\n",
                    g_texrep.dump_dir);
        }
    }
    g_free(path);
}
