#
# Copyright (c) 2026 Hajime Nakagami
#
# This file is part of rdpy.
#
# rdpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
H.264/AVC decoder for RDPGFX AVC420/AVC444/AVC444v2 codecs.
Uses PyAV (FFmpeg) with optional hardware acceleration
(VideoToolbox on macOS, VAAPI on Linux).

Wire formats per MS-RDPEGFX:
  AVC420 (2.2.4.4): numRegionRects(4) + regionRects(8*N) +
      quantQualityVals(2*N) + avc420EncodedBitstream(remaining)
  AVC444 (2.2.4.5): cbAvc420EncodedBitstream1(4, LC in top 2 bits) +
      avc420EncodedBitstream1 + avc420EncodedBitstream2(optional)
  AVC444v2 (2.2.4.6): same structure as AVC444, different chroma handling
"""

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
import time
import numpy as np
import rdpy.core.log as log

# Pixel formats that carry full-range (0-255) luma/chroma (YUVJ* family).
# For all other YUV formats we use limited-range BT.601 coefficients.
_FULL_RANGE_FORMATS = frozenset({'yuvj420p', 'yuvj422p', 'yuvj444p', 'yuvj440p'})

# ---------------------------------------------------------------------------
# Native C extension for YUV → BGRA conversion.
# Compiled on first use to a temp shared library; falls back to numpy if
# compilation is unavailable.
# ---------------------------------------------------------------------------
_C_SRC = r"""
#include <stdint.h>

static inline uint8_t clamp8(int v) {
    return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

/* YUV420p limited-range BT.601 -> BGRA.
 * Inner loop processes pixel pairs that share the same UV chroma sample,
 * computing the UV-dependent terms (Rb/Gb/Bb) once per pair.
 * restrict allows Clang/GCC to auto-vectorise the inner loop (NEON/AVX). */
void yuv420p_lr_to_bgra(
    const uint8_t *restrict Y, int y_stride,
    const uint8_t *restrict U, int uv_stride,
    const uint8_t *restrict V,
    uint8_t *restrict bgra, int width, int height)
{
    for (int row = 0; row < height; row++) {
        const uint8_t *Yr = Y + row * y_stride;
        const uint8_t *Ur = U + (row >> 1) * uv_stride;
        const uint8_t *Vr = V + (row >> 1) * uv_stride;
        uint8_t *dst = bgra + row * width * 4;
        int col = 0;
        /* Process pairs: both pixels share the same UV chroma sample. */
        for (; col + 1 < width; col += 2) {
            int d = (int)Ur[col >> 1] - 128;
            int e = (int)Vr[col >> 1] - 128;
            int Rb = 409*e + 128;
            int Gb = -100*d - 208*e + 128;
            int Bb = 516*d + 128;
            int c0 = ((int)Yr[col]     - 16) * 298;
            int c1 = ((int)Yr[col + 1] - 16) * 298;
            dst[0] = clamp8((c0 + Bb) >> 8); dst[1] = clamp8((c0 + Gb) >> 8);
            dst[2] = clamp8((c0 + Rb) >> 8); dst[3] = 255;
            dst[4] = clamp8((c1 + Bb) >> 8); dst[5] = clamp8((c1 + Gb) >> 8);
            dst[6] = clamp8((c1 + Rb) >> 8); dst[7] = 255;
            dst += 8;
        }
        /* Tail: odd width. */
        if (col < width) {
            int c = (int)Yr[col] - 16;
            int d = (int)Ur[col >> 1] - 128;
            int e = (int)Vr[col >> 1] - 128;
            dst[0] = clamp8((298*c + 516*d + 128) >> 8);
            dst[1] = clamp8((298*c - 100*d - 208*e + 128) >> 8);
            dst[2] = clamp8((298*c + 409*e + 128) >> 8);
            dst[3] = 255;
        }
    }
}

/* YUV420p full-range BT.601 -> BGRA (2-pixel unrolled inner loop). */
void yuv420p_fr_to_bgra(
    const uint8_t *restrict Y, int y_stride,
    const uint8_t *restrict U, int uv_stride,
    const uint8_t *restrict V,
    uint8_t *restrict bgra, int width, int height)
{
    for (int row = 0; row < height; row++) {
        const uint8_t *Yr = Y + row * y_stride;
        const uint8_t *Ur = U + (row >> 1) * uv_stride;
        const uint8_t *Vr = V + (row >> 1) * uv_stride;
        uint8_t *dst = bgra + row * width * 4;
        int col = 0;
        for (; col + 1 < width; col += 2) {
            int d = (int)Ur[col >> 1] - 128;
            int e = (int)Vr[col >> 1] - 128;
            int Rb = 359*e + 128;
            int Gb = -88*d - 183*e + 128;
            int Bb = 454*d + 128;
            int c0 = (int)Yr[col]     << 8;
            int c1 = (int)Yr[col + 1] << 8;
            dst[0] = clamp8((c0 + Bb) >> 8); dst[1] = clamp8((c0 + Gb) >> 8);
            dst[2] = clamp8((c0 + Rb) >> 8); dst[3] = 255;
            dst[4] = clamp8((c1 + Bb) >> 8); dst[5] = clamp8((c1 + Gb) >> 8);
            dst[6] = clamp8((c1 + Rb) >> 8); dst[7] = 255;
            dst += 8;
        }
        if (col < width) {
            int c = (int)Yr[col];
            int d = (int)Ur[col >> 1] - 128;
            int e = (int)Vr[col >> 1] - 128;
            dst[0] = clamp8((256*c + 454*d + 128) >> 8);
            dst[1] = clamp8((256*c - 88*d - 183*e + 128) >> 8);
            dst[2] = clamp8((256*c + 359*e + 128) >> 8);
            dst[3] = 255;
        }
    }
}

/* NV12 limited-range BT.601 -> BGRA (UV interleaved: U at even, V at odd).
 * 2-pixel unrolled: each UV pair naturally covers two horizontal pixels. */
void nv12_lr_to_bgra(
    const uint8_t *restrict Y,  int y_stride,
    const uint8_t *restrict UV, int uv_stride,
    uint8_t *restrict bgra, int width, int height)
{
    for (int row = 0; row < height; row++) {
        const uint8_t *Yr  = Y  + row * y_stride;
        const uint8_t *UVr = UV + (row >> 1) * uv_stride;
        uint8_t *dst = bgra + row * width * 4;
        int col = 0;
        for (; col + 1 < width; col += 2) {
            /* UVr[col] = U, UVr[col+1] = V for this pixel pair. */
            int d = (int)UVr[col] - 128;
            int e = (int)UVr[col + 1] - 128;
            int Rb = 409*e + 128;
            int Gb = -100*d - 208*e + 128;
            int Bb = 516*d + 128;
            int c0 = ((int)Yr[col]     - 16) * 298;
            int c1 = ((int)Yr[col + 1] - 16) * 298;
            dst[0] = clamp8((c0 + Bb) >> 8); dst[1] = clamp8((c0 + Gb) >> 8);
            dst[2] = clamp8((c0 + Rb) >> 8); dst[3] = 255;
            dst[4] = clamp8((c1 + Bb) >> 8); dst[5] = clamp8((c1 + Gb) >> 8);
            dst[6] = clamp8((c1 + Rb) >> 8); dst[7] = 255;
            dst += 8;
        }
        if (col < width) {
            int c = (int)Yr[col] - 16;
            int d = (int)UVr[col] - 128;
            int e = (int)UVr[col + 1] - 128;
            dst[0] = clamp8((298*c + 516*d + 128) >> 8);
            dst[1] = clamp8((298*c - 100*d - 208*e + 128) >> 8);
            dst[2] = clamp8((298*c + 409*e + 128) >> 8);
            dst[3] = 255;
        }
    }
}

/* Region-aware YUV420p limited-range BT.601 -> BGRA.
 * Only converts pixels within the n_rects dirty rectangles.
 * rects: flat array of [left, top, right, bottom, ...] uint16_t tuples.
 * Matches grdp's grdp_yuv420p_to_bgra_regions pattern. */
void yuv420p_lr_to_bgra_regions(
    const uint8_t *restrict Y, int y_stride,
    const uint8_t *restrict U, int uv_stride,
    const uint8_t *restrict V,
    uint8_t *restrict bgra, int width, int height,
    const uint16_t *rects, int n_rects)
{
    for (int i = 0; i < n_rects; i++) {
        int left   = (int)rects[i*4+0];
        int top    = (int)rects[i*4+1];
        int right  = (int)rects[i*4+2];
        int bottom = (int)rects[i*4+3];
        if (left   < 0)      left   = 0;
        if (top    < 0)      top    = 0;
        if (right  > width)  right  = width;
        if (bottom > height) bottom = height;
        if (left >= right || top >= bottom) continue;
        for (int row = top; row < bottom; row++) {
            const uint8_t *Yr = Y + row * y_stride;
            const uint8_t *Ur = U + (row >> 1) * uv_stride;
            const uint8_t *Vr = V + (row >> 1) * uv_stride;
            uint8_t *row_dst = bgra + row * width * 4;
            int col = left;
            /* Align to even column for the 2-pixel unrolled loop. */
            if (col & 1) {
                int d = (int)Ur[col >> 1] - 128;
                int e = (int)Vr[col >> 1] - 128;
                int c = ((int)Yr[col] - 16) * 298;
                row_dst[col*4]   = clamp8((c + 516*d + 128) >> 8);
                row_dst[col*4+1] = clamp8((c - 100*d - 208*e + 128) >> 8);
                row_dst[col*4+2] = clamp8((c + 409*e + 128) >> 8);
                row_dst[col*4+3] = 255;
                col++;
            }
            for (; col + 1 < right; col += 2) {
                int d = (int)Ur[col >> 1] - 128;
                int e = (int)Vr[col >> 1] - 128;
                int Rb = 409*e + 128;
                int Gb = -100*d - 208*e + 128;
                int Bb = 516*d + 128;
                int c0 = ((int)Yr[col]     - 16) * 298;
                int c1 = ((int)Yr[col + 1] - 16) * 298;
                row_dst[col*4]   = clamp8((c0 + Bb) >> 8);
                row_dst[col*4+1] = clamp8((c0 + Gb) >> 8);
                row_dst[col*4+2] = clamp8((c0 + Rb) >> 8);
                row_dst[col*4+3] = 255;
                row_dst[col*4+4] = clamp8((c1 + Bb) >> 8);
                row_dst[col*4+5] = clamp8((c1 + Gb) >> 8);
                row_dst[col*4+6] = clamp8((c1 + Rb) >> 8);
                row_dst[col*4+7] = 255;
            }
            if (col < right) {
                int d = (int)Ur[col >> 1] - 128;
                int e = (int)Vr[col >> 1] - 128;
                int c = ((int)Yr[col] - 16) * 298;
                row_dst[col*4]   = clamp8((c + 516*d + 128) >> 8);
                row_dst[col*4+1] = clamp8((c - 100*d - 208*e + 128) >> 8);
                row_dst[col*4+2] = clamp8((c + 409*e + 128) >> 8);
                row_dst[col*4+3] = 255;
            }
        }
    }
}

/* Region-aware YUV420p full-range BT.601 -> BGRA. */
void yuv420p_fr_to_bgra_regions(
    const uint8_t *restrict Y, int y_stride,
    const uint8_t *restrict U, int uv_stride,
    const uint8_t *restrict V,
    uint8_t *restrict bgra, int width, int height,
    const uint16_t *rects, int n_rects)
{
    for (int i = 0; i < n_rects; i++) {
        int left   = (int)rects[i*4+0];
        int top    = (int)rects[i*4+1];
        int right  = (int)rects[i*4+2];
        int bottom = (int)rects[i*4+3];
        if (left   < 0)      left   = 0;
        if (top    < 0)      top    = 0;
        if (right  > width)  right  = width;
        if (bottom > height) bottom = height;
        if (left >= right || top >= bottom) continue;
        for (int row = top; row < bottom; row++) {
            const uint8_t *Yr = Y + row * y_stride;
            const uint8_t *Ur = U + (row >> 1) * uv_stride;
            const uint8_t *Vr = V + (row >> 1) * uv_stride;
            uint8_t *row_dst = bgra + row * width * 4;
            int col = left;
            if (col & 1) {
                int d = (int)Ur[col >> 1] - 128;
                int e = (int)Vr[col >> 1] - 128;
                int c = (int)Yr[col] << 8;
                row_dst[col*4]   = clamp8((c + 454*d + 128) >> 8);
                row_dst[col*4+1] = clamp8((c -  88*d - 183*e + 128) >> 8);
                row_dst[col*4+2] = clamp8((c + 359*e + 128) >> 8);
                row_dst[col*4+3] = 255;
                col++;
            }
            for (; col + 1 < right; col += 2) {
                int d = (int)Ur[col >> 1] - 128;
                int e = (int)Vr[col >> 1] - 128;
                int Rb = 359*e + 128;
                int Gb = -88*d - 183*e + 128;
                int Bb = 454*d + 128;
                int c0 = (int)Yr[col]     << 8;
                int c1 = (int)Yr[col + 1] << 8;
                row_dst[col*4]   = clamp8((c0 + Bb) >> 8);
                row_dst[col*4+1] = clamp8((c0 + Gb) >> 8);
                row_dst[col*4+2] = clamp8((c0 + Rb) >> 8);
                row_dst[col*4+3] = 255;
                row_dst[col*4+4] = clamp8((c1 + Bb) >> 8);
                row_dst[col*4+5] = clamp8((c1 + Gb) >> 8);
                row_dst[col*4+6] = clamp8((c1 + Rb) >> 8);
                row_dst[col*4+7] = 255;
            }
            if (col < right) {
                int d = (int)Ur[col >> 1] - 128;
                int e = (int)Vr[col >> 1] - 128;
                int c = (int)Yr[col] << 8;
                row_dst[col*4]   = clamp8((c + 454*d + 128) >> 8);
                row_dst[col*4+1] = clamp8((c -  88*d - 183*e + 128) >> 8);
                row_dst[col*4+2] = clamp8((c + 359*e + 128) >> 8);
                row_dst[col*4+3] = 255;
            }
        }
    }
}

/* Region-aware NV12 limited-range BT.601 -> BGRA.
 * UV is interleaved: for luma column col, U=UV[col&~1], V=UV[(col&~1)+1].
 * Matches grdp's grdp_nv12_to_bgra_regions pattern. */
void nv12_lr_to_bgra_regions(
    const uint8_t *restrict Y,  int y_stride,
    const uint8_t *restrict UV, int uv_stride,
    uint8_t *restrict bgra, int width, int height,
    const uint16_t *rects, int n_rects)
{
    for (int i = 0; i < n_rects; i++) {
        int left   = (int)rects[i*4+0];
        int top    = (int)rects[i*4+1];
        int right  = (int)rects[i*4+2];
        int bottom = (int)rects[i*4+3];
        if (left   < 0)      left   = 0;
        if (top    < 0)      top    = 0;
        if (right  > width)  right  = width;
        if (bottom > height) bottom = height;
        if (left >= right || top >= bottom) continue;
        for (int row = top; row < bottom; row++) {
            const uint8_t *Yr  = Y  + row * y_stride;
            const uint8_t *UVr = UV + (row >> 1) * uv_stride;
            uint8_t *row_dst = bgra + row * width * 4;
            int col = left;
            /* Align to even column (UV pair boundary). */
            if (col & 1) {
                int d = (int)UVr[col & ~1] - 128;
                int e = (int)UVr[(col & ~1) + 1] - 128;
                int c = ((int)Yr[col] - 16) * 298;
                row_dst[col*4]   = clamp8((c + 516*d + 128) >> 8);
                row_dst[col*4+1] = clamp8((c - 100*d - 208*e + 128) >> 8);
                row_dst[col*4+2] = clamp8((c + 409*e + 128) >> 8);
                row_dst[col*4+3] = 255;
                col++;
            }
            for (; col + 1 < right; col += 2) {
                int d = (int)UVr[col] - 128;
                int e = (int)UVr[col + 1] - 128;
                int Rb = 409*e + 128;
                int Gb = -100*d - 208*e + 128;
                int Bb = 516*d + 128;
                int c0 = ((int)Yr[col]     - 16) * 298;
                int c1 = ((int)Yr[col + 1] - 16) * 298;
                row_dst[col*4]   = clamp8((c0 + Bb) >> 8);
                row_dst[col*4+1] = clamp8((c0 + Gb) >> 8);
                row_dst[col*4+2] = clamp8((c0 + Rb) >> 8);
                row_dst[col*4+3] = 255;
                row_dst[col*4+4] = clamp8((c1 + Bb) >> 8);
                row_dst[col*4+5] = clamp8((c1 + Gb) >> 8);
                row_dst[col*4+6] = clamp8((c1 + Rb) >> 8);
                row_dst[col*4+7] = 255;
            }
            if (col < right) {
                int d = (int)UVr[col] - 128;
                int e = (int)UVr[col + 1] - 128;
                int c = ((int)Yr[col] - 16) * 298;
                row_dst[col*4]   = clamp8((c + 516*d + 128) >> 8);
                row_dst[col*4+1] = clamp8((c - 100*d - 208*e + 128) >> 8);
                row_dst[col*4+2] = clamp8((c + 409*e + 128) >> 8);
                row_dst[col*4+3] = 255;
            }
        }
    }
}

/* YUV444 (full-res U/V) limited-range BT.601 -> BGRA.
 * Used for AVC444 LC=2 chroma-upgrade combine output.
 * U and V have the same stride as Y (all full-resolution). */
void yuv444_lr_to_bgra(
    const uint8_t *restrict Y, int y_stride,
    const uint8_t *restrict U, int u_stride,
    const uint8_t *restrict V, int v_stride,
    uint8_t *restrict bgra, int width, int height)
{
    for (int row = 0; row < height; row++) {
        const uint8_t *Yr = Y + row * y_stride;
        const uint8_t *Ur = U + row * u_stride;
        const uint8_t *Vr = V + row * v_stride;
        uint8_t *dst = bgra + row * width * 4;
        for (int col = 0; col < width; col++) {
            int c = (int)Yr[col] - 16;
            int d = (int)Ur[col] - 128;
            int e = (int)Vr[col] - 128;
            dst[col*4]   = clamp8((298*c + 516*d + 128) >> 8);
            dst[col*4+1] = clamp8((298*c - 100*d - 208*e + 128) >> 8);
            dst[col*4+2] = clamp8((298*c + 409*e + 128) >> 8);
            dst[col*4+3] = 255;
        }
    }
}

/* YUV444 (full-res U/V) full-range BT.601 -> BGRA. */
void yuv444_fr_to_bgra(
    const uint8_t *restrict Y, int y_stride,
    const uint8_t *restrict U, int u_stride,
    const uint8_t *restrict V, int v_stride,
    uint8_t *restrict bgra, int width, int height)
{
    for (int row = 0; row < height; row++) {
        const uint8_t *Yr = Y + row * y_stride;
        const uint8_t *Ur = U + row * u_stride;
        const uint8_t *Vr = V + row * v_stride;
        uint8_t *dst = bgra + row * width * 4;
        for (int col = 0; col < width; col++) {
            int y = (int)Yr[col];
            int d = (int)Ur[col] - 128;
            int e = (int)Vr[col] - 128;
            dst[col*4]   = clamp8((256*y + 454*d + 128) >> 8);
            dst[col*4+1] = clamp8((256*y -  88*d - 183*e + 128) >> 8);
            dst[col*4+2] = clamp8((256*y + 359*e + 128) >> 8);
            dst[col*4+3] = 255;
        }
    }
}
"""

_c_lib = None        # cached ctypes library (None = not yet loaded or failed)
_c_lib_loaded = False  # True once we have attempted loading


def _get_c_lib():
    """Compile and cache the native YUV→BGRA shared library.

    Returns the ctypes library on success, None on failure (e.g. no compiler).
    """
    global _c_lib, _c_lib_loaded
    if _c_lib_loaded:
        return _c_lib
    _c_lib_loaded = True
    so_path = None
    try:
        src_fd, src_path = tempfile.mkstemp(suffix='.c', prefix='rdpyqt_yuv_')
        so_fd,  so_path  = tempfile.mkstemp(suffix='.so', prefix='rdpyqt_yuv_')
        os.close(so_fd)
        with os.fdopen(src_fd, 'w') as f:
            f.write(_C_SRC)
        ret = subprocess.run(
            ['cc', '-O3', '-march=native', '-ffast-math',
             '-shared', '-fPIC', '-o', so_path, src_path],
            capture_output=True, timeout=30)
        os.unlink(src_path)
        if ret.returncode != 0:
            log.warning("AVC: native YUV->BGRA compile failed, using numpy: %s"
                        % ret.stderr.decode(errors='replace'))
            return None
        lib = ctypes.CDLL(so_path)
        ct_p = ctypes.c_void_p
        ct_i = ctypes.c_int
        lib.yuv420p_lr_to_bgra.argtypes = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_p, ct_i, ct_i]
        lib.yuv420p_fr_to_bgra.argtypes = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_p, ct_i, ct_i]
        lib.nv12_lr_to_bgra.argtypes    = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_i, ct_i]
        lib.yuv444_lr_to_bgra.argtypes  = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_i, ct_p, ct_i, ct_i]
        lib.yuv444_fr_to_bgra.argtypes  = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_i, ct_p, ct_i, ct_i]
        # Region-aware variants: same args + flat uint16 rect array + count
        ct_u16p = ctypes.c_void_p  # pointer to uint16_t array
        lib.yuv420p_lr_to_bgra_regions.argtypes = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_p, ct_i, ct_i, ct_u16p, ct_i]
        lib.yuv420p_fr_to_bgra_regions.argtypes = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_p, ct_i, ct_i, ct_u16p, ct_i]
        lib.nv12_lr_to_bgra_regions.argtypes    = [ct_p, ct_i, ct_p, ct_i, ct_p, ct_i, ct_i, ct_u16p, ct_i]
        _c_lib = lib
        log.debug("AVC: native YUV->BGRA extension loaded (%s)" % so_path)
    except Exception as e:
        log.warning("AVC: native YUV->BGRA load error: %s" % e)
    return _c_lib


def _arr_ptr(arr):
    """Return a ctypes void pointer to the start of a numpy array's data."""
    return arr.ctypes.data_as(ctypes.c_void_p)

# BT.601 color matrices for the float32 matmul path.
# Row order: R, G, B.  Applied to column vectors [Y', U', V'] where
#   limited range: Y'=Y-16,   U'=U-128, V'=V-128
#   full range:    Y'=Y,      U'=U-128, V'=V-128
def _upsample_chroma_into(buf_2d, half_plane, h, w):
    """2× nearest-neighbor upsample of *half_plane* directly into *buf_2d*.

    Writes uint8 half_plane values as float32 directly into the pre-allocated
    workspace buffer, avoiding the two temporary uint8 arrays that np.repeat
    would create.  Works correctly for odd h or w.
    """
    hh, hw = half_plane.shape           # chroma half-dimensions
    # Even positions get values from half_plane
    np.copyto(buf_2d[::2, ::2], half_plane, casting='unsafe')
    # Horizontal upsample: odd cols = adjacent even col (left).
    # n_odd_cols may be less than hw when w is odd — only copy first n_odd_cols pairs.
    n_odd_cols = w // 2
    buf_2d[::2, 1::2] = buf_2d[::2, 0:2 * n_odd_cols:2]
    # Vertical upsample: odd rows = row above.
    # Same care for odd h.
    n_odd_rows = h // 2
    buf_2d[1::2] = buf_2d[0:2 * n_odd_rows:2]


_BT601_LR_MAT = np.array([
    [1.164,  0.000,  1.596],   # R = 1.164*(Y-16) + 1.596*(V-128)
    [1.164, -0.392, -0.813],   # G = 1.164*(Y-16) - 0.392*(U-128) - 0.813*(V-128)
    [1.164,  2.017,  0.000],   # B = 1.164*(Y-16) + 2.017*(U-128)
], dtype=np.float32)

_BT601_FR_MAT = np.array([
    [1.000,  0.000,  1.402],   # R = Y + 1.402*(V-128)
    [1.000, -0.344, -0.714],   # G = Y - 0.344*(U-128) - 0.714*(V-128)
    [1.000,  1.772,  0.000],   # B = Y + 1.772*(U-128)
], dtype=np.float32)

# Per-resolution pre-allocated working buffers.
# Keys: (h, w).  Values: {'yuv': (3,N) f32, 'rgb': (3,N) f32, 'bgra': (h,w,4) u8}
# Eliminates per-frame heap allocation for the largest intermediate arrays.
# Thread-unsafe; assumes single-threaded RDPGFX decode (normal operation).
_yuv_buf_pool: dict = {}


def _get_yuv_bufs(h: int, w: int) -> dict:
    key = (h, w)
    if key not in _yuv_buf_pool:
        N = h * w
        _yuv_buf_pool[key] = {
            'yuv':  np.empty((3, N), dtype=np.float32),
            'rgb':  np.empty((3, N), dtype=np.float32),
            'bgra': np.zeros((h, w, 4), dtype=np.uint8),  # zeros: safe for region-aware partial updates
            'U444': np.empty((h, w), dtype=np.uint8),
            'V444': np.empty((h, w), dtype=np.uint8),
        }
    return _yuv_buf_pool[key]


def _bt601_yuv420_to_bgra(y, u_half, v_half, full_range):
    """Convert YUV420 planes to a pre-allocated (H, W, 4) BGRA uint8 array.

    Bypasses swscale entirely — on ARM64 (Apple Silicon) swscale's
    non-accelerated NV12/YUV420P → BGRA fallback ignores
    sws_setColorspaceDetails, producing wrong colours.  This matches the
    fix applied in grdp 0.7.7 (plugin/rdpgfx/h264_ffmpeg.go).

    Fast path: uses a native C extension (compiled at startup) that performs
    inline 2× chroma upsampling + BT.601 conversion + BGRA packing in a
    single pass — roughly 10× faster than the numpy path.

    Fallback: float32 BLAS SGEMM (numpy) when the C compiler is absent.

    The returned array is the pool's bgra buffer — callers must copy
    (e.g. via tobytes()) before the next decode call.
    """
    h, w = y.shape
    bgra = _get_yuv_bufs(h, w)['bgra']
    lib  = _get_c_lib()
    if lib is not None:
        fn = lib.yuv420p_fr_to_bgra if full_range else lib.yuv420p_lr_to_bgra
        fn(_arr_ptr(y),      y.strides[0],
           _arr_ptr(u_half), u_half.strides[0],
           _arr_ptr(v_half),
           _arr_ptr(bgra),
           w, h)
        return bgra

    # --- numpy fallback (float32 BLAS path) ---
    bufs = _get_yuv_bufs(h, w)
    yuv  = bufs['yuv']
    rgb  = bufs['rgb']
    _upsample_chroma_into(yuv[1].reshape(h, w), u_half, h, w)
    _upsample_chroma_into(yuv[2].reshape(h, w), v_half, h, w)
    np.copyto(yuv[0], y.ravel(), casting='unsafe')
    if full_range:
        yuv[1] -= 128.0
        yuv[2] -= 128.0
        np.matmul(_BT601_FR_MAT, yuv, out=rgb)
    else:
        yuv[0] -= 16.0
        yuv[1] -= 128.0
        yuv[2] -= 128.0
        np.matmul(_BT601_LR_MAT, yuv, out=rgb)
    np.clip(rgb, 0.0, 255.0, out=rgb)
    bgra[:, :, 0] = rgb[2].reshape(h, w)  # B
    bgra[:, :, 1] = rgb[1].reshape(h, w)  # G
    bgra[:, :, 2] = rgb[0].reshape(h, w)  # R
    bgra[:, :, 3] = 255
    return bgra


def _plane_to_array(plane):
    """Read a VideoPlane into a 2D uint8 numpy array, stripping line padding.

    Uses the buffer protocol directly (np.frombuffer(plane, ...)) to avoid
    the extra allocation that bytes(plane) would cause.  The resulting array
    is read-only (backed by the plane's internal buffer), which is fine since
    callers only pass it to the C conversion function for reading.
    """
    return (np.frombuffer(plane, dtype=np.uint8)
            .reshape(plane.height, plane.line_size)[:, :plane.width])


def _frame_yuv420p_to_bgra(frame, full_range):
    """Extract YUV420P planes from *frame* and return a BGRA numpy array."""
    y = _plane_to_array(frame.planes[0])
    u = _plane_to_array(frame.planes[1])
    v = _plane_to_array(frame.planes[2])
    return _bt601_yuv420_to_bgra(y, u, v, full_range)


def _frame_nv12_to_bgra(frame):
    """Extract NV12 planes from *frame* and return a BGRA numpy array.

    NV12 (VideoToolbox output on macOS) is always limited-range BT.601.

    NV12 UV plane: plane.width = w//2 (UV pair count), but line_size = w
    (bytes per row, U and V interleaved).  We must slice with frame.width,
    not plane.width, to get all interleaved bytes.

    The C path calls nv12_lr_to_bgra directly with the interleaved UV buffer,
    avoiding the strided u_half/v_half views.  The numpy fallback extracts
    u_half/v_half as views (no allocation) and delegates to _bt601_yuv420_to_bgra.
    """
    h, w = frame.height, frame.width
    y_p  = frame.planes[0]
    uv_p = frame.planes[1]
    y  = (np.frombuffer(y_p,  dtype=np.uint8)
          .reshape(y_p.height,  y_p.line_size)[:h, :w])
    uv = (np.frombuffer(uv_p, dtype=np.uint8)
          .reshape(uv_p.height, uv_p.line_size)[:(h + 1) // 2, :w])

    lib = _get_c_lib()
    if lib is not None:
        bgra = _get_yuv_bufs(h, w)['bgra']
        lib.nv12_lr_to_bgra(
            _arr_ptr(y),  y.strides[0],
            _arr_ptr(uv), uv.strides[0],
            _arr_ptr(bgra),
            w, h)
        return bgra

    # numpy fallback: u_half/v_half are strided views — no allocation
    u_half = uv[:, 0::2]
    v_half = uv[:, 1::2]
    return _bt601_yuv420_to_bgra(y, u_half, v_half, False)


def _frame_to_bgra_regions(frame, fmt, regions, h, w):
    """Region-aware YUV→BGRA: convert only pixels in *regions* dirty rects.

    Matches grdp's regionHint / setRegionHint pattern.  Only the pixels
    within each (left, top, right, bottom) rectangle are written into the
    persistent pool BGRA buffer; unchanged areas keep their previous values.

    Fast path: C extension (compiled at startup) — 3-13× faster than numpy
    because it processes all BGRA channels in a single pass with UV terms
    computed once per pixel pair, and Clang auto-vectorises to NEON/AVX.

    Numpy fallback: when the C extension is unavailable, performs per-region
    int32 BT.601 arithmetic using numpy's vectorised element-wise ops.  Still
    much faster than full-frame conversion when only a small fraction of the
    screen is dirty.

    *regions* is a list of (left, top, right, bottom, ...) tuples from
    _parse_avc420_stream.  Any tuple elements beyond index 3 are ignored.
    Returns the pool BGRA array, or None for unsupported formats (caller falls
    back to full-frame conversion).
    """
    bgra = _get_yuv_bufs(h, w)['bgra']

    lib = _get_c_lib()
    if lib is not None:
        # C fast path: single-pass NEON/AVX auto-vectorised conversion.
        rects_flat = np.array(
            [v for r in regions for v in (r[0], r[1], r[2], r[3])],
            dtype=np.uint16)
        n_rects = ctypes.c_int(len(regions))
        rects_ptr = _arr_ptr(rects_flat)

        if fmt in ('yuv420p', 'yuvj420p'):
            full_range = fmt in _FULL_RANGE_FORMATS
            y = _plane_to_array(frame.planes[0])
            u = _plane_to_array(frame.planes[1])
            v = _plane_to_array(frame.planes[2])
            fn = lib.yuv420p_fr_to_bgra_regions if full_range else lib.yuv420p_lr_to_bgra_regions
            fn(_arr_ptr(y), y.strides[0],
               _arr_ptr(u), u.strides[0],
               _arr_ptr(v),
               _arr_ptr(bgra), w, h,
               rects_ptr, n_rects)
            return bgra

        if fmt == 'nv12':
            y_p  = frame.planes[0]
            uv_p = frame.planes[1]
            y  = (np.frombuffer(y_p,  dtype=np.uint8)
                  .reshape(y_p.height,  y_p.line_size)[:h, :w])
            uv = (np.frombuffer(uv_p, dtype=np.uint8)
                  .reshape(uv_p.height, uv_p.line_size)[:(h + 1) // 2, :w])
            lib.nv12_lr_to_bgra_regions(
                _arr_ptr(y),  y.strides[0],
                _arr_ptr(uv), uv.strides[0],
                _arr_ptr(bgra), w, h,
                rects_ptr, n_rects)
            return bgra

        return None  # unknown format — caller uses full-frame path

    # --- numpy fallback (no C compiler) ---
    # Process only the dirty rectangles rather than the entire frame,
    # saving work proportional to the unchanged screen area.
    if fmt not in ('yuv420p', 'yuvj420p', 'nv12'):
        return None

    if fmt in ('yuv420p', 'yuvj420p'):
        full_range = fmt in _FULL_RANGE_FORMATS
        Y = _plane_to_array(frame.planes[0])
        U_half = _plane_to_array(frame.planes[1])
        V_half = _plane_to_array(frame.planes[2])
        _yuv420p_regions_numpy(Y, U_half, V_half, bgra, regions, full_range)
    else:  # nv12
        y_p  = frame.planes[0]
        uv_p = frame.planes[1]
        Y  = (np.frombuffer(y_p,  dtype=np.uint8)
              .reshape(y_p.height,  y_p.line_size)[:h, :w])
        UV = (np.frombuffer(uv_p, dtype=np.uint8)
              .reshape(uv_p.height, uv_p.line_size)[:(h + 1) // 2, :w])
        U_half = UV[:, 0::2]
        V_half = UV[:, 1::2]
        _yuv420p_regions_numpy(Y, U_half, V_half, bgra, regions, False)
    return bgra


def _yuv420p_regions_numpy(Y, U_half, V_half, bgra, regions, full_range):
    """Numpy per-region BT.601 YUV420p→BGRA conversion (C-extension fallback).

    For each dirty rectangle, extracts numpy views of the YUV planes,
    upsamples UV with np.repeat, and applies integer BT.601 arithmetic.
    Much faster than full-frame conversion when dirty area is small, but
    3-13× slower than the C extension (multiple passes, intermediate arrays).

    Uses int32 arithmetic (matching the C integer coefficients) to avoid
    overflow: e.g. (235-16)*298 = 65322 exceeds int16 max (32767).
    """
    if full_range:
        ky, kr, kgu, kgv, kb, yoff = 256, 359, 88, 183, 454, 0
    else:
        ky, kr, kgu, kgv, kb, yoff = 298, 409, 100, 208, 516, 16

    for r in regions:
        left, top, right, bottom = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        left   = max(left,   0)
        top    = max(top,    0)
        right  = min(right,  Y.shape[1])
        bottom = min(bottom, Y.shape[0])
        if left >= right or top >= bottom:
            continue

        rh = bottom - top
        rw = right - left

        # Y region as int32 view (no copy for contiguous rows)
        y_i = Y[top:bottom, left:right].astype(np.int32)

        # UV half-res subregion: upsample 2× with np.repeat
        u_raw = U_half[top // 2:(bottom + 1) // 2, left // 2:(right + 1) // 2]
        v_raw = V_half[top // 2:(bottom + 1) // 2, left // 2:(right + 1) // 2]
        u_i = np.repeat(np.repeat(u_raw, 2, axis=0)[:rh], 2, axis=1)[:, :rw].astype(np.int32) - 128
        v_i = np.repeat(np.repeat(v_raw, 2, axis=0)[:rh], 2, axis=1)[:, :rw].astype(np.int32) - 128

        c = (y_i - yoff) * ky
        bgra[top:bottom, left:right, 0] = np.clip((c + kb * u_i + 128) >> 8, 0, 255)
        bgra[top:bottom, left:right, 1] = np.clip((c - kgu * u_i - kgv * v_i + 128) >> 8, 0, 255)
        bgra[top:bottom, left:right, 2] = np.clip((c + kr * v_i + 128) >> 8, 0, 255)
        bgra[top:bottom, left:right, 3] = 255


def _bt601_yuv444_to_bgra(y, u_full, v_full, full_range):
    """Convert YUV444 (full-resolution U/V planes) to a BGRA numpy array.

    Used for AVC444 LC=2 combine output where U/V have already been
    reconstructed to full resolution by _combine_avc444v2_bgra.

    Fast path: C extension (yuv444_lr/fr_to_bgra).
    Fallback: float32 numpy arithmetic.
    """
    h, w = y.shape
    bgra = _get_yuv_bufs(h, w)['bgra']
    lib = _get_c_lib()
    if lib is not None:
        fn = lib.yuv444_fr_to_bgra if full_range else lib.yuv444_lr_to_bgra
        fn(_arr_ptr(y),      y.strides[0],
           _arr_ptr(u_full), u_full.strides[0],
           _arr_ptr(v_full), v_full.strides[0],
           _arr_ptr(bgra), w, h)
        return bgra

    # numpy fallback (float32 path)
    Y = y.astype(np.float32)
    U = u_full.astype(np.float32) - 128.0
    V = v_full.astype(np.float32) - 128.0
    if full_range:
        R = np.clip(Y + 1.402 * V, 0, 255)
        G = np.clip(Y - 0.344 * U - 0.714 * V, 0, 255)
        B = np.clip(Y + 1.772 * U, 0, 255)
    else:
        Y -= 16.0
        R = np.clip(1.164 * Y + 1.596 * V, 0, 255)
        G = np.clip(1.164 * Y - 0.392 * U - 0.813 * V, 0, 255)
        B = np.clip(1.164 * Y + 2.017 * U, 0, 255)
    bgra[:, :, 0] = B
    bgra[:, :, 1] = G
    bgra[:, :, 2] = R
    bgra[:, :, 3] = 255
    return bgra


def _combine_avc444v2_bgra(y_plane, u_half, v_half, aux_frame, full_range, w, h):
    """Combine stream1 luma with stream2 chroma per MS-RDPEGFX 3.3.8.3.3.

    Reconstructs full-resolution U444/V444 planes from two sources:
      - stream1 cached half-res chroma (u_half, v_half) for B2/B3 positions
      - stream2 auxiliary I420 planes (aux_frame) for B4/B5/B6/B7/B8/B9

    Channel mapping (AVC444v2, codecId=0x000F):
      B4/B5 — aux_y row [col=2k+1]:
        bytes [0, w/2)    = Cb at all odd-x columns
        bytes [w/2, w)    = Cr at all odd-x columns
      B6/B7 — aux_u half-height row (row=2j+1, col=4k):
        bytes [0, w/4)    = Cb, bytes [w/4, w/2) = Cr
      B8/B9 — aux_v half-height row (row=2j+1, col=4k+2):
        bytes [0, w/4)    = Cb, bytes [w/4, w/2) = Cr
      B2/B3 — even col/even row: from stream1 half-res u_half/v_half

    Parameters:
      y_plane:   (h, w)         uint8 — luma from stream1
      u_half:    (h//2, w//2)   uint8 — Cb from stream1 (half-res)
      v_half:    (h//2, w//2)   uint8 — Cr from stream1 (half-res)
      aux_frame: av.VideoFrame  yuv420p — decoded from stream2
      full_range: bool
      w, h: target frame dimensions

    Returns BGRA numpy array (h, w, 4) uint8 from the shared pool.
    Caller must copy before next decode if retaining.
    """
    half_w = w // 2
    even_w = (w + 1) // 2
    quarter_w = w // 4
    half_h = (h + 1) // 2
    odd_h = h // 2

    aux_y = _plane_to_array(aux_frame.planes[0])  # (h, w) logical
    aux_u = _plane_to_array(aux_frame.planes[1])  # (half_h, half_w) logical
    aux_v = _plane_to_array(aux_frame.planes[2])  # (half_h, half_w) logical

    bufs = _get_yuv_bufs(h, w)
    U444 = bufs['U444']
    V444 = bufs['V444']

    # B4/B5: odd columns, all rows — from aux_y
    U444[:h, 1::2] = aux_y[:h, :half_w]
    V444[:h, 1::2] = aux_y[:h, half_w:half_w + half_w]

    # B2/B3: even columns, even rows — from stream1 half-res chroma
    U444[0::2, 0::2] = u_half[:half_h, :even_w]
    V444[0::2, 0::2] = v_half[:half_h, :even_w]

    # B6/B7: col % 4 == 0, odd rows — from aux_u
    U444[1::2, 0::4] = aux_u[:odd_h, :quarter_w]
    V444[1::2, 0::4] = aux_u[:odd_h, quarter_w:quarter_w * 2]

    # B8/B9: col % 4 == 2, odd rows — from aux_v
    U444[1::2, 2::4] = aux_v[:odd_h, :quarter_w]
    V444[1::2, 2::4] = aux_v[:odd_h, quarter_w:quarter_w * 2]

    return _bt601_yuv444_to_bgra(y_plane, U444, V444, full_range)


try:
    import av
    _HAS_AV = True
except ImportError:
    _HAS_AV = False
    log.warning("PyAV (av) not installed; H.264/AVC codec support disabled")


# H.264 Annex B start code for NAL unit framing
_ANNEX_B_START = b'\x00\x00\x00\x01'


def _try_open_hw_decoder():
    """Try to open a hardware-accelerated H.264 decoder.
    Returns (CodecContext, hw_name) or (None, None).

    Hardware decoders (VideoToolbox on macOS, VAAPI on Linux) do not
    perform B-frame reorder buffering, so every packet produces
    immediate output — matching grdp's behaviour.
    """
    hw_configs = []
    if sys.platform == 'darwin':
        hw_configs.append('videotoolbox')
    elif sys.platform.startswith('linux'):
        hw_configs.append('vaapi')

    codec = av.codec.Codec('h264', 'r')
    for hw_name in hw_configs:
        try:
            from av.codec.hwaccel import HWAccel
            accel = HWAccel(hw_name)
            ctx = av.codec.CodecContext.create(codec, hwaccel=accel)
            ctx.open()
            log.debug("AVC: opened hardware decoder: %s" % hw_name)
            return ctx, hw_name
        except Exception as e:
            log.debug("AVC: hardware decoder '%s' unavailable: %s" % (hw_name, e))

    return None, None


def _open_sw_decoder():
    """Open a software H.264 decoder.

    Use FFmpeg's built-in h264 decoder with LOW_DELAY and FAST flags.
    LOW_DELAY minimises frame reordering, and FAST skips some post-
    processing.

    SLICE threading (thread_type=2) parallelises entropy coding and
    residual reconstruction within a single frame — no reorder delay.
    Verified: delay=0 with SLICE mode for any thread_count, so we can
    safely use min(cpu_count, 4) threads for 2–4× decode speedup.

    libopenh264 was tried as an alternative (no reorder buffering at
    all) but it returns AVERROR_UNKNOWN on certain High-profile streams
    that RDP servers commonly send, making it unsuitable.
    """
    codec = av.codec.Codec('h264', 'r')
    ctx = av.codec.CodecContext.create(codec)
    ctx.thread_count = min(os.cpu_count() or 1, 4)
    ctx.thread_type = 2  # SLICE — no reorder delay, parallelises within frame
    ctx.options = {'flags': '+low_delay', 'flags2': '+fast', 'refs': '1'}
    ctx.open()
    log.debug("AVC: using software H.264 decoder (thread_count=%d, SLICE, low_delay, refs=1)" %
              ctx.thread_count)
    return ctx


class AvcDecoder:
    """Decodes H.264/AVC bitstreams from RDPGFX AVC420/AVC444 codec data."""

    def __init__(self):
        if not _HAS_AV:
            raise RuntimeError("PyAV (av) is required for H.264/AVC support")

        self._ctx = None
        self._hw_name = None
        self._decode_count = 0
        self._null_count = 0
        self._last_receive_time = 0.0  # monotonic time of most recent _decode_h264 call
        # SPS/PPS recovery state
        self._sps_nal = b''          # cached SPS NAL (Annex B, with start code)
        self._pps_nal = b''          # cached PPS NAL (Annex B, with start code)
        self._hw_error_count = 0       # consecutive non-IDR generic errors (AVERROR_UNKNOWN etc.)
        self._hw_reset_count = 0       # total hard resets
        self._needs_keyframe = False   # True after reset/error; skips non-IDR frames until IDR
        self._keyframe_wait_count = 0  # non-IDR frames dropped while waiting for IDR
        self._keyframe_wait_start = 0.0  # monotonic time when IDR-wait began
        self._prepend_sps_next_idr = False  # inject SPS+PPS before next IDR
        self.on_hard_reset = None    # callback() invoked after each hard reset
        # AVC444 LC=2 chroma-upgrade support
        self._dec2 = None            # auxiliary decoder for stream2 (LC=2 P-frames)
        self._avc444_y = None        # cached luma (h, w) uint8 from last LC=0/1 decode
        self._avc444_u = None        # cached Cb (h//2, w//2) uint8 from last LC=0/1 decode
        self._avc444_v = None        # cached Cr (h//2, w//2) uint8 from last LC=0/1 decode
        self._avc444_full_range = False
        self._avc444_w = 0
        self._avc444_h = 0
        self._avc444_y_updated_at = 0.0  # monotonic time of last Y-plane cache refresh
        self._init_decoder()

    def _init_decoder(self):
        """Initialize H.264 decoder, trying hardware first."""
        self._decode_count = 0
        self._null_count = 0
        ctx, hw_name = _try_open_hw_decoder()
        if ctx is not None:
            self._ctx = ctx
            self._hw_name = hw_name
        else:
            self._ctx = _open_sw_decoder()
            self._hw_name = None

    @property
    def is_hardware(self):
        return self._hw_name is not None

    @property
    def null_count(self):
        """Number of consecutive frames where the decoder produced no output."""
        return self._null_count

    def close(self):
        if self._ctx is not None:
            self._ctx = None
        if self._dec2 is not None:
            self._dec2.close()
            self._dec2 = None

    def flush(self):
        """Flush decoder buffers while preserving the codec context.

        VideoToolbox (hardware H.264 decoder on macOS) silently returns null
        frames when it is waiting for an IDR to restart the reference chain.
        This is *correct behaviour*, not an error state.  Calling
        flush_buffers() while VideoToolbox is in this wait state has been
        confirmed to push it into a hard AVERROR_UNKNOWN error from which
        only full decoder recreation recovers.

        Therefore we intentionally do NOT call flush_buffers() here.  We
        just reset the null-frame counters so _onAvcNoOutput does not keep
        firing every 1.5 s.  The decoder stays alive with its full codec
        context (SPS, PPS, reference frames) intact, and it will resume
        naturally as soon as the server sends the next IDR — exactly the
        behaviour VideoToolbox expects.

        For software (libavcodec) decoders an IDR also naturally restarts
        the reference chain without needing an explicit flush, so this
        approach is correct for both paths.

        Explicit flush_buffers() is still used in _flush_and_retry_idr(),
        which is called only when an IDR packet fails to decode — a
        fundamentally different situation where the codec state is already
        known to be corrupt.
        """
        self._decode_count = 0
        self._null_count = 0
        if self._dec2 is not None:
            self._dec2.flush()

    def request_keyframe(self):
        """Enter IDR-wait state without flushing the codec context.

        Called by the RDPGFX stall handler (_onAvcNoOutput) after detecting
        that the decoder has been producing no output for too long.

        Sets _needs_keyframe=True so subsequent P-frames are silently dropped
        (mirroring grdp's needsKeyFrame pattern) while we wait for the IDR
        the server will send in response to our ForceRefresh request.

        No flush_buffers() is called: the VideoToolbox codec context
        (SPS/PPS, reference frames) is preserved intact so that when the
        IDR arrives it is decoded naturally.  If VideoToolbox still returns
        null for the IDR, the existing flush+retry path in _decode_h264
        handles it — IDR frames are self-contained, so that flush is safe.

        Without this call the decoder would keep feeding P-frames into
        stalled VideoToolbox (wasted CPU, risk of cascading errors) for
        the entire 15-second refresh cooldown before the next IDR request
        is sent.  Mirrors grdp's post-stall needsKeyFrame=true pattern.
        """
        self._needs_keyframe = True
        self._keyframe_wait_count = 0
        self._keyframe_wait_start = 0.0
        if self._dec2 is not None:
            self._dec2.request_keyframe()

    def _parse_and_cache_nals(self, h264_data):
        """Single-pass NAL unit scanner: return NAL type list and update SPS/PPS cache.

        Uses bytes.find() to jump directly to start codes instead of a
        byte-by-byte scan, which is O(n) in C rather than Python.
        """
        data = bytes(h264_data) if not isinstance(h264_data, bytes) else h264_data
        n = len(data)
        nal_types = []
        search_from = 0
        while search_from < n:
            # Jump directly to the next \x00\x00\x01 pattern (C-speed search)
            p = data.find(b'\x00\x00\x01', search_from)
            if p == -1:
                break
            # Distinguish 4-byte (\x00\x00\x00\x01) from 3-byte start code
            if p > 0 and data[p - 1] == 0:
                sc_start = p - 1
                sc_len = 4
            else:
                sc_start = p
                sc_len = 3
            nal_start = sc_start + sc_len
            if nal_start >= n:
                break
            nal_type = data[nal_start] & 0x1F
            nal_types.append(nal_type)
            if nal_type in (7, 8):
                # Find end of this NAL: next start code (also via find())
                p2 = data.find(b'\x00\x00\x01', nal_start + 1)
                if p2 == -1:
                    nal_end = n
                elif p2 > 0 and data[p2 - 1] == 0:
                    nal_end = p2 - 1  # 4-byte start code: end before leading \x00
                else:
                    nal_end = p2
                nal_bytes = data[sc_start:nal_end]
                if nal_type == 7:
                    if nal_bytes != self._sps_nal:
                        self._sps_nal = nal_bytes
                        log.debug("AVC: cached SPS NAL (%d bytes)" % len(nal_bytes))
                else:
                    if nal_bytes != self._pps_nal:
                        self._pps_nal = nal_bytes
                        log.debug("AVC: cached PPS NAL (%d bytes)" % len(nal_bytes))
                search_from = nal_end
            else:
                search_from = nal_start + 1
        return nal_types

    def _hard_reset(self, reason='hardware error'):
        """Destroy and recreate the decoder.

        Called either after a persistent hardware error (AVERROR_UNKNOWN cascade)
        or when the RDPGFX layer detects a freeze (null output for too long).

        VideoToolbox (macOS hardware decoder) can enter an unrecoverable state
        where flush_buffers() has no effect; the only fix is to destroy and
        recreate the codec context.

        After recreation the decoder has no SPS/PPS context.  If we have
        cached SPS/PPS from the original session we set _prepend_sps_next_idr
        so those headers are prepended to the next IDR, giving the fresh
        decoder the codec context it needs to decode the server's bare IDRs.

        After 3 hard resets we permanently fall back to the software decoder
        to avoid an infinite reset loop if hardware acceleration is broken.
        """
        self._hw_reset_count += 1
        log.debug("AVC: hard reset #%d (%s, recreating decoder)"
                    % (self._hw_reset_count, reason))
        self._ctx = None
        self._hw_error_count = 0
        self._decode_count = 0
        self._null_count = 0

        # Always try hardware decoder; only fall back to software if unavailable.
        self._needs_keyframe = True
        self._keyframe_wait_count = 0
        self._keyframe_wait_start = 0.0
        ctx, hw_name = _try_open_hw_decoder()
        if ctx is not None:
            self._ctx = ctx
            self._hw_name = hw_name
        else:
            self._ctx = _open_sw_decoder()
            self._hw_name = None

        if self._sps_nal and self._pps_nal:
            self._prepend_sps_next_idr = True
            log.debug("AVC: will inject cached SPS+PPS (%d+%d bytes) before next IDR"
                      % (len(self._sps_nal), len(self._pps_nal)))

        # Notify the caller (e.g. RDPGFX layer) so it can immediately request
        # a new IDR from the server.  The fresh decoder has no reference frames,
        # so any arriving P-frames will fail until the server sends an IDR.
        if self.on_hard_reset is not None:
            self.on_hard_reset()

    def soft_reset(self):
        """Public entry point for an externally-triggered decoder reset.

        Equivalent to _hard_reset() but named to mirror grdp's soft-reset
        pattern where the RDPGFX layer resets the decoder context on stall
        detection (rather than waiting for the internal 900-frame/15 s limit).
        """
        self._hard_reset(reason='soft reset (stall)')

    # -----------------------------------------------------------------
    # AVC420 bitmap stream (MS-RDPEGFX 2.2.4.4)
    # -----------------------------------------------------------------

    def decode_avc420_arr(self, data, dest_width, dest_height):
        """Decode AVC420 bitmap stream.

        Returns BGRA numpy array (dest_height, dest_width, 4) or None.
        No tobytes() copy — caller owns the returned array until done.
        """
        regions, h264_data = self._parse_avc420_stream(data)
        if h264_data is None or len(h264_data) == 0:
            return None

        frame = self._decode_h264(h264_data)
        if frame is None:
            return None

        return self._frame_to_bgra(frame, dest_width, dest_height, regions=regions)

    def decode_avc420(self, data, dest_width, dest_height):
        """Decode AVC420 bitmap stream.

        Returns BGRA bytes (dest_width * dest_height * 4) or None.
        """
        arr = self.decode_avc420_arr(data, dest_width, dest_height)
        if arr is None:
            return None
        return arr.tobytes()

    def _parse_avc420_stream(self, data):
        """Parse AVC420 bitmap stream per MS-RDPEGFX 2.2.4.4.

        Returns (regions, h264_data) where regions is a list of
        (left, top, right, bottom, qp, qualityMode) tuples.
        """
        if len(data) < 4:
            return [], None

        numRegionRects = struct.unpack_from('<I', data, 0)[0]
        off = 4

        # Region rects: each 8 bytes (left(2) + top(2) + right(2) + bottom(2))
        rects_size = numRegionRects * 8
        if len(data) < off + rects_size:
            log.warning("AVC420: insufficient data for %d region rects" % numRegionRects)
            return [], None

        # Parse all rects in one struct call (4 uint16 per rect).
        raw = struct.unpack_from('<%dH' % (numRegionRects * 4), data, off)
        qvals_off = off + rects_size
        qvals_size = numRegionRects * 2
        if len(data) < qvals_off + qvals_size:
            log.warning("AVC420: insufficient data for quant quality values")
            return [], None
        regions = [(raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3],
                    data[qvals_off + i*2], data[qvals_off + i*2 + 1])
                   for i in range(numRegionRects)]
        off = qvals_off + qvals_size

        h264_data = data[off:]
        log.debug("AVC420: %d regions, %d bytes H.264 data" %
                  (numRegionRects, len(h264_data)))
        return regions, bytes(h264_data)

    # -----------------------------------------------------------------
    # AVC444 / AVC444v2 bitmap stream (MS-RDPEGFX 2.2.4.5 / 2.2.4.6)
    # -----------------------------------------------------------------

    def decode_avc444_arr(self, data, dest_width, dest_height):
        """Decode AVC444/AVC444v2 bitmap stream.

        LC values:
          0 = both luma (YUV420) and chroma streams present; primes aux decoder
          1 = luma stream only
          2 = chroma-upgrade only; combined with cached luma via aux decoder

        Returns BGRA numpy array (dest_height, dest_width, 4), b"" sentinel
        when LC=2 cannot be decoded (aux decoder not yet primed), or None on
        failure.
        """
        if len(data) < 4:
            return None

        cbField = struct.unpack_from('<I', data, 0)[0]
        lc = (cbField >> 30) & 0x03
        cbAvc420Stream1 = cbField & 0x3FFFFFFF
        rest = data[4:]

        log.debug("AVC444: LC=%d cbStream1=%d restLen=%d" %
                  (lc, cbAvc420Stream1, len(rest)))

        if lc == 2:
            return self._decode_avc444_lc2(rest, dest_width, dest_height)

        # LC=0 or LC=1: decode stream1 and cache YUV planes for future LC=2.
        if lc == 0:
            if cbAvc420Stream1 > len(rest):
                log.warning("AVC444: stream1 size %d exceeds data %d" %
                            (cbAvc420Stream1, len(rest)))
                return None
            stream1_data = rest[:cbAvc420Stream1]
        elif lc == 1:
            stream1_data = rest
            if cbAvc420Stream1 > 0 and cbAvc420Stream1 <= len(rest):
                stream1_data = rest[:cbAvc420Stream1]
        else:
            log.warning("AVC444: unexpected LC value %d" % lc)
            return None

        _regions, h264_data = self._parse_avc420_stream(stream1_data)
        if h264_data is None or len(h264_data) == 0:
            return None

        frame = self._decode_h264(h264_data)
        if frame is None:
            return None

        # Cache YUV planes for future LC=2 chroma-upgrade combine.
        self._cache_stream1_yuv(frame)

        result = self._frame_to_bgra(frame, dest_width, dest_height)

        # LC=0: prime the auxiliary decoder with stream2 so it has the IDR
        # for subsequent standalone LC=2 P-frames.
        if lc == 0 and cbAvc420Stream1 < len(rest):
            stream2_data = rest[cbAvc420Stream1:]
            if stream2_data:
                _r2, h264_data2 = self._parse_avc420_stream(stream2_data)
                if h264_data2 and len(h264_data2) > 0:
                    self._prime_aux_decoder(h264_data2)

        return result

    def decode_avc444(self, data, dest_width, dest_height):
        """Decode AVC444/AVC444v2 bitmap stream.

        LC values:
          0 = both luma (YUV420) and chroma streams present
          1 = luma stream only
          2 = chroma-upgrade; combined with cached luma when aux decoder is ready

        Returns BGRA bytes (dest_width * dest_height * 4), b"" sentinel when
        LC=2 cannot yet be decoded, or None on failure.
        """
        arr = self.decode_avc444_arr(data, dest_width, dest_height)
        if arr is None:
            return None
        if isinstance(arr, bytes):  # b"" sentinel: LC=2 not yet decodable
            return b""
        return arr.tobytes()

    def _cache_stream1_yuv(self, frame):
        """Extract and cache YUV planes from stream1 frame for LC=2 combine.

        Normalises the frame to yuv420p so plane access is uniform regardless
        of whether the decoder is hardware (VideoToolbox/VAAPI) or software.
        Copies the planes to avoid holding references to the frame buffer.
        """
        fmt = frame.format.name
        if fmt not in ('yuv420p', 'yuvj420p'):
            frame = frame.reformat(format='yuv420p')
        self._avc444_full_range = frame.format.name in _FULL_RANGE_FORMATS
        self._avc444_y = _plane_to_array(frame.planes[0]).copy()
        self._avc444_u = _plane_to_array(frame.planes[1]).copy()
        self._avc444_v = _plane_to_array(frame.planes[2]).copy()
        self._avc444_w = frame.width
        self._avc444_h = frame.height
        self._avc444_y_updated_at = time.monotonic()

    def _prime_aux_decoder(self, h264_data):
        """Feed stream2 data to the auxiliary decoder to advance past its IDR.

        The IDR for the auxiliary H.264 sequence is always carried inside the
        LC=0 stream2.  Subsequent LC=2 packets are P-frames in that same
        sequence.  By decoding (and discarding) the IDR here, h264dec2 is
        primed with the codec state needed to decode the later P-frames.
        """
        dec2 = self._get_aux_decoder()
        try:
            dec2._decode_h264(h264_data)
        except Exception as e:
            log.debug("AVC444: aux decoder prime error: %s" % e)
            if dec2._hw_error_count >= dec2._HW_ERROR_THRESHOLD:
                dec2.close()
                self._dec2 = None

    def _get_aux_decoder(self):
        """Lazy-initialize the auxiliary H.264 decoder for stream2."""
        if self._dec2 is None:
            self._dec2 = AvcDecoder()
            # Propagate aux decoder hard-resets to the parent's on_hard_reset callback
            # so that ForceRefresh is sent whenever the chroma decoder times out.
            # Use a lambda so it always picks up the current value of self.on_hard_reset.
            self._dec2.on_hard_reset = lambda: self.on_hard_reset() if self.on_hard_reset else None
            log.debug("AVC444: auxiliary decoder initialized")
        return self._dec2

    # Maximum age of the Y-plane cache before an LC=2 combine is suppressed.
    # Mirrors grdp's avc444YStaleness (500 ms).  When the main decoder is
    # stalling (producing null frames), the Y cache is not refreshed; combining
    # stale luma with fresh chroma produces visible colour artefacts.
    _AVC444_Y_STALENESS = 0.5  # seconds

    def _decode_avc444_lc2(self, data, dest_width, dest_height):
        """Decode AVC444 LC=2 (chroma-upgrade) frame and combine with cached luma.

        Returns a BGRA numpy array on success, or b"" when the auxiliary
        decoder is not yet primed, the luma cache is empty, or the luma cache
        is stale (main decoder likely stalling).
        """
        if self._dec2 is None:
            log.debug("AVC444: LC=2 skipped (aux decoder not yet initialized)")
            return b""
        if self._avc444_y is None:
            log.debug("AVC444: LC=2 skipped (no cached luma)")
            return b""
        # Staleness guard: mirrors grdp decodeAVC444LC2 avc444YStaleness check.
        # If the main decoder hasn't refreshed the Y-plane for > _AVC444_Y_STALENESS
        # seconds it is likely stalling.  Suppress LC=2 output to avoid colour
        # artefacts from combining old luma with fresh chroma, and signal the
        # caller that a keyframe is needed.
        if self._avc444_y_updated_at > 0.0:
            age = time.monotonic() - self._avc444_y_updated_at
            if age > self._AVC444_Y_STALENESS:
                log.debug("AVC444: LC=2 skipped (Y cache stale %.0fms, main decoder likely stalling)"
                          % (age * 1000))
                return b""

        _regions, h264_data = self._parse_avc420_stream(data)
        if h264_data is None or len(h264_data) == 0:
            log.debug("AVC444: LC=2 skipped (empty stream2 H.264 data)")
            return b""

        try:
            aux_frame = self._dec2._decode_h264(h264_data)
        except Exception as e:
            log.debug("AVC444: LC=2 aux decode error: %s" % e)
            return b""

        if aux_frame is None:
            log.debug("AVC444: LC=2 aux decode buffering")
            return b""

        # Normalise to yuv420p for uniform plane access.
        fmt = aux_frame.format.name
        if fmt not in ('yuv420p', 'yuvj420p'):
            aux_frame = aux_frame.reformat(format='yuv420p')

        w, h = self._avc444_w, self._avc444_h
        bgra = _combine_avc444v2_bgra(
            self._avc444_y, self._avc444_u, self._avc444_v,
            aux_frame, self._avc444_full_range, w, h)

        log.debug("AVC444: LC=2 decoded and combined (%dx%d dest=%dx%d)" %
                  (w, h, dest_width, dest_height))

        # Crop/pad to dest size (mirrors _frame_to_bgra behaviour).
        src_h, src_w = bgra.shape[:2]
        if src_w == dest_width and src_h == dest_height:
            return bgra

        copy_w = min(src_w, dest_width)
        copy_h = min(src_h, dest_height)
        out = np.zeros((dest_height, dest_width, 4), dtype=np.uint8)
        out[:copy_h, :copy_w] = bgra[:copy_h, :copy_w]
        return out



    _NAL_NAMES = {1: 'P', 5: 'IDR', 6: 'SEI', 7: 'SPS', 8: 'PPS', 9: 'AUD'}

    def _decode_h264(self, h264_data):
        """Decode H.264 bitstream and return the decoded frame (av.VideoFrame) or None.

        The decoder may buffer frames internally (B-frame reordering etc.),
        so a single send_packet can produce multiple frames.  We drain them
        all and return the *last* (most recent) frame — matching grdp's
        ffmpegDecoder.Decode behaviour.

        When the decoder returns no frame for a packet that contains an IDR
        slice (NAL type 5), we flush the decoder's internal buffers and
        retry.  This handles the case where a new SPS triggers internal
        reinitialization that causes the decoder to buffer the IDR instead
        of outputting it immediately — even with LOW_DELAY set.  Flushing
        discards reference frames, but IDR slices are self-contained so the
        retry always succeeds.

        When avcodec_send_packet() raises InvalidDataError (most P-frame errors),
        flush_buffers() is called and _needs_keyframe is set True.  While
        _needs_keyframe is True, all subsequent non-IDR frames are dropped without
        decode attempts, preventing cascading errors.  IDR requests are sent to the
        server every 2 s via the drdynvc keyframe-wait path.  When an IDR arrives,
        _needs_keyframe is cleared and decoding resumes.

        When avcodec_send_packet() raises a generic error (e.g. VideoToolbox
        AVERROR_UNKNOWN on macOS), the same flush + _needs_keyframe handling
        applies, plus _hw_error_count is incremented.  After _HW_ERROR_THRESHOLD
        consecutive generic errors we call _hard_reset() to destroy and recreate
        the codec context.  Cached SPS/PPS are then injected before the next IDR.
        """
        if self._ctx is None:
            return None

        self._decode_count += 1
        self._last_receive_time = time.monotonic()

        # Single-pass: collect NAL types and update SPS/PPS cache if changed
        nal_types = self._parse_and_cache_nals(h264_data)
        has_idr = 5 in nal_types

        # Log NAL types for early frames, every IDR, and during freeze
        # (null_count > 0 means the decoder is not producing output — logging NAL
        # types helps confirm whether the server is sending IDR frames during recovery)
        if self._decode_count <= 5 or has_idr or self._null_count > 0:
            names = [self._NAL_NAMES.get(t, str(t)) for t in nal_types]
            log.debug("AVC: decode #%d NAL types: %s (h264Len=%d)" %
                     (self._decode_count, ','.join(names), len(h264_data)))

        # While waiting for an IDR (after reset or error), drop non-IDR frames
        # without decode attempts.  This prevents cascading InvalidDataError
        # from P-frames that the decoder cannot handle without a reference frame.
        # Mirrors grdp's needsKeyFrame / keyframeWaitLimit / keyframeWaitTimeout pattern.
        if self._needs_keyframe and not has_idr:
            self._keyframe_wait_count += 1
            if self._keyframe_wait_count == 1:
                self._keyframe_wait_start = time.monotonic()
            if self._keyframe_wait_count == 1 or self._keyframe_wait_count % 30 == 0:
                log.debug("AVC: decode #%d waiting for IDR (skipped %d non-IDR frames)" %
                          (self._decode_count, self._keyframe_wait_count))
            waited_too_long = (self._keyframe_wait_start > 0 and
                               time.monotonic() - self._keyframe_wait_start >= (
                                   self._KEYFRAME_WAIT_TIMEOUT if self.is_hardware
                                   else self._KEYFRAME_WAIT_TIMEOUT_SW))
            if self._keyframe_wait_count >= self._KEYFRAME_WAIT_LIMIT or waited_too_long:
                kf_timeout = (self._KEYFRAME_WAIT_TIMEOUT if self.is_hardware
                              else self._KEYFRAME_WAIT_TIMEOUT_SW)
                reason = ('keyframe wait timeout (%.1fs)' % kf_timeout
                          if waited_too_long
                          else 'keyframe wait limit (%d)' % self._KEYFRAME_WAIT_LIMIT)
                self._hard_reset(reason=reason)
            return None

        # After a hard reset the fresh decoder has no SPS context.  Inject our
        # cached SPS+PPS before the next IDR so it can decode the server's bare IDR.
        decode_data = h264_data
        if has_idr and self._prepend_sps_next_idr and self._sps_nal and self._pps_nal:
            decode_data = self._sps_nal + self._pps_nal + bytes(h264_data)
            log.debug("AVC: decode #%d injecting cached SPS+PPS (%d+%d bytes) before IDR"
                      % (self._decode_count, len(self._sps_nal), len(self._pps_nal)))

        packet = None
        try:
            packet = av.Packet(decode_data)
            result = self._receive_frame(packet)

            if result is None and has_idr:
                # New SPS caused the decoder to buffer instead of output.
                # Flush internal state and retry — IDR is self-contained.
                log.debug("AVC: decode #%d null on IDR, flushing and retrying" %
                         self._decode_count)
                self._ctx.flush_buffers()
                result = self._receive_frame(packet)
                if result is not None:
                    log.debug("AVC: decode #%d retry OK %dx%d" %
                             (self._decode_count, result.width, result.height))

            if result is None:
                self._null_count += 1
                log.debug("AVC: decode #%d returned no frame (null_count=%d, h264Len=%d)" %
                         (self._decode_count, self._null_count, len(h264_data)))
                # NOTE: We deliberately do NOT call flush_buffers() here.
                # On VideoToolbox, flush_buffers() while the decoder is silent
                # pushes it into a hard error state where every subsequent
                # avcodec_send_packet() raises AVERROR_UNKNOWN, triggering
                # _hard_reset().  After hard reset, Windows RDPGFX never sends
                # a new IDR (only P-frames), so the fresh decoder is stuck
                # forever with InvalidDataError.  Better to wait silently for
                # the server's next IDR (which arrives on scene changes).
            else:
                self._null_count = 0
                self._hw_error_count = 0
                self._needs_keyframe = False
                self._keyframe_wait_count = 0
                self._keyframe_wait_start = 0.0
                self._prepend_sps_next_idr = False
                if self._decode_count <= 3:
                    log.debug("AVC: decode #%d OK frame=%dx%d fmt=%s h264Len=%d" %
                             (self._decode_count, result.width, result.height,
                              result.format.name, len(h264_data)))
            return result
        except av.error.InvalidDataError as e:
            log.debug("AVC: H.264 decode error (invalid data): %s" % e)
            if has_idr and packet is not None:
                # The decoder state was corrupted by earlier bad frames.
                # An IDR is self-contained, so flush and retry immediately
                # rather than waiting for a future IDR to succeed on its own.
                return self._flush_and_retry_idr(packet)
            # Non-IDR invalid data: flush decoder buffers and wait for IDR.
            # Attempting to decode subsequent P-frames with a corrupt decoder
            # state would cause a cascade of InvalidDataError; flushing clears
            # the state and _needs_keyframe drops all non-IDR frames until the
            # server sends a fresh IDR.  Mirrors grdp's needsKeyFrame pattern.
            self._ctx.flush_buffers()
            self._needs_keyframe = True
            self._keyframe_wait_count = 0
            self._keyframe_wait_start = 0.0
        except Exception as e:
            log.warning("AVC: H.264 decode error: %s" % e)
            if has_idr and packet is not None:
                # IDR is self-contained; flush and retry may recover.
                return self._flush_and_retry_idr(packet)
            # Non-IDR hardware error: flush and wait for IDR.  Also count
            # consecutive failures; after _HW_ERROR_THRESHOLD AVERROR_UNKNOWN
            # errors the hardware decoder is in an unrecoverable state so we
            # hard-reset it.
            self._ctx.flush_buffers()
            self._needs_keyframe = True
            self._keyframe_wait_count = 0
            self._keyframe_wait_start = 0.0
            self._hw_error_count += 1
            if self._hw_error_count >= self._HW_ERROR_THRESHOLD:
                self._hard_reset()

        return None

    _HW_ERROR_THRESHOLD = 5    # consecutive AVERROR_UNKNOWN errors before hard reset
    _KEYFRAME_WAIT_LIMIT = 900  # non-IDR frames dropped before giving up and hard-resetting
    _KEYFRAME_WAIT_TIMEOUT = 15.0     # seconds in IDR-wait before hard-resetting (HW decoder, mirrors grdp keyframeWaitTimeout)
    _KEYFRAME_WAIT_TIMEOUT_SW = 5.0   # shorter timeout for SW decoder (mirrors grdp keyframeWaitTimeoutSW)

    @property
    def last_receive_time(self):
        """Monotonic time of the most recent _decode_h264() call, or 0.0 if never called.

        Used by the RDPGFX stall handler to distinguish a genuinely idle server
        (no packets arriving) from a HW decoder stall (packets arriving but no
        decoded frames produced).  Mirrors grdp's LastReceiveTime() interface.
        """
        return self._last_receive_time

    @property
    def needs_keyframe(self):
        """True when the decoder is waiting for an IDR frame.

        Set after a hard reset or a decode error.  While True, all non-IDR
        frames are dropped silently.  Callers may poll this to request a
        keyframe from the server more frequently.
        """
        return self._needs_keyframe

    def _flush_and_retry_idr(self, packet):
        """Flush decoder buffers and retry an IDR packet after an error.

        Called when decoding an IDR frame raises an exception, which means
        the decoder state is corrupt.  Since IDR frames are self-contained
        (no reference frames needed), flushing and retrying always recovers
        — without waiting for the next server-sent IDR.

        If the retry also fails (e.g. VideoToolbox in hard error state) we
        count that toward _hw_error_count so a subsequent _hard_reset() is
        triggered if the threshold is reached.
        """
        try:
            log.debug("AVC: decode #%d flushing corrupted decoder and retrying IDR" %
                      self._decode_count)
            self._ctx.flush_buffers()
            result = self._receive_frame(packet)
            if result is not None:
                self._null_count = 0
                self._hw_error_count = 0
                self._needs_keyframe = False
                self._keyframe_wait_count = 0
                self._keyframe_wait_start = 0.0
                self._prepend_sps_next_idr = False
                log.debug("AVC: decode #%d IDR recovery after flush OK %dx%d" %
                          (self._decode_count, result.width, result.height))
            return result
        except Exception as e:
            log.debug("AVC: decode #%d IDR retry also failed: %s" %
                      (self._decode_count, e))
            self._hw_error_count += 1
            if self._hw_error_count >= self._HW_ERROR_THRESHOLD:
                self._hard_reset()
            return None

    def _receive_frame(self, packet):
        """Send *packet* to the decoder and return the last decoded frame, or None."""
        result = None
        for frame in self._ctx.decode(packet):
            result = frame
        return result

    def _frame_to_bgra(self, frame, dest_width, dest_height, regions=None):
        """Convert av.VideoFrame to a BGRA numpy array cropped/padded to dest_width x dest_height.

        Returns numpy array shape (dest_height, dest_width, 4) dtype uint8.

        For yuv420p, yuvj420p and nv12 formats the conversion is performed
        directly in numpy using BT.601 coefficients, bypassing swscale.  On
        ARM64 (Apple Silicon) swscale's non-accelerated fallback ignores
        sws_setColorspaceDetails and produces wrong colours; the manual path
        fixes both correctness and performance.  This mirrors the optimisation
        introduced in grdp 0.7.7 (plugin/rdpgfx/h264_ffmpeg.go).

        VideoToolbox hardware frames (format='videotoolbox') are downloaded to
        NV12 first (a zero-cost pixel-copy, no colour conversion), then the
        NV12 fast path is taken.

        When *regions* is a non-empty list and sizes match, uses region-aware
        C functions (matching grdp's regionHint/setRegionHint pattern) to
        update only dirty rectangle pixels in the persistent pool buffer,
        saving YUV→BGRA conversion work proportional to the unchanged area.
        """
        fmt = frame.format.name

        # Download hardware frame to CPU as NV12 (no colour conversion).
        if fmt == 'videotoolbox':
            frame = frame.reformat(format='nv12')
            fmt = 'nv12'

        if self._decode_count <= 3:
            log.debug("AVC: frame_to_bgra #%d src=%dx%d dest=%dx%d fmt=%s" %
                     (self._decode_count, frame.width, frame.height,
                      dest_width, dest_height, fmt))

        src_h, src_w = frame.height, frame.width

        # Region-aware path: only convert dirty rects into the persistent pool
        # buffer (matching grdp's regionHint pattern).  Unchanged pixels keep
        # their values from the previous frame, saving conversion work.
        # Only valid when sizes match exactly and regions are available.
        if (regions and len(regions) > 0 and
                src_w == dest_width and src_h == dest_height and
                fmt in ('yuv420p', 'yuvj420p', 'nv12')):
            bgra = _frame_to_bgra_regions(frame, fmt, regions, src_h, src_w)
            if bgra is not None:
                if self._decode_count <= 1:
                    self._save_debug_frame(bgra, dest_width, dest_height)
                return bgra

        # Full-frame path: direct BT.601 YUV→BGRA without swscale.
        if fmt in ('yuv420p', 'yuvj420p'):
            bgra = _frame_yuv420p_to_bgra(frame, fmt in _FULL_RANGE_FORMATS)
        elif fmt == 'nv12':
            bgra = _frame_nv12_to_bgra(frame)
        else:
            # Fallback for any other format (e.g. yuv444p from some codecs).
            bgra = frame.reformat(format='bgra').to_ndarray()

        if src_w == dest_width and src_h == dest_height:
            if self._decode_count <= 1:
                self._save_debug_frame(bgra, dest_width, dest_height)
            return bgra

        copy_w = min(src_w, dest_width)
        copy_h = min(src_h, dest_height)
        out = np.zeros((dest_height, dest_width, 4), dtype=np.uint8)
        out[:copy_h, :copy_w] = bgra[:copy_h, :copy_w]
        if self._decode_count <= 1:
            self._save_debug_frame(out, dest_width, dest_height)
        return out

    def _save_debug_frame(self, bgra_array, width, height):
        """Save a BGRA numpy array as PNG for diagnostic inspection."""
        try:
            from PIL import Image
            # BGRA → RGBA for PIL
            rgba = bgra_array.copy()
            rgba[:, :, 0], rgba[:, :, 2] = bgra_array[:, :, 2].copy(), bgra_array[:, :, 0].copy()
            img = Image.fromarray(rgba[:height, :width], 'RGBA')
            path = '/tmp/rdpyqt_frame_%d.png' % self._decode_count
            img.save(path)
            log.debug("AVC: saved diagnostic frame to %s" % path)
        except Exception as e:
            log.debug("AVC: could not save diagnostic frame: %s" % e)


def is_available():
    """Check if H.264/AVC decoding is available."""
    return _HAS_AV
