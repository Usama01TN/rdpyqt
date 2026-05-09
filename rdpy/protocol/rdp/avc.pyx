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
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
H.264/AVC decoder for RDPGFX AVC420/AVC444/AVC444v2 codecs (Cython edition).

This is a compiled replacement for the pure-Python avc.py fallback.
When built, Python prefers the .so over .py automatically.

Key optimisation over avc.py:
  avc.py compiles a C shared library at runtime via subprocess + ctypes,
  incurring startup latency and per-call ctypes dispatch overhead.
  This file bakes the same C arithmetic directly into Cython cdef functions
  (typed memoryviews, noexcept nogil) so the YUV→BGRA conversion compiles
  once at install time and runs with zero Python or ctypes overhead.

Wire formats per MS-RDPEGFX:
  AVC420 (2.2.4.4): numRegionRects(4) + regionRects(8*N) +
      quantQualityVals(2*N) + avc420EncodedBitstream(remaining)
  AVC444 (2.2.4.5): cbAvc420EncodedBitstream1(4, LC in top 2 bits) +
      avc420EncodedBitstream1 + avc420EncodedBitstream2(optional)
  AVC444v2 (2.2.4.6): same structure as AVC444, different chroma handling
"""

import os
import struct
import sys
import time
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t

np.import_array()

import rdpy.core.log as log

# Pixel formats that carry full-range (0-255) luma/chroma (YUVJ* family).
_FULL_RANGE_FORMATS = frozenset({'yuvj420p', 'yuvj422p', 'yuvj444p', 'yuvj440p'})

# ---------------------------------------------------------------------------
# Cython C-level YUV → BGRA conversion.
# Replaces the runtime-compiled _C_SRC / ctypes path in avc.py.
# All functions: noexcept nogil → pure C, no Python overhead, no bounds checks.
# Typed memoryviews use each array's strides, so padded planes (line_size > w)
# are handled correctly without any extra work by the caller.
# ---------------------------------------------------------------------------

cdef inline uint8_t clamp8(int v) noexcept nogil:
    if v < 0:
        return 0
    if v > 255:
        return 255
    return <uint8_t>v


# ---- full-frame: YUV420p limited-range BT.601 → BGRA ----------------------

cdef void _cy_yuv420p_lr_to_bgra(
    const uint8_t[:, :] Y,   # (h, >=w), strides honour line padding
    const uint8_t[:, :] U,   # (h//2, >=w//2)
    const uint8_t[:, :] V,   # (h//2, >=w//2)
    uint8_t[:, :, :] bgra,   # (h, w, 4)  output pool buffer
    int width, int height
) noexcept nogil:
    cdef int row, col, d, e, Rb, Gb, Bb, c0, c1
    for row in range(height):
        col = 0
        while col + 1 < width:
            d = <int>U[row >> 1, col >> 1] - 128
            e = <int>V[row >> 1, col >> 1] - 128
            Rb = 409 * e + 128
            Gb = -100 * d - 208 * e + 128
            Bb = 516 * d + 128
            c0 = (<int>Y[row, col    ] - 16) * 298
            c1 = (<int>Y[row, col + 1] - 16) * 298
            bgra[row, col,     0] = clamp8((c0 + Bb) >> 8)
            bgra[row, col,     1] = clamp8((c0 + Gb) >> 8)
            bgra[row, col,     2] = clamp8((c0 + Rb) >> 8)
            bgra[row, col,     3] = 255
            bgra[row, col + 1, 0] = clamp8((c1 + Bb) >> 8)
            bgra[row, col + 1, 1] = clamp8((c1 + Gb) >> 8)
            bgra[row, col + 1, 2] = clamp8((c1 + Rb) >> 8)
            bgra[row, col + 1, 3] = 255
            col += 2
        if col < width:
            d = <int>U[row >> 1, col >> 1] - 128
            e = <int>V[row >> 1, col >> 1] - 128
            c0 = (<int>Y[row, col] - 16) * 298
            bgra[row, col, 0] = clamp8((c0 + 516 * d + 128) >> 8)
            bgra[row, col, 1] = clamp8((c0 - 100 * d - 208 * e + 128) >> 8)
            bgra[row, col, 2] = clamp8((c0 + 409 * e + 128) >> 8)
            bgra[row, col, 3] = 255


# ---- full-frame: YUV420p full-range BT.601 → BGRA -------------------------

cdef void _cy_yuv420p_fr_to_bgra(
    const uint8_t[:, :] Y,
    const uint8_t[:, :] U,
    const uint8_t[:, :] V,
    uint8_t[:, :, :] bgra,
    int width, int height
) noexcept nogil:
    cdef int row, col, d, e, Rb, Gb, Bb, c0, c1
    for row in range(height):
        col = 0
        while col + 1 < width:
            d = <int>U[row >> 1, col >> 1] - 128
            e = <int>V[row >> 1, col >> 1] - 128
            Rb = 359 * e + 128
            Gb = -88 * d - 183 * e + 128
            Bb = 454 * d + 128
            c0 = <int>Y[row, col    ] << 8
            c1 = <int>Y[row, col + 1] << 8
            bgra[row, col,     0] = clamp8((c0 + Bb) >> 8)
            bgra[row, col,     1] = clamp8((c0 + Gb) >> 8)
            bgra[row, col,     2] = clamp8((c0 + Rb) >> 8)
            bgra[row, col,     3] = 255
            bgra[row, col + 1, 0] = clamp8((c1 + Bb) >> 8)
            bgra[row, col + 1, 1] = clamp8((c1 + Gb) >> 8)
            bgra[row, col + 1, 2] = clamp8((c1 + Rb) >> 8)
            bgra[row, col + 1, 3] = 255
            col += 2
        if col < width:
            d = <int>U[row >> 1, col >> 1] - 128
            e = <int>V[row >> 1, col >> 1] - 128
            c0 = <int>Y[row, col] << 8
            bgra[row, col, 0] = clamp8((c0 + 454 * d + 128) >> 8)
            bgra[row, col, 1] = clamp8((c0 -  88 * d - 183 * e + 128) >> 8)
            bgra[row, col, 2] = clamp8((c0 + 359 * e + 128) >> 8)
            bgra[row, col, 3] = 255


# ---- full-frame: NV12 limited-range BT.601 → BGRA -------------------------
# UV interleaved: for luma column col, U = UV[row//2, col], V = UV[row//2, col+1]

cdef void _cy_nv12_lr_to_bgra(
    const uint8_t[:, :] Y,    # (h, >=w)
    const uint8_t[:, :] UV,   # (h//2, >=w)  interleaved UV
    uint8_t[:, :, :] bgra,
    int width, int height
) noexcept nogil:
    cdef int row, col, d, e, Rb, Gb, Bb, c0, c1
    for row in range(height):
        col = 0
        while col + 1 < width:
            d = <int>UV[row >> 1, col    ] - 128   # U
            e = <int>UV[row >> 1, col + 1] - 128   # V
            Rb = 409 * e + 128
            Gb = -100 * d - 208 * e + 128
            Bb = 516 * d + 128
            c0 = (<int>Y[row, col    ] - 16) * 298
            c1 = (<int>Y[row, col + 1] - 16) * 298
            bgra[row, col,     0] = clamp8((c0 + Bb) >> 8)
            bgra[row, col,     1] = clamp8((c0 + Gb) >> 8)
            bgra[row, col,     2] = clamp8((c0 + Rb) >> 8)
            bgra[row, col,     3] = 255
            bgra[row, col + 1, 0] = clamp8((c1 + Bb) >> 8)
            bgra[row, col + 1, 1] = clamp8((c1 + Gb) >> 8)
            bgra[row, col + 1, 2] = clamp8((c1 + Rb) >> 8)
            bgra[row, col + 1, 3] = 255
            col += 2
        if col < width:
            d = <int>UV[row >> 1, col    ] - 128
            e = <int>UV[row >> 1, col + 1] - 128
            c0 = (<int>Y[row, col] - 16) * 298
            bgra[row, col, 0] = clamp8((c0 + 516 * d + 128) >> 8)
            bgra[row, col, 1] = clamp8((c0 - 100 * d - 208 * e + 128) >> 8)
            bgra[row, col, 2] = clamp8((c0 + 409 * e + 128) >> 8)
            bgra[row, col, 3] = 255


# ---- region-aware: YUV420p limited-range BT.601 → BGRA -------------------
# rects: flat [left, top, right, bottom, ...] uint16 array, n_rects entries.

cdef void _cy_yuv420p_lr_to_bgra_regions(
    const uint8_t[:, :] Y,
    const uint8_t[:, :] U,
    const uint8_t[:, :] V,
    uint8_t[:, :, :] bgra,
    int width, int height,
    const uint16_t[:] rects,
    int n_rects
) noexcept nogil:
    cdef int i, row, col, d, e, Rb, Gb, Bb, c0, c1
    cdef int left, top, right, bottom
    for i in range(n_rects):
        left   = <int>rects[i * 4    ]
        top    = <int>rects[i * 4 + 1]
        right  = <int>rects[i * 4 + 2]
        bottom = <int>rects[i * 4 + 3]
        if left   < 0:      left   = 0
        if top    < 0:      top    = 0
        if right  > width:  right  = width
        if bottom > height: bottom = height
        if left >= right or top >= bottom:
            continue
        for row in range(top, bottom):
            col = left
            # Align to even column for the 2-pixel unrolled loop.
            if col & 1:
                d = <int>U[row >> 1, col >> 1] - 128
                e = <int>V[row >> 1, col >> 1] - 128
                c0 = (<int>Y[row, col] - 16) * 298
                bgra[row, col, 0] = clamp8((c0 + 516 * d + 128) >> 8)
                bgra[row, col, 1] = clamp8((c0 - 100 * d - 208 * e + 128) >> 8)
                bgra[row, col, 2] = clamp8((c0 + 409 * e + 128) >> 8)
                bgra[row, col, 3] = 255
                col += 1
            while col + 1 < right:
                d = <int>U[row >> 1, col >> 1] - 128
                e = <int>V[row >> 1, col >> 1] - 128
                Rb = 409 * e + 128
                Gb = -100 * d - 208 * e + 128
                Bb = 516 * d + 128
                c0 = (<int>Y[row, col    ] - 16) * 298
                c1 = (<int>Y[row, col + 1] - 16) * 298
                bgra[row, col,     0] = clamp8((c0 + Bb) >> 8)
                bgra[row, col,     1] = clamp8((c0 + Gb) >> 8)
                bgra[row, col,     2] = clamp8((c0 + Rb) >> 8)
                bgra[row, col,     3] = 255
                bgra[row, col + 1, 0] = clamp8((c1 + Bb) >> 8)
                bgra[row, col + 1, 1] = clamp8((c1 + Gb) >> 8)
                bgra[row, col + 1, 2] = clamp8((c1 + Rb) >> 8)
                bgra[row, col + 1, 3] = 255
                col += 2
            if col < right:
                d = <int>U[row >> 1, col >> 1] - 128
                e = <int>V[row >> 1, col >> 1] - 128
                c0 = (<int>Y[row, col] - 16) * 298
                bgra[row, col, 0] = clamp8((c0 + 516 * d + 128) >> 8)
                bgra[row, col, 1] = clamp8((c0 - 100 * d - 208 * e + 128) >> 8)
                bgra[row, col, 2] = clamp8((c0 + 409 * e + 128) >> 8)
                bgra[row, col, 3] = 255


# ---- region-aware: YUV420p full-range BT.601 → BGRA ----------------------

cdef void _cy_yuv420p_fr_to_bgra_regions(
    const uint8_t[:, :] Y,
    const uint8_t[:, :] U,
    const uint8_t[:, :] V,
    uint8_t[:, :, :] bgra,
    int width, int height,
    const uint16_t[:] rects,
    int n_rects
) noexcept nogil:
    cdef int i, row, col, d, e, Rb, Gb, Bb, c0, c1
    cdef int left, top, right, bottom
    for i in range(n_rects):
        left   = <int>rects[i * 4    ]
        top    = <int>rects[i * 4 + 1]
        right  = <int>rects[i * 4 + 2]
        bottom = <int>rects[i * 4 + 3]
        if left   < 0:      left   = 0
        if top    < 0:      top    = 0
        if right  > width:  right  = width
        if bottom > height: bottom = height
        if left >= right or top >= bottom:
            continue
        for row in range(top, bottom):
            col = left
            if col & 1:
                d = <int>U[row >> 1, col >> 1] - 128
                e = <int>V[row >> 1, col >> 1] - 128
                c0 = <int>Y[row, col] << 8
                bgra[row, col, 0] = clamp8((c0 + 454 * d + 128) >> 8)
                bgra[row, col, 1] = clamp8((c0 -  88 * d - 183 * e + 128) >> 8)
                bgra[row, col, 2] = clamp8((c0 + 359 * e + 128) >> 8)
                bgra[row, col, 3] = 255
                col += 1
            while col + 1 < right:
                d = <int>U[row >> 1, col >> 1] - 128
                e = <int>V[row >> 1, col >> 1] - 128
                Rb = 359 * e + 128
                Gb = -88 * d - 183 * e + 128
                Bb = 454 * d + 128
                c0 = <int>Y[row, col    ] << 8
                c1 = <int>Y[row, col + 1] << 8
                bgra[row, col,     0] = clamp8((c0 + Bb) >> 8)
                bgra[row, col,     1] = clamp8((c0 + Gb) >> 8)
                bgra[row, col,     2] = clamp8((c0 + Rb) >> 8)
                bgra[row, col,     3] = 255
                bgra[row, col + 1, 0] = clamp8((c1 + Bb) >> 8)
                bgra[row, col + 1, 1] = clamp8((c1 + Gb) >> 8)
                bgra[row, col + 1, 2] = clamp8((c1 + Rb) >> 8)
                bgra[row, col + 1, 3] = 255
                col += 2
            if col < right:
                d = <int>U[row >> 1, col >> 1] - 128
                e = <int>V[row >> 1, col >> 1] - 128
                c0 = <int>Y[row, col] << 8
                bgra[row, col, 0] = clamp8((c0 + 454 * d + 128) >> 8)
                bgra[row, col, 1] = clamp8((c0 -  88 * d - 183 * e + 128) >> 8)
                bgra[row, col, 2] = clamp8((c0 + 359 * e + 128) >> 8)
                bgra[row, col, 3] = 255


# ---- region-aware: NV12 limited-range BT.601 → BGRA ----------------------
# UV interleaved: for luma col, U = UV[row//2, col & ~1], V = UV[row//2, (col & ~1)+1]

cdef void _cy_nv12_lr_to_bgra_regions(
    const uint8_t[:, :] Y,
    const uint8_t[:, :] UV,   # (h//2, >=w)
    uint8_t[:, :, :] bgra,
    int width, int height,
    const uint16_t[:] rects,
    int n_rects
) noexcept nogil:
    cdef int i, row, col, d, e, Rb, Gb, Bb, c0, c1, even_col
    cdef int left, top, right, bottom
    for i in range(n_rects):
        left   = <int>rects[i * 4    ]
        top    = <int>rects[i * 4 + 1]
        right  = <int>rects[i * 4 + 2]
        bottom = <int>rects[i * 4 + 3]
        if left   < 0:      left   = 0
        if top    < 0:      top    = 0
        if right  > width:  right  = width
        if bottom > height: bottom = height
        if left >= right or top >= bottom:
            continue
        for row in range(top, bottom):
            col = left
            # Align to even UV pair boundary.
            if col & 1:
                even_col = col & ~1
                d = <int>UV[row >> 1, even_col    ] - 128
                e = <int>UV[row >> 1, even_col + 1] - 128
                c0 = (<int>Y[row, col] - 16) * 298
                bgra[row, col, 0] = clamp8((c0 + 516 * d + 128) >> 8)
                bgra[row, col, 1] = clamp8((c0 - 100 * d - 208 * e + 128) >> 8)
                bgra[row, col, 2] = clamp8((c0 + 409 * e + 128) >> 8)
                bgra[row, col, 3] = 255
                col += 1
            while col + 1 < right:
                d = <int>UV[row >> 1, col    ] - 128
                e = <int>UV[row >> 1, col + 1] - 128
                Rb = 409 * e + 128
                Gb = -100 * d - 208 * e + 128
                Bb = 516 * d + 128
                c0 = (<int>Y[row, col    ] - 16) * 298
                c1 = (<int>Y[row, col + 1] - 16) * 298
                bgra[row, col,     0] = clamp8((c0 + Bb) >> 8)
                bgra[row, col,     1] = clamp8((c0 + Gb) >> 8)
                bgra[row, col,     2] = clamp8((c0 + Rb) >> 8)
                bgra[row, col,     3] = 255
                bgra[row, col + 1, 0] = clamp8((c1 + Bb) >> 8)
                bgra[row, col + 1, 1] = clamp8((c1 + Gb) >> 8)
                bgra[row, col + 1, 2] = clamp8((c1 + Rb) >> 8)
                bgra[row, col + 1, 3] = 255
                col += 2
            if col < right:
                d = <int>UV[row >> 1, col    ] - 128
                e = <int>UV[row >> 1, col + 1] - 128
                c0 = (<int>Y[row, col] - 16) * 298
                bgra[row, col, 0] = clamp8((c0 + 516 * d + 128) >> 8)
                bgra[row, col, 1] = clamp8((c0 - 100 * d - 208 * e + 128) >> 8)
                bgra[row, col, 2] = clamp8((c0 + 409 * e + 128) >> 8)
                bgra[row, col, 3] = 255


# ---- YUV444 limited-range BT.601 → BGRA -----------------------------------
# Full-resolution U/V planes (same shape as Y).  Used for AVC444 LC=2.

cdef void _cy_yuv444_lr_to_bgra(
    const uint8_t[:, :] Y,
    const uint8_t[:, :] U,
    const uint8_t[:, :] V,
    uint8_t[:, :, :] bgra,
    int width, int height
) noexcept nogil:
    cdef int row, col, c, d, e
    for row in range(height):
        for col in range(width):
            c = (<int>Y[row, col] - 16) * 298
            d = <int>U[row, col] - 128
            e = <int>V[row, col] - 128
            bgra[row, col, 0] = clamp8((c + 516 * d + 128) >> 8)
            bgra[row, col, 1] = clamp8((c - 100 * d - 208 * e + 128) >> 8)
            bgra[row, col, 2] = clamp8((c + 409 * e + 128) >> 8)
            bgra[row, col, 3] = 255


# ---- YUV444 full-range BT.601 → BGRA --------------------------------------

cdef void _cy_yuv444_fr_to_bgra(
    const uint8_t[:, :] Y,
    const uint8_t[:, :] U,
    const uint8_t[:, :] V,
    uint8_t[:, :, :] bgra,
    int width, int height
) noexcept nogil:
    cdef int row, col, c, d, e
    for row in range(height):
        for col in range(width):
            c = <int>Y[row, col] << 8
            d = <int>U[row, col] - 128
            e = <int>V[row, col] - 128
            bgra[row, col, 0] = clamp8((c + 454 * d + 128) >> 8)
            bgra[row, col, 1] = clamp8((c -  88 * d - 183 * e + 128) >> 8)
            bgra[row, col, 2] = clamp8((c + 359 * e + 128) >> 8)
            bgra[row, col, 3] = 255


# ---------------------------------------------------------------------------
# Compatibility stubs (avc.py uses _get_c_lib() internally; callers outside
# this module should not need it, but provide a safe no-op stub).
# ---------------------------------------------------------------------------

def _get_c_lib():
    """Cython build: always returns None (C extension not needed)."""
    return None


# ---------------------------------------------------------------------------
# Per-resolution pre-allocated working buffers.
# Keys: (h, w).  Values: {'yuv': (3,N) f32, 'rgb': (3,N) f32, 'bgra': (h,w,4) u8}
# Eliminates per-frame heap allocation for the largest intermediate arrays.
# ---------------------------------------------------------------------------
_yuv_buf_pool: dict = {}


def _get_yuv_bufs(int h, int w):
    key = (h, w)
    if key not in _yuv_buf_pool:
        N = h * w
        _yuv_buf_pool[key] = {
            'yuv':  np.empty((3, N), dtype=np.float32),
            'rgb':  np.empty((3, N), dtype=np.float32),
            'bgra': np.zeros((h, w, 4), dtype=np.uint8),
            'U444': np.empty((h, w), dtype=np.uint8),
            'V444': np.empty((h, w), dtype=np.uint8),
        }
    return _yuv_buf_pool[key]


def _plane_to_array(plane):
    """Read a VideoPlane into a 2D uint8 numpy array, stripping line padding.

    Uses the buffer protocol directly (np.frombuffer(plane, ...)) to avoid
    the extra allocation that bytes(plane) would cause.
    """
    return (np.frombuffer(plane, dtype=np.uint8)
            .reshape(plane.height, plane.line_size)[:, :plane.width])


# ---------------------------------------------------------------------------
# Python-level YUV → BGRA wrappers.
# These replace the C-lib-dispatching functions in avc.py.  They call the
# Cython cdef functions above directly — no ctypes overhead, no runtime
# compilation, no numpy intermediate arrays for the main conversion.
# ---------------------------------------------------------------------------

def _bt601_yuv420_to_bgra(y, u_half, v_half, full_range):
    """Convert YUV420 planes to a pre-allocated (H, W, 4) BGRA uint8 array.

    Calls the Cython cdef conversion functions (always available in the
    compiled extension).  The returned array is the pool's bgra buffer —
    callers must copy (e.g. via tobytes()) before the next decode call.
    """
    cdef int h = y.shape[0]
    cdef int w = y.shape[1]
    bgra = _get_yuv_bufs(h, w)['bgra']
    if full_range:
        _cy_yuv420p_fr_to_bgra(y, u_half, v_half, bgra, w, h)
    else:
        _cy_yuv420p_lr_to_bgra(y, u_half, v_half, bgra, w, h)
    return bgra


def _frame_yuv420p_to_bgra(frame, full_range):
    """Extract YUV420P planes from *frame* and return a BGRA numpy array."""
    y = _plane_to_array(frame.planes[0])
    u = _plane_to_array(frame.planes[1])
    v = _plane_to_array(frame.planes[2])
    return _bt601_yuv420_to_bgra(y, u, v, full_range)


def _frame_nv12_to_bgra(frame):
    """Extract NV12 planes from *frame* and return a BGRA numpy array.

    NV12 (VideoToolbox output on macOS) is always limited-range BT.601.
    UV plane: line_size = w (U and V interleaved per row).
    """
    cdef int h = frame.height
    cdef int w = frame.width
    y_p  = frame.planes[0]
    uv_p = frame.planes[1]
    y  = (np.frombuffer(y_p,  dtype=np.uint8)
          .reshape(y_p.height,  y_p.line_size)[:h, :w])
    uv = (np.frombuffer(uv_p, dtype=np.uint8)
          .reshape(uv_p.height, uv_p.line_size)[:(h + 1) // 2, :w])
    bgra = _get_yuv_bufs(h, w)['bgra']
    _cy_nv12_lr_to_bgra(y, uv, bgra, w, h)
    return bgra


def _frame_to_bgra_regions(frame, fmt, regions, int h, int w):
    """Region-aware YUV→BGRA: convert only pixels in *regions* dirty rects.

    Uses the Cython cdef region-aware functions (always available in the
    compiled extension).  Unchanged pixels in the pool buffer persist from
    the previous frame, so only dirty rectangles are recomputed.

    Returns the pool BGRA array, or None for unsupported formats (caller
    falls back to full-frame conversion).
    """
    bgra = _get_yuv_bufs(h, w)['bgra']
    rects_flat = np.array(
        [v for r in regions for v in (r[0], r[1], r[2], r[3])],
        dtype=np.uint16)
    cdef int n_rects = len(regions)

    if fmt in ('yuv420p', 'yuvj420p'):
        y = _plane_to_array(frame.planes[0])
        u = _plane_to_array(frame.planes[1])
        v = _plane_to_array(frame.planes[2])
        if fmt in _FULL_RANGE_FORMATS:
            _cy_yuv420p_fr_to_bgra_regions(y, u, v, bgra, w, h, rects_flat, n_rects)
        else:
            _cy_yuv420p_lr_to_bgra_regions(y, u, v, bgra, w, h, rects_flat, n_rects)
        return bgra

    if fmt == 'nv12':
        y_p  = frame.planes[0]
        uv_p = frame.planes[1]
        y  = (np.frombuffer(y_p,  dtype=np.uint8)
              .reshape(y_p.height,  y_p.line_size)[:h, :w])
        uv = (np.frombuffer(uv_p, dtype=np.uint8)
              .reshape(uv_p.height, uv_p.line_size)[:(h + 1) // 2, :w])
        _cy_nv12_lr_to_bgra_regions(y, uv, bgra, w, h, rects_flat, n_rects)
        return bgra

    return None  # unknown format — caller uses full-frame path


def _yuv420p_regions_numpy(Y, U_half, V_half, bgra, regions, full_range):
    """Per-region YUV420p→BGRA (Cython-accelerated; calls cdef functions).

    In the compiled extension this is equivalent in speed to
    _frame_to_bgra_regions — it is provided for API compatibility with
    code that calls it directly.
    """
    rects_flat = np.array(
        [v for r in regions for v in (r[0], r[1], r[2], r[3])],
        dtype=np.uint16)
    cdef int n_rects = len(regions)
    cdef int h = Y.shape[0]
    cdef int w = Y.shape[1]
    if full_range:
        _cy_yuv420p_fr_to_bgra_regions(Y, U_half, V_half, bgra, w, h, rects_flat, n_rects)
    else:
        _cy_yuv420p_lr_to_bgra_regions(Y, U_half, V_half, bgra, w, h, rects_flat, n_rects)


def _bt601_yuv444_to_bgra(y, u_full, v_full, full_range):
    """Convert YUV444 (full-resolution U/V planes) to a BGRA numpy array.

    Used for AVC444 LC=2 combine output where U/V have already been
    reconstructed to full resolution by _combine_avc444v2_bgra.
    """
    cdef int h = y.shape[0]
    cdef int w = y.shape[1]
    bgra = _get_yuv_bufs(h, w)['bgra']
    if full_range:
        _cy_yuv444_fr_to_bgra(y, u_full, v_full, bgra, w, h)
    else:
        _cy_yuv444_lr_to_bgra(y, u_full, v_full, bgra, w, h)
    return bgra


def _combine_avc444v2_bgra(y_plane, u_half, v_half, aux_frame, full_range, int w, int h):
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
    cdef int half_w = w // 2
    cdef int quarter_w = w // 4
    cdef int half_h = (h + 1) // 2

    aux_y = _plane_to_array(aux_frame.planes[0])
    aux_u = _plane_to_array(aux_frame.planes[1])
    aux_v = _plane_to_array(aux_frame.planes[2])

    bufs = _get_yuv_bufs(h, w)
    U444 = bufs['U444']
    V444 = bufs['V444']

    # B4/B5: odd columns, all rows — from aux_y
    U444[:h, 1::2] = aux_y[:h, :half_w]
    V444[:h, 1::2] = aux_y[:h, half_w:half_w + (w - half_w)]

    # B2/B3: even columns, even rows — from stream1 half-res chroma
    U444[0::2, 0::2] = u_half[:half_h, :half_w]
    V444[0::2, 0::2] = v_half[:half_h, :half_w]

    # B6/B7: col % 4 == 0, odd rows — from aux_u
    U444[1::2, 0::4] = aux_u[:half_h, :quarter_w]
    V444[1::2, 0::4] = aux_u[:half_h, quarter_w:quarter_w * 2]

    # B8/B9: col % 4 == 2, odd rows — from aux_v
    U444[1::2, 2::4] = aux_v[:half_h, :quarter_w]
    V444[1::2, 2::4] = aux_v[:half_h, quarter_w:quarter_w * 2]

    return _bt601_yuv444_to_bgra(y_plane, U444, V444, full_range)


# ---------------------------------------------------------------------------
# PyAV integration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# AvcDecoder class
# ---------------------------------------------------------------------------

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
        self._sps_nal = b''
        self._pps_nal = b''
        self._hw_error_count = 0       # consecutive non-IDR generic errors
        self._hw_reset_count = 0       # total hard resets
        self._needs_keyframe = False   # True after reset/error; skips non-IDR frames
        self._keyframe_wait_count = 0
        self._keyframe_wait_start = 0.0
        self._prepend_sps_next_idr = False
        self.on_hard_reset = None
        # AVC444 LC=2 chroma-upgrade support
        self._dec2 = None
        self._avc444_y = None
        self._avc444_u = None
        self._avc444_v = None
        self._avc444_full_range = False
        self._avc444_w = 0
        self._avc444_h = 0
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
            p = data.find(b'\x00\x00\x01', search_from)
            if p == -1:
                break
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
                p2 = data.find(b'\x00\x00\x01', nal_start + 1)
                if p2 == -1:
                    nal_end = n
                elif p2 > 0 and data[p2 - 1] == 0:
                    nal_end = p2 - 1
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

        if self.on_hard_reset is not None:
            self.on_hard_reset()

    def soft_reset(self):
        """Public entry point for an externally-triggered decoder reset."""
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

        rects_size = numRegionRects * 8
        if len(data) < off + rects_size:
            log.warning("AVC420: insufficient data for %d region rects" % numRegionRects)
            return [], None

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

        self._cache_stream1_yuv(frame)

        result = self._frame_to_bgra(frame, dest_width, dest_height)

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
        if isinstance(arr, bytes):
            return b""
        return arr.tobytes()

    def _cache_stream1_yuv(self, frame):
        """Extract and cache YUV planes from stream1 frame for LC=2 combine."""
        fmt = frame.format.name
        if fmt not in ('yuv420p', 'yuvj420p'):
            frame = frame.reformat(format='yuv420p')
        self._avc444_full_range = frame.format.name in _FULL_RANGE_FORMATS
        self._avc444_y = _plane_to_array(frame.planes[0]).copy()
        self._avc444_u = _plane_to_array(frame.planes[1]).copy()
        self._avc444_v = _plane_to_array(frame.planes[2]).copy()
        self._avc444_w = frame.width
        self._avc444_h = frame.height

    def _prime_aux_decoder(self, h264_data):
        """Feed stream2 data to the auxiliary decoder to advance past its IDR."""
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
            self._dec2.on_hard_reset = lambda: self.on_hard_reset() if self.on_hard_reset else None
            log.debug("AVC444: auxiliary decoder initialized")
        return self._dec2

    def _decode_avc444_lc2(self, data, dest_width, dest_height):
        """Decode AVC444 LC=2 (chroma-upgrade) frame and combine with cached luma.

        Returns a BGRA numpy array on success, or b"" when the auxiliary
        decoder is not yet primed or the luma cache is empty.
        """
        if self._dec2 is None:
            log.debug("AVC444: LC=2 skipped (aux decoder not yet initialized)")
            return b""
        if self._avc444_y is None:
            log.debug("AVC444: LC=2 skipped (no cached luma)")
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

        fmt = aux_frame.format.name
        if fmt not in ('yuv420p', 'yuvj420p'):
            aux_frame = aux_frame.reformat(format='yuv420p')

        w, h = self._avc444_w, self._avc444_h
        bgra = _combine_avc444v2_bgra(
            self._avc444_y, self._avc444_u, self._avc444_v,
            aux_frame, self._avc444_full_range, w, h)

        log.debug("AVC444: LC=2 decoded and combined (%dx%d dest=%dx%d)" %
                  (w, h, dest_width, dest_height))

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

        nal_types = self._parse_and_cache_nals(h264_data)
        has_idr = 5 in nal_types

        if self._decode_count <= 5 or has_idr or self._null_count > 0:
            names = [self._NAL_NAMES.get(t, str(t)) for t in nal_types]
            log.debug("AVC: decode #%d NAL types: %s (h264Len=%d)" %
                     (self._decode_count, ','.join(names), len(h264_data)))

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
                return self._flush_and_retry_idr(packet)
            self._ctx.flush_buffers()
            self._needs_keyframe = True
            self._keyframe_wait_count = 0
            self._keyframe_wait_start = 0.0
        except Exception as e:
            log.warning("AVC: H.264 decode error: %s" % e)
            if has_idr and packet is not None:
                return self._flush_and_retry_idr(packet)
            self._ctx.flush_buffers()
            self._needs_keyframe = True
            self._keyframe_wait_count = 0
            self._keyframe_wait_start = 0.0
            self._hw_error_count += 1
            if self._hw_error_count >= self._HW_ERROR_THRESHOLD:
                self._hard_reset()

        return None

    _HW_ERROR_THRESHOLD = 5
    _KEYFRAME_WAIT_LIMIT = 900
    _KEYFRAME_WAIT_TIMEOUT = 15.0     # seconds (HW decoder, mirrors grdp keyframeWaitTimeout)
    _KEYFRAME_WAIT_TIMEOUT_SW = 5.0   # seconds (SW decoder, mirrors grdp keyframeWaitTimeoutSW)

    @property
    def last_receive_time(self):
        """Monotonic time of the most recent _decode_h264() call, or 0.0 if never called."""
        return self._last_receive_time

    @property
    def needs_keyframe(self):
        """True when the decoder is waiting for an IDR frame."""
        return self._needs_keyframe

    def _flush_and_retry_idr(self, packet):
        """Flush decoder buffers and retry an IDR packet after an error."""
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
        directly using BT.601 coefficients, bypassing swscale.  On ARM64
        (Apple Silicon) swscale's non-accelerated fallback ignores
        sws_setColorspaceDetails and produces wrong colours; the manual path
        fixes both correctness and performance.  This mirrors the optimisation
        introduced in grdp 0.7.7 (plugin/rdpgfx/h264_ffmpeg.go).

        VideoToolbox hardware frames (format='videotoolbox') are downloaded to
        NV12 first (a zero-cost pixel-copy, no colour conversion), then the
        NV12 fast path is taken.

        When *regions* is a non-empty list and sizes match, uses region-aware
        Cython functions (matching grdp's regionHint/setRegionHint pattern) to
        update only dirty rectangle pixels in the persistent pool buffer,
        saving YUV→BGRA conversion work proportional to the unchanged area.
        """
        fmt = frame.format.name

        if fmt == 'videotoolbox':
            frame = frame.reformat(format='nv12')
            fmt = 'nv12'

        if self._decode_count <= 3:
            log.debug("AVC: frame_to_bgra #%d src=%dx%d dest=%dx%d fmt=%s" %
                     (self._decode_count, frame.width, frame.height,
                      dest_width, dest_height, fmt))

        src_h, src_w = frame.height, frame.width

        if (regions and len(regions) > 0 and
                src_w == dest_width and src_h == dest_height and
                fmt in ('yuv420p', 'yuvj420p', 'nv12')):
            bgra = _frame_to_bgra_regions(frame, fmt, regions, src_h, src_w)
            if bgra is not None:
                if self._decode_count <= 1:
                    self._save_debug_frame(bgra, dest_width, dest_height)
                return bgra

        if fmt in ('yuv420p', 'yuvj420p'):
            bgra = _frame_yuv420p_to_bgra(frame, fmt in _FULL_RANGE_FORMATS)
        elif fmt == 'nv12':
            bgra = _frame_nv12_to_bgra(frame)
        else:
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
