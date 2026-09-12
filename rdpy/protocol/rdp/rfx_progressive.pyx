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
RFX Progressive Codec decoder (MS-RDPRFX / MS-RDPEGFX 2.2.4) — Cython accelerated.
Handles RDPGFX_CODECID_CAPROGRESSIVE (0x0009) in WIRE_TO_SURFACE_PDU_2.
"""

import struct
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int16_t, int32_t
from libc.string cimport memcpy, memset

np.import_array()

import rdpy.core.log as log
from rdpy.protocol.rdp.rlgr1_decode import rlgr1_decode

# Progressive block types
PROG_WBT_SYNC = 0xCCC0
PROG_WBT_FRAME_BEGIN = 0xCCC1
PROG_WBT_FRAME_END = 0xCCC2
PROG_WBT_CONTEXT = 0xCCC3
PROG_WBT_REGION = 0xCCC4
PROG_WBT_TILE_SIMPLE = 0xCCC5
PROG_WBT_TILE_FIRST = 0xCCC6
PROG_WBT_TILE_UPGRADE = 0xCCC7

RFX_TILE_SIZE = 64

# Sub-band sizes in RLGR decode order:
# HL1(1024) LH1(1024) HH1(1024) HL2(256) LH2(256) HH2(256) HL3(64) LH3(64) HH3(64) LL3(64)
_SUBBAND_SIZES = [1024, 1024, 1024, 256, 256, 256, 64, 64, 64, 64]
# Sub-band sizes for extrapolated DWT (RFX_DWT_REDUCE_EXTRAPOLATE):
_SUBBAND_SIZES_EXT = [1023, 1023, 961, 272, 272, 256, 72, 72, 64, 81]

cdef int[10] C_SUBBAND_SIZES = [1024, 1024, 1024, 256, 256, 256, 64, 64, 64, 64]
cdef int[10] C_SUBBAND_SIZES_EXT = [1023, 1023, 961, 272, 272, 256, 72, 72, 64, 81]


# ---------------------------------------------------------------
# Quantization classes
# ---------------------------------------------------------------

class RfxQuant:
    """TS_RFX_CODEC_QUANT: 5-byte packed quantization values per MS-RDPRFX 2.2.2.1.5."""
    __slots__ = ('LL3', 'LH3', 'HL3', 'HH3', 'LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1')

    def __init__(self, data):
        self.LL3 = data[0] & 0x0F
        self.HL3 = data[0] >> 4
        self.LH3 = data[1] & 0x0F
        self.HH3 = data[1] >> 4
        self.HL2 = data[2] & 0x0F
        self.LH2 = data[2] >> 4
        self.HH2 = data[3] & 0x0F
        self.HL1 = data[3] >> 4
        self.LH1 = data[4] & 0x0F
        self.HH1 = data[4] >> 4

    @staticmethod
    def default():
        q = object.__new__(RfxQuant)
        q.LL3 = q.LH3 = q.HL3 = q.HH3 = 6
        q.LH2 = q.HL2 = q.HH2 = 6
        q.LH1 = q.HL1 = q.HH1 = 6
        return q

    def subband_shifts(self):
        """Return shift values per subband in RLGR decode order."""
        return [self.HL1, self.LH1, self.HH1,
                self.HL2, self.LH2, self.HH2,
                self.HL3, self.LH3, self.HH3, self.LL3]


class ProgQuant:
    """RFX_PROGRESSIVE_CODEC_QUANT: 16-byte progressive quant entry."""
    __slots__ = ('quality', 'y', 'cb', 'cr')

    def __init__(self, data):
        self.quality = data[0]
        self.y = RfxQuant(data[1:6])
        self.cb = RfxQuant(data[6:11])
        self.cr = RfxQuant(data[11:16])


def _shift_val(v):
    return (v - 1) if v > 0 else 0


def _prog_quant_shift(q, pq=None, has_pq=False):
    res = object.__new__(RfxQuant)
    if not has_pq or pq is None:
        res.LL3 = _shift_val(q.LL3)
        res.HL3 = _shift_val(q.HL3)
        res.LH3 = _shift_val(q.LH3)
        res.HH3 = _shift_val(q.HH3)
        res.HL2 = _shift_val(q.HL2)
        res.LH2 = _shift_val(q.LH2)
        res.HH2 = _shift_val(q.HH2)
        res.HL1 = _shift_val(q.HL1)
        res.LH1 = _shift_val(q.LH1)
        res.HH1 = _shift_val(q.HH1)
    else:
        res.LL3 = _shift_val(q.LL3 + pq.LL3)
        res.HL3 = _shift_val(q.HL3 + pq.HL3)
        res.LH3 = _shift_val(q.LH3 + pq.LH3)
        res.HH3 = _shift_val(q.HH3 + pq.HH3)
        res.HL2 = _shift_val(q.HL2 + pq.HL2)
        res.LH2 = _shift_val(q.LH2 + pq.LH2)
        res.HH2 = _shift_val(q.HH2 + pq.HH2)
        res.HL1 = _shift_val(q.HL1 + pq.HL1)
        res.LH1 = _shift_val(q.LH1 + pq.LH1)
        res.HH1 = _shift_val(q.HH1 + pq.HH1)
    return res


def _quant_add(q1, q2):
    q = object.__new__(RfxQuant)
    q.LL3 = q1.LL3 + q2.LL3
    q.HL3 = q1.HL3 + q2.HL3
    q.LH3 = q1.LH3 + q2.LH3
    q.HH3 = q1.HH3 + q2.HH3
    q.HL2 = q1.HL2 + q2.HL2
    q.LH2 = q1.LH2 + q2.LH2
    q.HH2 = q1.HH2 + q2.HH2
    q.HL1 = q1.HL1 + q2.HL1
    q.LH1 = q1.LH1 + q2.LH1
    q.HH1 = q1.HH1 + q2.HH1
    return q


def _quant_sub(q1, q2):
    q = object.__new__(RfxQuant)
    sub = lambda a, b: max(0, a - b)
    q.LL3 = sub(q1.LL3, q2.LL3)
    q.HL3 = sub(q1.HL3, q2.HL3)
    q.LH3 = sub(q1.LH3, q2.LH3)
    q.HH3 = sub(q1.HH3, q2.HH3)
    q.HL2 = sub(q1.HL2, q2.HL2)
    q.LH2 = sub(q1.LH2, q2.LH2)
    q.HH2 = sub(q1.HH2, q2.HH2)
    q.HL1 = sub(q1.HL1, q2.HL1)
    q.LH1 = sub(q1.LH1, q2.LH1)
    q.HH1 = sub(q1.HH1, q2.HH1)
    return q


# ---------------------------------------------------------------
# Inverse 2D DWT (3 levels) — C level
# ---------------------------------------------------------------

cdef inline int16_t _clampi16(int val) noexcept nogil:
    if val < -32768:
        return -32768
    if val > 32767:
        return 32767
    return <int16_t>val


cdef inline int _trunc_div2(int x) noexcept nogil:
    return (x // 2) if x >= 0 else -(-x // 2)


cdef void _cy_idwt_2d_level(int16_t* buf, int n) noexcept nogil:
    """One level of inverse 2D DWT in pure C, ported from grdp rfxIDWT2DLevel.
    buf contains [HL(n²)|LH(n²)|HH(n²)|LL(n²)] → (2n)×(2n) result.
    """
    cdef int nn = n * n
    cdef int size = 2 * n
    cdef int row, col, x, row_off, l_dst_off, h_dst_off
    cdef int16_t prev_even_l, prev_even_h, even_l, even_h
    cdef int32_t l_val, h_val, even, prev_even, odd, last_even, last_h
    cdef int l_idx, h_idx, h_prev_idx

    cdef int16_t tmp[4096]

    cdef int16_t* hl = buf
    cdef int16_t* lh = buf + nn
    cdef int16_t* hh = buf + 2 * nn
    cdef int16_t* ll = buf + 3 * nn

    for row in range(n):
        row_off = row * n
        l_dst_off = row * size
        h_dst_off = (row + n) * size

        prev_even_l = ll[row_off] - <int16_t>((<int32_t>hl[row_off] * 2 + 1) >> 1)
        prev_even_h = lh[row_off] - <int16_t>((<int32_t>hh[row_off] * 2 + 1) >> 1)
        tmp[l_dst_off] = prev_even_l
        tmp[h_dst_off] = prev_even_h

        for col in range(1, n):
            x = col << 1
            even_l = ll[row_off + col] - <int16_t>((<int32_t>hl[row_off + col - 1] + <int32_t>hl[row_off + col] + 1) >> 1)
            even_h = lh[row_off + col] - <int16_t>((<int32_t>hh[row_off + col - 1] + <int32_t>hh[row_off + col] + 1) >> 1)
            tmp[l_dst_off + x - 1] = <int16_t>((<int32_t>hl[row_off + col - 1] << 1) + ((<int32_t>prev_even_l + <int32_t>even_l) >> 1))
            tmp[h_dst_off + x - 1] = <int16_t>((<int32_t>hh[row_off + col - 1] << 1) + ((<int32_t>prev_even_h + <int32_t>even_h) >> 1))
            tmp[l_dst_off + x] = even_l
            tmp[h_dst_off + x] = even_h
            prev_even_l = even_l
            prev_even_h = even_h

        x = (n - 1) << 1
        tmp[l_dst_off + x + 1] = <int16_t>((<int32_t>hl[row_off + n - 1] << 1) + <int32_t>prev_even_l)
        tmp[h_dst_off + x + 1] = <int16_t>((<int32_t>hh[row_off + n - 1] << 1) + <int32_t>prev_even_h)

    for col in range(size):
        l_val = <int32_t>tmp[col]
        h_val = <int32_t>tmp[n * size + col]
        buf[col] = <int16_t>(l_val - ((h_val * 2 + 1) >> 1))

        for row in range(1, n):
            l_idx = row * size + col
            h_idx = (row + n) * size + col
            h_prev_idx = (row - 1 + n) * size + col

            even = <int32_t>tmp[l_idx] - ((<int32_t>tmp[h_prev_idx] + <int32_t>tmp[h_idx] + 1) >> 1)
            buf[2 * row * size + col] = <int16_t>even

            prev_even = <int32_t>buf[(2 * row - 2) * size + col]
            odd = (<int32_t>tmp[h_prev_idx] << 1) + ((prev_even + even) >> 1)
            buf[(2 * row - 1) * size + col] = <int16_t>odd

        last_even = <int32_t>buf[(2 * n - 2) * size + col]
        last_h = <int32_t>tmp[(2 * n - 1) * size + col]
        buf[(2 * n - 1) * size + col] = <int16_t>((last_h << 1) + last_even)


cdef void _cy_inverse_dwt_2d(int16_t* coeffs) noexcept nogil:
    """3-level inverse 2D DWT in-place."""
    _cy_idwt_2d_level(coeffs + 3840, 8)   # Level 3: 8×8 → 16×16
    _cy_idwt_2d_level(coeffs + 3072, 16)  # Level 2: 16×16 → 32×32
    _cy_idwt_2d_level(coeffs + 0, 32)     # Level 1: 32×32 → 64×64


cdef void _cy_idwt_x(const int16_t* p_low, const int16_t* p_high, int16_t* p_dst,
                     int n_low_step, int n_high_step, int n_dst_step,
                     int n_low_count, int n_high_count, int n_dst_count) noexcept nogil:
    cdef int i, j, p_l, p_h, p_x
    cdef int H0, H1, L0, X0, X1, X2

    for i in range(n_dst_count):
        p_l = i * n_low_step
        p_h = i * n_high_step
        p_x = i * n_dst_step

        H0 = <int>p_high[p_h]
        p_h += 1
        L0 = <int>p_low[p_l]
        p_l += 1

        X0 = _clampi16(L0 - H0)
        X2 = X0

        for j in range(n_high_count - 1):
            H1 = <int>p_high[p_h]
            p_h += 1
            L0 = <int>p_low[p_l]
            p_l += 1
            X2 = _clampi16(L0 - _trunc_div2(H0 + H1))
            X1 = _clampi16(_trunc_div2(X0 + X2) + (2 * H0))
            p_dst[p_x] = X0
            p_dst[p_x + 1] = X1
            p_x += 2
            X0 = X2
            H0 = H1

        if n_low_count <= n_high_count + 1:
            if n_low_count <= n_high_count:
                p_dst[p_x] = X2
                p_dst[p_x + 1] = _clampi16(X2 + 2 * H0)
            else:
                L0 = <int>p_low[p_l]
                p_l += 1
                X0 = _clampi16(L0 - H0)
                p_dst[p_x] = X2
                p_dst[p_x + 1] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
                p_dst[p_x + 2] = X0
        else:
            L0 = <int>p_low[p_l]
            p_l += 1
            X0 = _clampi16(L0 - _trunc_div2(H0))
            p_dst[p_x] = X2
            p_dst[p_x + 1] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
            p_dst[p_x + 2] = X0
            L0 = <int>p_low[p_l]
            p_dst[p_x + 3] = _clampi16(_trunc_div2(X0 + L0))


cdef void _cy_idwt_y(const int16_t* p_low, const int16_t* p_high, int16_t* p_dst,
                     int n_low_step, int n_high_step, int n_dst_step,
                     int n_low_count, int n_high_count, int n_dst_count) noexcept nogil:
    cdef int i, j, p_l, p_h, p_x
    cdef int H0, H1, L0, X0, X1, X2

    for i in range(n_dst_count):
        p_l = i
        p_h = i
        p_x = i

        H0 = <int>p_high[p_h]
        p_h += n_high_step
        L0 = <int>p_low[p_l]
        p_l += n_low_step

        X0 = _clampi16(L0 - H0)
        X2 = X0

        for j in range(n_high_count - 1):
            H1 = <int>p_high[p_h]
            p_h += n_high_step
            L0 = <int>p_low[p_l]
            p_l += n_low_step
            X2 = _clampi16(L0 - _trunc_div2(H0 + H1))
            X1 = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
            p_dst[p_x] = X0
            p_x += n_dst_step
            p_dst[p_x] = X1
            p_x += n_dst_step
            X0 = X2
            H0 = H1

        if n_low_count <= n_high_count + 1:
            if n_low_count <= n_high_count:
                p_dst[p_x] = X2
                p_x += n_dst_step
                p_dst[p_x] = _clampi16(X2 + 2 * H0)
            else:
                L0 = <int>p_low[p_l]
                p_l += n_low_step
                X0 = _clampi16(L0 - H0)
                p_dst[p_x] = X2
                p_x += n_dst_step
                p_dst[p_x] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
                p_x += n_dst_step
                p_dst[p_x] = X0
        else:
            L0 = <int>p_low[p_l]
            p_l += n_low_step
            X0 = _clampi16(L0 - _trunc_div2(H0))
            p_dst[p_x] = X2
            p_x += n_dst_step
            p_dst[p_x] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
            p_x += n_dst_step
            p_dst[p_x] = X0
            p_x += n_dst_step
            L0 = <int>p_low[p_l]
            p_dst[p_x] = _clampi16(_trunc_div2(X0 + L0))


cdef void _cy_dwt_2d_decode_block(int16_t* buffer, int16_t* temp, int level) noexcept nogil:
    cdef int n_band_l = (64 >> level) + 1
    cdef int n_band_h
    if level == 1:
        n_band_h = (64 >> 1) - 1
    else:
        n_band_h = (64 + (1 << (level - 1))) >> level

    cdef int offset = 0
    cdef int16_t* hl = buffer + offset
    offset += n_band_h * n_band_l
    cdef int16_t* lh = buffer + offset
    offset += n_band_l * n_band_h
    cdef int16_t* hh = buffer + offset
    offset += n_band_h * n_band_h
    cdef int16_t* ll = buffer + offset

    cdef int n_dst_step = n_band_l + n_band_h

    cdef int16_t* temp_l = temp
    cdef int16_t* temp_h = temp + n_band_l * n_dst_step

    # horizontal (LL + HL -> L)
    _cy_idwt_x(ll, hl, temp_l, n_band_l, n_band_h, n_dst_step, n_band_l, n_band_h, n_band_l)

    # horizontal (LH + HH -> H)
    _cy_idwt_x(lh, hh, temp_h, n_band_l, n_band_h, n_dst_step, n_band_l, n_band_h, n_band_h)

    # vertical (L + H -> buffer)
    _cy_idwt_y(temp_l, temp_h, buffer, n_dst_step, n_dst_step, n_dst_step, n_band_l, n_band_h, n_band_l + n_band_h)


cdef void _cy_inverse_dwt_extrapolate(int16_t* buffer) noexcept nogil:
    cdef int16_t temp[4096]
    memset(temp, 0, sizeof(temp))
    _cy_dwt_2d_decode_block(buffer + 3807, temp, 3)
    _cy_dwt_2d_decode_block(buffer + 3007, temp, 2)
    _cy_dwt_2d_decode_block(buffer + 0, temp, 1)


# ---------------------------------------------------------------
# Color Transform and Surface Blit — C level
# ---------------------------------------------------------------

cdef void _cy_place_tile_abs(
    const int16_t* y_coeffs,
    const int16_t* cb_coeffs,
    const int16_t* cr_coeffs,
    int tile_x, int tile_y,
    uint8_t* output, int out_w, int out_h
) noexcept nogil:
    cdef int tile_w = 64
    cdef int tile_h = 64
    if tile_w > out_w - tile_x:
        tile_w = out_w - tile_x
    if tile_h > out_h - tile_y:
        tile_h = out_h - tile_y
    if tile_w <= 0 or tile_h <= 0:
        return

    cdef int row, col, y_idx
    cdef int y_val, cb_val, cr_val, y_scaled, r, g, b
    cdef int dst_stride = out_w * 4
    cdef uint8_t* dst_row
    cdef uint8_t* dst_pixel

    for row in range(tile_h):
        dst_row = output + (tile_y + row) * dst_stride + tile_x * 4
        for col in range(tile_w):
            y_idx = row * 64 + col
            y_val = <int>y_coeffs[y_idx]
            cb_val = <int>cb_coeffs[y_idx]
            cr_val = <int>cr_coeffs[y_idx]

            y_scaled = (y_val + 4096) << 16
            r = (cr_val * 91916 + y_scaled) >> 21
            g = (y_scaled - cb_val * 22527 - cr_val * 46819) >> 21
            b = (cb_val * 115992 + y_scaled) >> 21

            if r < 0: r = 0
            elif r > 255: r = 255
            if g < 0: g = 0
            elif g > 255: g = 255
            if b < 0: b = 0
            elif b > 255: b = 255

            dst_pixel = dst_row + col * 4
            dst_pixel[0] = <uint8_t>b
            dst_pixel[1] = <uint8_t>g
            dst_pixel[2] = <uint8_t>r
            dst_pixel[3] = 0xFF


# ---------------------------------------------------------------
# SRL & RAW BitStream for Progressive Upgrades — C level
# ---------------------------------------------------------------

cdef struct _CBitStream:
    const uint8_t* data
    int pos
    int length
    uint64_t bits
    int bit_len


cdef inline void _bs_init(_CBitStream* bs, const uint8_t* data, int length) noexcept nogil:
    bs.data = data
    bs.pos = 0
    bs.length = length
    bs.bits = 0
    bs.bit_len = 0
    _bs_fill(bs)


cdef inline void _bs_fill(_CBitStream* bs) noexcept nogil:
    while bs.bit_len <= 56 and bs.pos < bs.length:
        bs.bits |= (<uint64_t>bs.data[bs.pos]) << (56 - bs.bit_len)
        bs.pos += 1
        bs.bit_len += 8


cdef inline int _bs_read_bit(_CBitStream* bs) noexcept nogil:
    if bs.bit_len < 1:
        _bs_fill(bs)
        if bs.bit_len < 1:
            return 0
    cdef int bit = <int>((bs.bits >> 63) & 1)
    bs.bits = (bs.bits << 1)
    bs.bit_len -= 1
    return bit


cdef inline int _bs_read_bits(_CBitStream* bs, int n) noexcept nogil:
    if n <= 0:
        return 0
    if bs.bit_len < n:
        _bs_fill(bs)
        if bs.bit_len < n:
            n = bs.bit_len
            if n <= 0:
                return 0
    cdef int val = <int>((bs.bits >> (64 - n)) & ((1ULL << n) - 1))
    bs.bits = (bs.bits << n)
    bs.bit_len -= n
    return val


cdef struct _CSRLState:
    _CBitStream srl
    _CBitStream raw
    int kp
    int nz
    int mode


cdef inline void _srl_init(_CSRLState* st, const uint8_t* srl_data, int srl_len,
                           const uint8_t* raw_data, int raw_len) noexcept nogil:
    _bs_init(&st.srl, srl_data, srl_len)
    _bs_init(&st.raw, raw_data, raw_len)
    st.kp = 8
    st.nz = 0
    st.mode = 0


cdef inline int _srl_read(_CSRLState* st, int num_bits) noexcept nogil:
    cdef int k, bit, sign, mag, max_val
    if st.nz > 0:
        st.nz -= 1
        return 0
    k = st.kp // 8
    if not st.mode:
        bit = _bs_read_bit(&st.srl)
        if bit == 0:
            st.nz = (1 << k) - 1
            st.kp += 4
            if st.kp > 80:
                st.kp = 80
            return 0
        else:
            st.nz = 0
            st.mode = 1
            if k > 0:
                st.nz = _bs_read_bits(&st.srl, k)
            if st.nz > 0:
                st.nz -= 1
                return 0
    st.mode = 0
    sign = _bs_read_bit(&st.srl)
    if st.kp < 6:
        st.kp = 0
    else:
        st.kp -= 6
    if num_bits == 1:
        return -1 if sign != 0 else 1
    mag = 1
    max_val = (1 << num_bits) - 1
    while mag < max_val:
        bit = _bs_read_bit(&st.srl)
        if bit != 0:
            break
        mag += 1
    return -mag if sign != 0 else mag


cdef void _cy_upgrade_block(_CSRLState* st, int16_t* current, int16_t* sign,
                            int length, int shift, int num_bits, int is_ll) noexcept nogil:
    cdef int i, input_val, srl_val
    if num_bits == 0 or length <= 0:
        return
    if is_ll:
        for i in range(length):
            input_val = _bs_read_bits(&st.raw, num_bits)
            current[i] += <int16_t>(input_val << shift)
        return

    for i in range(length):
        if sign[i] > 0:
            input_val = _bs_read_bits(&st.raw, num_bits)
        elif sign[i] < 0:
            input_val = -_bs_read_bits(&st.raw, num_bits)
        else:
            srl_val = _srl_read(st, num_bits)
            sign[i] = <int16_t>srl_val
            input_val = srl_val
        current[i] += <int16_t>(input_val << shift)


cdef void _cy_upgrade_component(const uint8_t* srl_data, int srl_len,
                                const uint8_t* raw_data, int raw_len,
                                shift_shifts, num_bits_shifts,
                                int16_t* current, int16_t* sign,
                                int extrapolate) noexcept:
    cdef _CSRLState st
    _srl_init(&st, srl_data, srl_len, raw_data, raw_len)

    cdef int s_HL1 = shift_shifts.HL1, nb_HL1 = num_bits_shifts.HL1
    cdef int s_LH1 = shift_shifts.LH1, nb_LH1 = num_bits_shifts.LH1
    cdef int s_HH1 = shift_shifts.HH1, nb_HH1 = num_bits_shifts.HH1
    cdef int s_HL2 = shift_shifts.HL2, nb_HL2 = num_bits_shifts.HL2
    cdef int s_LH2 = shift_shifts.LH2, nb_LH2 = num_bits_shifts.LH2
    cdef int s_HH2 = shift_shifts.HH2, nb_HH2 = num_bits_shifts.HH2
    cdef int s_HL3 = shift_shifts.HL3, nb_HL3 = num_bits_shifts.HL3
    cdef int s_LH3 = shift_shifts.LH3, nb_LH3 = num_bits_shifts.LH3
    cdef int s_HH3 = shift_shifts.HH3, nb_HH3 = num_bits_shifts.HH3
    cdef int s_LL3 = shift_shifts.LL3, nb_LL3 = num_bits_shifts.LL3

    with nogil:
        if not extrapolate:
            _cy_upgrade_block(&st, current + 0,    sign + 0,    1024, s_HL1, nb_HL1, 0)
            _cy_upgrade_block(&st, current + 1024, sign + 1024, 1024, s_LH1, nb_LH1, 0)
            _cy_upgrade_block(&st, current + 2048, sign + 2048, 1024, s_HH1, nb_HH1, 0)
            _cy_upgrade_block(&st, current + 3072, sign + 3072, 256,  s_HL2, nb_HL2, 0)
            _cy_upgrade_block(&st, current + 3328, sign + 3328, 256,  s_LH2, nb_LH2, 0)
            _cy_upgrade_block(&st, current + 3584, sign + 3584, 256,  s_HH2, nb_HH2, 0)
            _cy_upgrade_block(&st, current + 3840, sign + 3840, 64,   s_HL3, nb_HL3, 0)
            _cy_upgrade_block(&st, current + 3904, sign + 3904, 64,   s_LH3, nb_LH3, 0)
            _cy_upgrade_block(&st, current + 3968, sign + 3968, 64,   s_HH3, nb_HH3, 0)
            _cy_upgrade_block(&st, current + 4032, sign + 4032, 64,   s_LL3, nb_LL3, 1)
        else:
            _cy_upgrade_block(&st, current + 0,    sign + 0,    1023, s_HL1, nb_HL1, 0)
            _cy_upgrade_block(&st, current + 1023, sign + 1023, 1023, s_LH1, nb_LH1, 0)
            _cy_upgrade_block(&st, current + 2046, sign + 2046, 961,  s_HH1, nb_HH1, 0)
            _cy_upgrade_block(&st, current + 3007, sign + 3007, 272,  s_HL2, nb_HL2, 0)
            _cy_upgrade_block(&st, current + 3279, sign + 3279, 272,  s_LH2, nb_LH2, 0)
            _cy_upgrade_block(&st, current + 3551, sign + 3551, 256,  s_HH2, nb_HH2, 0)
            _cy_upgrade_block(&st, current + 3807, sign + 3807, 72,   s_HL3, nb_HL3, 0)
            _cy_upgrade_block(&st, current + 3879, sign + 3879, 72,   s_LH3, nb_LH3, 0)
            _cy_upgrade_block(&st, current + 3951, sign + 3951, 64,   s_HH3, nb_HH3, 0)
            _cy_upgrade_block(&st, current + 4015, sign + 4015, 81,   s_LL3, nb_LL3, 1)


# ---------------------------------------------------------------
# Python Wrapper functions (matching rfx_progressive.py signatures)
# ---------------------------------------------------------------

def _inverse_dwt_2d(np.ndarray[int16_t, ndim=1] coeffs):
    _cy_inverse_dwt_2d(<int16_t*>coeffs.data)


def _inverse_dwt_extrapolate(np.ndarray[int16_t, ndim=1] buffer):
    _cy_inverse_dwt_extrapolate(<int16_t*>buffer.data)


def _place_tile_abs(y_coeffs,
                    cb_coeffs,
                    cr_coeffs,
                    int tile_x, int tile_y,
                    output, int out_w, int out_h):
    if isinstance(y_coeffs, (tuple, list)):
        y_coeffs = y_coeffs[0]
    if isinstance(cb_coeffs, (tuple, list)):
        cb_coeffs = cb_coeffs[0]
    if isinstance(cr_coeffs, (tuple, list)):
        cr_coeffs = cr_coeffs[0]

    cdef np.ndarray[int16_t, ndim=1] y_arr = y_coeffs
    cdef np.ndarray[int16_t, ndim=1] cb_arr = cb_coeffs
    cdef np.ndarray[int16_t, ndim=1] cr_arr = cr_coeffs

    cdef uint8_t[:] out_mv
    if isinstance(output, (bytes, bytearray, memoryview, np.ndarray)):
        out_mv = np.frombuffer(output, dtype=np.uint8)
    else:
        out_mv = output
    _cy_place_tile_abs(<int16_t*>y_arr.data,
                       <int16_t*>cb_arr.data,
                       <int16_t*>cr_arr.data,
                       tile_x, tile_y,
                       &out_mv[0], out_w, out_h)


def _place_tile(y_coeffs, cb_coeffs, cr_coeffs, int x_idx, int y_idx, output, int out_w, int out_h):
    _place_tile_abs(y_coeffs, cb_coeffs, cr_coeffs,
                    x_idx * RFX_TILE_SIZE, y_idx * RFX_TILE_SIZE,
                    output, out_w, out_h)


def _decode_component(data, shift, prev_dequant=None, is_diff=False, extrapolate=False):
    """Decode one color component (Y/Cb/Cr) for a 64×64 tile."""
    cdef np.ndarray[int16_t, ndim=1] coeffs
    cdef np.ndarray[int16_t, ndim=1] raw_sign
    cdef np.ndarray[int16_t, ndim=1] dequant
    cdef np.ndarray[int16_t, ndim=1] work

    if data:
        coeffs = rlgr1_decode(data, 4096)
        raw_sign = coeffs.copy()
    else:
        coeffs = np.zeros(4096, dtype=np.int16)
        raw_sign = np.zeros(4096, dtype=np.int16)

    cdef int16_t* c_ptr = <int16_t*>coeffs.data
    cdef int i, acc, s_idx, cnt, pos

    shifts = shift.subband_shifts()
    cdef int[10] sub_shifts
    for i in range(10):
        sub_shifts[i] = shifts[i]

    if not extrapolate:
        acc = 0
        for i in range(4032, 4096):
            acc += c_ptr[i]
            c_ptr[i] = <int16_t>acc

        pos = 0
        for s_idx in range(10):
            cnt = C_SUBBAND_SIZES[s_idx]
            for i in range(cnt):
                c_ptr[pos + i] = <int16_t>(c_ptr[pos + i] << sub_shifts[s_idx])
            pos += cnt
    else:
        acc = 0
        for i in range(4015, 4096):
            acc += c_ptr[i]
            c_ptr[i] = <int16_t>acc

        pos = 0
        for s_idx in range(10):
            cnt = C_SUBBAND_SIZES_EXT[s_idx]
            for i in range(cnt):
                c_ptr[pos + i] = <int16_t>(c_ptr[pos + i] << sub_shifts[s_idx])
            pos += cnt

    if is_diff and prev_dequant is not None:
        prev_ptr = <int16_t*>(<np.ndarray[int16_t, ndim=1]>prev_dequant).data
        for i in range(4096):
            c_ptr[i] = _clampi16(<int>c_ptr[i] + <int>prev_ptr[i])

    dequant = coeffs.copy()
    work = coeffs.copy()
    if not extrapolate:
        _cy_inverse_dwt_2d(<int16_t*>work.data)
    else:
        _cy_inverse_dwt_extrapolate(<int16_t*>work.data)

    return work, dequant, raw_sign


def _upgrade_component(data_srl, data_raw, shift, num_bits,
                       np.ndarray[int16_t, ndim=1] current,
                       np.ndarray[int16_t, ndim=1] sign,
                       extrapolate=False):
    cdef const uint8_t* srl_ptr = NULL
    cdef const uint8_t* raw_ptr = NULL
    cdef int srl_len = len(data_srl) if data_srl else 0
    cdef int raw_len = len(data_raw) if data_raw else 0
    cdef bytes srl_bytes = bytes(data_srl) if data_srl else b''
    cdef bytes raw_bytes = bytes(data_raw) if data_raw else b''

    if srl_len > 0:
        srl_ptr = <const uint8_t*>srl_bytes
    if raw_len > 0:
        raw_ptr = <const uint8_t*>raw_bytes

    _cy_upgrade_component(srl_ptr, srl_len, raw_ptr, raw_len,
                          shift, num_bits,
                          <int16_t*>current.data, <int16_t*>sign.data,
                          1 if extrapolate else 0)

    cdef np.ndarray[int16_t, ndim=1] work = current.copy()
    if not extrapolate:
        _cy_inverse_dwt_2d(<int16_t*>work.data)
    else:
        _cy_inverse_dwt_extrapolate(<int16_t*>work.data)
    return work


# ---------------------------------------------------------------
# Progressive stream parser & tile decoder
# ---------------------------------------------------------------

class RfxProgressiveDecoder:
    """Decodes RFX Progressive codec data from RDPGFX WIRE_TO_SURFACE_2."""

    def __init__(self):
        self._debug_tile_count = 0
        self._tileState = {}

    def reset(self):
        """Reset tile state (called on DELETEENCODINGCONTEXT / RESET_GRAPHICS)."""
        self._tileState.clear()
        self._debug_tile_count = 0

    def decode(self, data, surf_data, int width, int height):
        """Decode progressive bitmap stream, render tiles onto surf_data (BGRA buffer).
        Returns list of (x, y, w, h) bounding rectangles."""
        rects = []
        cdef int offset = 0
        cdef int data_len = len(data)
        cdef uint16_t block_type
        cdef uint32_t block_len

        while offset + 6 <= data_len:
            block_type, block_len = struct.unpack_from('<HI', data, offset)
            if block_len < 6 or offset + block_len > data_len:
                break

            block_data = data[offset + 6:offset + block_len]

            if block_type == PROG_WBT_REGION:
                rects.extend(self._decode_region(block_data, surf_data, width, height))
            elif block_type in (PROG_WBT_SYNC, PROG_WBT_FRAME_BEGIN,
                                PROG_WBT_FRAME_END, PROG_WBT_CONTEXT):
                pass

            offset += block_len

        return rects

    def _decode_region(self, data, output, int width, int height):
        if len(data) < 12:
            return []

        cdef uint16_t num_rects = struct.unpack_from('<H', data, 1)[0]
        cdef uint8_t num_quant = data[3]
        cdef uint8_t num_prog_quant = data[4]
        cdef uint8_t flags = data[5]
        cdef uint16_t num_tiles = struct.unpack_from('<H', data, 6)[0]

        cdef int offset = 12
        cdef int data_len = len(data)
        cdef uint16_t rx, ry, rw, rh, tx, ty

        rects = []
        for _ in range(num_rects):
            if offset + 8 > data_len:
                return []
            rx, ry, rw, rh = struct.unpack_from('<HHHH', data, offset)
            rects.append((rx, ry, rw, rh))
            offset += 8

        quants = []
        for _ in range(num_quant):
            if offset + 5 > data_len:
                return []
            quants.append(RfxQuant(data[offset:offset + 5]))
            offset += 5

        prog_quants = []
        for _ in range(num_prog_quant):
            if offset + 16 > data_len:
                return []
            prog_quants.append(ProgQuant(data[offset:offset + 16]))
            offset += 16

        extrapolate = (flags & 0x01) != 0

        tile_positions = []
        cdef uint16_t tile_type
        cdef uint32_t tile_len

        for _ in range(num_tiles):
            if offset + 6 > data_len:
                break
            tile_type, tile_len = struct.unpack_from('<HI', data, offset)
            if tile_len < 6 or offset + tile_len > data_len:
                break

            tile_data = data[offset + 6:offset + tile_len]
            if len(tile_data) >= 7:
                tx, ty = struct.unpack_from('<HH', tile_data, 3)
                tile_positions.append((tx, ty))

            if tile_type == PROG_WBT_TILE_SIMPLE:
                self._decode_tile_simple(tile_data, quants, output, width, height, extrapolate)
            elif tile_type == PROG_WBT_TILE_FIRST:
                self._decode_tile_first(tile_data, quants, prog_quants, output, width, height, extrapolate)
            elif tile_type == PROG_WBT_TILE_UPGRADE:
                self._decode_tile_upgrade(tile_data, quants, prog_quants, output, width, height, extrapolate)

            offset += tile_len

        if num_rects == 0 and tile_positions:
            min_x = min(tx * 64 for (tx, ty) in tile_positions)
            min_y = min(ty * 64 for (tx, ty) in tile_positions)
            max_x = min(max(tx * 64 + 64 for (tx, ty) in tile_positions), width)
            max_y = min(max(ty * 64 + 64 for (tx, ty) in tile_positions), height)
            rects = [(min_x, min_y, max_x - min_x, max_y - min_y)]
        elif num_rects == 0:
            rects = [(0, 0, width, height)]

        return rects

    def _get_quant(self, quants, int idx):
        if idx < len(quants):
            return quants[idx]
        return RfxQuant.default()

    def _decode_tile_simple(self, data, quants, output, int out_w, int out_h, extrapolate=False):
        if len(data) < 16:
            return
        cdef uint8_t quant_idx_y = data[0]
        cdef uint8_t quant_idx_cb = data[1]
        cdef uint8_t quant_idx_cr = data[2]
        cdef uint16_t x_idx, y_idx
        x_idx, y_idx = struct.unpack_from('<HH', data, 3)
        cdef uint8_t flags = data[7]
        cdef uint16_t y_len, cb_len, cr_len
        y_len, cb_len, cr_len = struct.unpack_from('<HHH', data, 8)

        is_diff = (flags & 0x01) != 0

        cdef int off = 16
        y_data = data[off:off + y_len] if y_len > 0 else None
        off += y_len
        cb_data = data[off:off + cb_len] if cb_len > 0 else None
        off += cb_len
        cr_data = data[off:off + cr_len] if cr_len > 0 else None

        q_y = self._get_quant(quants, quant_idx_y)
        q_cb = self._get_quant(quants, quant_idx_cb)
        q_cr = self._get_quant(quants, quant_idx_cr)

        shift_y = _prog_quant_shift(q_y)
        shift_cb = _prog_quant_shift(q_cb)
        shift_cr = _prog_quant_shift(q_cr)

        key = (x_idx, y_idx)
        prev_y = prev_cb = prev_cr = None
        if is_diff and key in self._tileState:
            prev_y = self._tileState[key]['y']
            prev_cb = self._tileState[key]['cb']
            prev_cr = self._tileState[key]['cr']

        y_pixels, new_y, sign_y = _decode_component(y_data, shift_y, prev_y, is_diff, extrapolate)
        cb_pixels, new_cb, sign_cb = _decode_component(cb_data, shift_cb, prev_cb, is_diff, extrapolate)
        cr_pixels, new_cr, sign_cr = _decode_component(cr_data, shift_cr, prev_cr, is_diff, extrapolate)

        _place_tile(y_pixels, cb_pixels, cr_pixels, x_idx, y_idx, output, out_w, out_h)

        self._tileState[key] = {
            'y': new_y, 'cb': new_cb, 'cr': new_cr,
            'sign_y': sign_y, 'sign_cb': sign_cb, 'sign_cr': sign_cr,
            'y_bit_pos': q_y, 'cb_bit_pos': q_cb, 'cr_bit_pos': q_cr,
            'quant_y': q_y, 'quant_cb': q_cb, 'quant_cr': q_cr,
        }

    def _decode_tile_first(self, data, quants, prog_quants, output, int out_w, int out_h, extrapolate=False):
        if len(data) < 17:
            return
        cdef uint8_t quant_idx_y = data[0]
        cdef uint8_t quant_idx_cb = data[1]
        cdef uint8_t quant_idx_cr = data[2]
        cdef uint16_t x_idx, y_idx
        x_idx, y_idx = struct.unpack_from('<HH', data, 3)
        cdef uint8_t flags = data[7]
        cdef uint8_t quality = data[8]
        cdef uint16_t y_len, cb_len, cr_len
        y_len, cb_len, cr_len = struct.unpack_from('<HHH', data, 9)

        is_diff = (flags & 0x01) != 0

        cdef int off = 17
        y_data = data[off:off + y_len] if y_len > 0 else None
        off += y_len
        cb_data = data[off:off + cb_len] if cb_len > 0 else None
        off += cb_len
        cr_data = data[off:off + cr_len] if cr_len > 0 else None

        q_y = self._get_quant(quants, quant_idx_y)
        q_cb = self._get_quant(quants, quant_idx_cb)
        q_cr = self._get_quant(quants, quant_idx_cr)

        pq = self._get_prog_quant(prog_quants, quality)
        pq_y = pq.y if pq else None
        pq_cb = pq.cb if pq else None
        pq_cr = pq.cr if pq else None
        has_pq = pq is not None

        shift_y = _prog_quant_shift(q_y, pq_y, has_pq)
        shift_cb = _prog_quant_shift(q_cb, pq_cb, has_pq)
        shift_cr = _prog_quant_shift(q_cr, pq_cr, has_pq)

        bit_pos_y = _quant_add(q_y, pq_y) if has_pq else q_y
        bit_pos_cb = _quant_add(q_cb, pq_cb) if has_pq else q_cb
        bit_pos_cr = _quant_add(q_cr, pq_cr) if has_pq else q_cr

        key = (x_idx, y_idx)
        prev_y = prev_cb = prev_cr = None
        if is_diff and key in self._tileState:
            prev_y = self._tileState[key]['y']
            prev_cb = self._tileState[key]['cb']
            prev_cr = self._tileState[key]['cr']

        y_pixels, new_y, sign_y = _decode_component(y_data, shift_y, prev_y, is_diff, extrapolate)
        cb_pixels, new_cb, sign_cb = _decode_component(cb_data, shift_cb, prev_cb, is_diff, extrapolate)
        cr_pixels, new_cr, sign_cr = _decode_component(cr_data, shift_cr, prev_cr, is_diff, extrapolate)

        _place_tile(y_pixels, cb_pixels, cr_pixels, x_idx, y_idx, output, out_w, out_h)

        self._tileState[key] = {
            'y': new_y, 'cb': new_cb, 'cr': new_cr,
            'sign_y': sign_y, 'sign_cb': sign_cb, 'sign_cr': sign_cr,
            'y_bit_pos': bit_pos_y, 'cb_bit_pos': bit_pos_cb, 'cr_bit_pos': bit_pos_cr,
            'quant_y': q_y, 'quant_cb': q_cb, 'quant_cr': q_cr,
        }

    def _decode_tile_upgrade(self, data, quants, prog_quants, output, int out_w, int out_h, extrapolate=False):
        if len(data) < 20:
            return
        cdef uint8_t quant_idx_y = data[0]
        cdef uint8_t quant_idx_cb = data[1]
        cdef uint8_t quant_idx_cr = data[2]
        cdef uint16_t x_idx, y_idx
        x_idx, y_idx = struct.unpack_from('<HH', data, 3)
        cdef uint8_t quality = data[7]
        cdef uint16_t y_srl_len, y_raw_len, cb_srl_len, cb_raw_len, cr_srl_len, cr_raw_len
        y_srl_len, y_raw_len, cb_srl_len, cb_raw_len, cr_srl_len, cr_raw_len = struct.unpack_from('<HHHHHH', data, 8)

        cdef int off = 20
        y_srl = data[off:off + y_srl_len]; off += y_srl_len
        y_raw = data[off:off + y_raw_len]; off += y_raw_len
        cb_srl = data[off:off + cb_srl_len]; off += cb_srl_len
        cb_raw = data[off:off + cb_raw_len]; off += cb_raw_len
        cr_srl = data[off:off + cr_srl_len]; off += cr_srl_len
        cr_raw = data[off:off + cr_raw_len]; off += cr_raw_len

        key = (x_idx, y_idx)
        state = self._tileState.get(key)
        if state is None or 'y' not in state or 'sign_y' not in state:
            return

        q_y = self._get_quant(quants, quant_idx_y)
        q_cb = self._get_quant(quants, quant_idx_cb)
        q_cr = self._get_quant(quants, quant_idx_cr)

        pq = self._get_prog_quant(prog_quants, quality)
        pq_y = pq.y if pq else None
        pq_cb = pq.cb if pq else None
        pq_cr = pq.cr if pq else None

        new_y_bit_pos = _quant_add(q_y, pq_y) if pq_y else q_y
        new_cb_bit_pos = _quant_add(q_cb, pq_cb) if pq_cb else q_cb
        new_cr_bit_pos = _quant_add(q_cr, pq_cr) if pq_cr else q_cr

        num_bits_y = _quant_sub(state['y_bit_pos'], new_y_bit_pos)
        num_bits_cb = _quant_sub(state['cb_bit_pos'], new_cb_bit_pos)
        num_bits_cr = _quant_sub(state['cr_bit_pos'], new_cr_bit_pos)

        shift_y = _prog_quant_shift(new_y_bit_pos)
        shift_cb = _prog_quant_shift(new_cb_bit_pos)
        shift_cr = _prog_quant_shift(new_cr_bit_pos)

        state['y_bit_pos'] = new_y_bit_pos
        state['cb_bit_pos'] = new_cb_bit_pos
        state['cr_bit_pos'] = new_cr_bit_pos

        y_pixels = _upgrade_component(y_srl, y_raw, shift_y, num_bits_y, state['y'], state['sign_y'], extrapolate)
        cb_pixels = _upgrade_component(cb_srl, cb_raw, shift_cb, num_bits_cb, state['cb'], state['sign_cb'], extrapolate)
        cr_pixels = _upgrade_component(cr_srl, cr_raw, shift_cr, num_bits_cr, state['cr'], state['sign_cr'], extrapolate)

        _place_tile(y_pixels, cb_pixels, cr_pixels, x_idx, y_idx, output, out_w, out_h)

    def _get_prog_quant(self, prog_quants, int quality):
        if quality < len(prog_quants):
            return prog_quants[quality]
        return None
