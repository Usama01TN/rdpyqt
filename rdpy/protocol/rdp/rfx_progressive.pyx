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
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset

np.import_array()

import rdpy.core.log as log

# Progressive block types
PROG_WBT_SYNC = 0xCCC0
PROG_WBT_FRAME_BEGIN = 0xCCC1
PROG_WBT_FRAME_END = 0xCCC2
PROG_WBT_CONTEXT = 0xCCC3
PROG_WBT_REGION = 0xCCC4
PROG_WBT_TILE_SIMPLE = 0xCCC5
PROG_WBT_TILE_FIRST = 0xCCC6
PROG_WBT_TILE_UPGRADE = 0xCCC7

cdef enum:
    C_RFX_TILE_SIZE = 64

RFX_TILE_SIZE = 64


class RfxQuant:
    """Quantization factors for RFX sub-bands."""

    def __init__(self, data=b'\x66\x66\x66\x66\x66'):
        if isinstance(data, (bytes, bytearray)) and len(data) >= 5:
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
            self.data = bytes(data[:5])
        else:
            self.LL3 = self.HL3 = self.LH3 = self.HH3 = 6
            self.HL2 = self.LH2 = self.HH2 = 6
            self.HL1 = self.LH1 = self.HH1 = 6
            self.data = b'\x66\x66\x66\x66\x66'


# Sub-band sizes in RLGR decode order:
# HL1(1024) LH1(1024) HH1(1024) HL2(256) LH2(256) HH2(256) HL3(64) LH3(64) HH3(64) LL3(64)
cdef int[10] C_SUBBAND_SIZES = [1024, 1024, 1024, 256, 256, 256, 64, 64, 64, 64]
# Sub-band sizes for extrapolated DWT (RFX_DWT_REDUCE_EXTRAPOLATE):
cdef int[10] C_SUBBAND_SIZES_EXT = [1023, 1023, 961, 272, 272, 256, 72, 72, 64, 81]


# ---------------------------------------------------------------
# Quantization C structures & operations
# ---------------------------------------------------------------

cdef struct CRfxQuant:
    uint8_t LL3, LH3, HL3, HH3
    uint8_t LH2, HL2, HH2
    uint8_t LH1, HL1, HH1


cdef inline CRfxQuant _c_default_quant() noexcept nogil:
    cdef CRfxQuant q
    q.LL3 = q.LH3 = q.HL3 = q.HH3 = 6
    q.LH2 = q.HL2 = q.HH2 = 6
    q.LH1 = q.HL1 = q.HH1 = 6
    return q


cdef inline CRfxQuant _c_parse_quant(const uint8_t* data) noexcept nogil:
    cdef CRfxQuant q
    q.LL3 = data[0] & 0x0F
    q.HL3 = data[0] >> 4
    q.LH3 = data[1] & 0x0F
    q.HH3 = data[1] >> 4
    q.HL2 = data[2] & 0x0F
    q.LH2 = data[2] >> 4
    q.HH2 = data[3] & 0x0F
    q.HL1 = data[3] >> 4
    q.LH1 = data[4] & 0x0F
    q.HH1 = data[4] >> 4
    return q


cdef struct CRfxProgQuant:
    uint8_t quality
    CRfxQuant y
    CRfxQuant cb
    CRfxQuant cr


cdef inline CRfxProgQuant _c_parse_prog_quant(const uint8_t* data) noexcept nogil:
    cdef CRfxProgQuant pq
    pq.quality = data[0]
    pq.y = _c_parse_quant(data + 1)
    pq.cb = _c_parse_quant(data + 6)
    pq.cr = _c_parse_quant(data + 11)
    return pq


cdef inline uint8_t _c_shift_val(uint8_t v) noexcept nogil:
    return (v - 1) if v > 0 else 0


cdef inline CRfxQuant _c_prog_quant_shift(CRfxQuant q, CRfxQuant pq, bint has_pq) noexcept nogil:
    cdef CRfxQuant res
    if not has_pq:
        res.LL3 = _c_shift_val(q.LL3)
        res.HL3 = _c_shift_val(q.HL3)
        res.LH3 = _c_shift_val(q.LH3)
        res.HH3 = _c_shift_val(q.HH3)
        res.HL2 = _c_shift_val(q.HL2)
        res.LH2 = _c_shift_val(q.LH2)
        res.HH2 = _c_shift_val(q.HH2)
        res.HL1 = _c_shift_val(q.HL1)
        res.LH1 = _c_shift_val(q.LH1)
        res.HH1 = _c_shift_val(q.HH1)
    else:
        res.LL3 = _c_shift_val(q.LL3 + pq.LL3)
        res.HL3 = _c_shift_val(q.HL3 + pq.HL3)
        res.LH3 = _c_shift_val(q.LH3 + pq.LH3)
        res.HH3 = _c_shift_val(q.HH3 + pq.HH3)
        res.HL2 = _c_shift_val(q.HL2 + pq.HL2)
        res.LH2 = _c_shift_val(q.LH2 + pq.LH2)
        res.HH2 = _c_shift_val(q.HH2 + pq.HH2)
        res.HL1 = _c_shift_val(q.HL1 + pq.HL1)
        res.LH1 = _c_shift_val(q.LH1 + pq.LH1)
        res.HH1 = _c_shift_val(q.HH1 + pq.HH1)
    return res


cdef inline CRfxQuant _c_quant_add(CRfxQuant q1, CRfxQuant q2) noexcept nogil:
    cdef CRfxQuant q
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


cdef inline uint8_t _c_sub(uint8_t a, uint8_t b) noexcept nogil:
    return (a - b) if a >= b else 0


cdef inline CRfxQuant _c_quant_sub(CRfxQuant q1, CRfxQuant q2) noexcept nogil:
    cdef CRfxQuant q
    q.LL3 = _c_sub(q1.LL3, q2.LL3)
    q.HL3 = _c_sub(q1.HL3, q2.HL3)
    q.LH3 = _c_sub(q1.LH3, q2.LH3)
    q.HH3 = _c_sub(q1.HH3, q2.HH3)
    q.HL2 = _c_sub(q1.HL2, q2.HL2)
    q.LH2 = _c_sub(q1.LH2, q2.LH2)
    q.HH2 = _c_sub(q1.HH2, q2.HH2)
    q.HL1 = _c_sub(q1.HL1, q2.HL1)
    q.LH1 = _c_sub(q1.LH1, q2.LH1)
    q.HH1 = _c_sub(q1.HH1, q2.HH1)
    return q


# Python compatibility wrapper classes
class RfxQuant:
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
        return [self.HL1, self.LH1, self.HH1,
                self.HL2, self.LH2, self.HH2,
                self.HL3, self.LH3, self.HH3, self.LL3]


class ProgQuant:
    __slots__ = ('quality', 'y', 'cb', 'cr')

    def __init__(self, data):
        self.quality = data[0]
        self.y = RfxQuant(data[1:6])
        self.cb = RfxQuant(data[6:11])
        self.cr = RfxQuant(data[11:16])


cdef extern int __builtin_clzll(unsigned long long) noexcept nogil

# ---------------------------------------------------------------
# RLGR1 Entropy Decoder — Pure C nogil (MSB-aligned with __builtin_clzll)
# ---------------------------------------------------------------

DEF _LSGR = 3
DEF _KPMAX = 80
DEF _UPGR = 4
DEF _DNGR = 6
DEF _UQGR = 3
DEF _DQGR = 3


cdef struct _CRLGRBitReader:
    const uint8_t* data
    int length
    int byte_pos
    uint64_t acc
    int bits_in_acc


cdef inline void _rlgr_br_init(_CRLGRBitReader* br, const uint8_t* data, int length) noexcept nogil:
    br.data = data
    br.length = length
    br.byte_pos = 0
    br.acc = 0
    br.bits_in_acc = 0


cdef inline int _rlgr_br_remaining(_CRLGRBitReader* br) noexcept nogil:
    return (br.length - br.byte_pos) * 8 + br.bits_in_acc


cdef inline void _rlgr_br_fill(_CRLGRBitReader* br, int need) noexcept nogil:
    while br.bits_in_acc < need and br.byte_pos < br.length:
        br.acc |= (<uint64_t>br.data[br.byte_pos]) << (56 - br.bits_in_acc)
        br.byte_pos += 1
        br.bits_in_acc += 8


cdef inline uint32_t _rlgr_br_read_bits(_CRLGRBitReader* br, int n) noexcept nogil:
    if n <= 0:
        return 0
    if br.bits_in_acc < n:
        _rlgr_br_fill(br, n)
        if br.bits_in_acc < n:
            br.bits_in_acc = 0
            return 0
    cdef uint32_t val = <uint32_t>(br.acc >> (64 - n))
    br.acc <<= n
    br.bits_in_acc -= n
    return val


cdef inline uint32_t _rlgr_br_count0(_CRLGRBitReader* br) noexcept nogil:
    cdef uint32_t count = 0
    cdef int lz, consume
    while True:
        if br.bits_in_acc < 56 and br.byte_pos < br.length:
            _rlgr_br_fill(br, 56)
        if br.bits_in_acc == 0:
            return count
        lz = __builtin_clzll(br.acc) if br.acc != 0 else 64
        if lz >= br.bits_in_acc:
            count += <uint32_t>br.bits_in_acc
            br.acc = 0
            br.bits_in_acc = 0
            continue
        count += <uint32_t>lz
        consume = lz + 1
        br.acc <<= consume
        br.bits_in_acc -= consume
        return count


cdef inline uint32_t _rlgr_br_count1(_CRLGRBitReader* br) noexcept nogil:
    cdef uint32_t count = 0
    cdef int lo, consume
    cdef uint64_t inv
    while True:
        if br.bits_in_acc < 56 and br.byte_pos < br.length:
            _rlgr_br_fill(br, 56)
        if br.bits_in_acc == 0:
            return count
        inv = ~br.acc
        lo = __builtin_clzll(inv) if inv != 0 else 64
        if lo >= br.bits_in_acc:
            count += <uint32_t>br.bits_in_acc
            br.acc = 0
            br.bits_in_acc = 0
            continue
        count += <uint32_t>lo
        consume = lo + 1
        br.acc <<= consume
        br.bits_in_acc -= consume
        return count


cdef void _c_rlgr1_decode(const uint8_t *data, int data_len, int16_t *output, int output_size) noexcept nogil:
    if data == NULL or data_len <= 0:
        memset(output, 0, output_size * sizeof(int16_t))
        return

    memset(output, 0, output_size * sizeof(int16_t))

    cdef _CRLGRBitReader br
    _rlgr_br_init(&br, data, data_len)

    cdef int cnt = 0
    cdef uint32_t k = 1
    cdef uint32_t kp = 8
    cdef uint32_t kr = 1
    cdef uint32_t krp = 8
    cdef uint32_t vk, vk2, run, sign_bit, code, i
    cdef int16_t mag
    cdef int run_end

    while _rlgr_br_remaining(&br) > 0 and cnt < output_size:
        if k > 0:
            vk = _rlgr_br_count0(&br)

            run = 0
            for i in range(vk):
                run += (1 << k)
                kp += _UPGR
                if kp > _KPMAX: kp = _KPMAX
                k = kp >> _LSGR

            if k > 0:
                run += _rlgr_br_read_bits(&br, k)

            sign_bit = _rlgr_br_read_bits(&br, 1)

            vk2 = _rlgr_br_count1(&br)

            code = 0
            if kr > 0:
                code = _rlgr_br_read_bits(&br, kr)
            code |= (vk2 << kr)

            if vk2 == 0:
                if krp > 2: krp -= 2
                else: krp = 0
                kr = krp >> _LSGR
            elif vk2 != 1:
                krp += vk2
                if krp > _KPMAX: krp = _KPMAX
                kr = krp >> _LSGR

            if kp > _DNGR: kp -= _DNGR
            else: kp = 0
            k = kp >> _LSGR

            mag = <int16_t>(code + 1)
            if sign_bit != 0:
                mag = -mag

            run_end = cnt + <int>run
            if run_end > output_size: run_end = output_size
            cnt = run_end
            if cnt < output_size:
                output[cnt] = mag
                cnt += 1
        else:
            vk = _rlgr_br_count1(&br)

            code = 0
            if kr > 0:
                code = _rlgr_br_read_bits(&br, kr)
            code |= (vk << kr)

            if vk == 0:
                if krp > 2: krp -= 2
                else: krp = 0
                kr = krp >> _LSGR
            elif vk != 1:
                krp += vk
                if krp > _KPMAX: krp = _KPMAX
                kr = krp >> _LSGR

            if code == 0:
                kp += _UQGR
                if kp > _KPMAX: kp = _KPMAX
                k = kp >> _LSGR
                if cnt < output_size:
                    cnt += 1
            else:
                if kp > _DQGR: kp -= _DQGR
                else: kp = 0
                k = kp >> _LSGR

                if code & 1:
                    mag = -<int16_t>((code + 1) >> 1)
                else:
                    mag = <int16_t>(code >> 1)

                if cnt < output_size:
                    output[cnt] = mag
                    cnt += 1


# ---------------------------------------------------------------
# Inverse 2D DWT (3 levels) — C level nogil
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
    """One level of inverse 2D DWT in pure C, ported from grdp rfxIDWT2DLevel."""
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
# Optimized Color Transform and Surface Blit — 32-bit packed writes
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
    cdef uint32_t* dst_u32

    for row in range(tile_h):
        dst_row = output + (tile_y + row) * dst_stride + tile_x * 4
        dst_u32 = <uint32_t*>dst_row
        y_idx = row * 64

        for col in range(tile_w):
            y_val = <int>y_coeffs[y_idx + col]
            cb_val = <int>cb_coeffs[y_idx + col]
            cr_val = <int>cr_coeffs[y_idx + col]

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

            # 32-bit direct store (BGRA)
            dst_u32[col] = (<uint32_t>b) | ((<uint32_t>g) << 8) | ((<uint32_t>r) << 16) | 0xFF000000U


# ---------------------------------------------------------------
# SRL & RAW BitStream for Progressive Upgrades — C level nogil
# ---------------------------------------------------------------

cdef struct _CBitStream:
    const uint8_t* data
    int pos
    int length
    uint64_t bits
    int bit_len


cdef inline void _bs_fill(_CBitStream* bs) noexcept nogil:
    while bs.bit_len <= 56 and bs.pos < bs.length:
        bs.bits |= (<uint64_t>bs.data[bs.pos]) << (56 - bs.bit_len)
        bs.pos += 1
        bs.bit_len += 8


cdef inline void _bs_init(_CBitStream* bs, const uint8_t* data, int length) noexcept nogil:
    bs.data = data
    bs.pos = 0
    bs.length = length
    bs.bits = 0
    bs.bit_len = 0
    _bs_fill(bs)


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


cdef void _c_upgrade_component_subbands(const uint8_t* srl_data, int srl_len,
                                        const uint8_t* raw_data, int raw_len,
                                        CRfxQuant shift_shifts, CRfxQuant num_bits_shifts,
                                        int16_t* current, int16_t* sign,
                                        bint extrapolate) noexcept nogil:
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
# C Component Decode & Upgrade (Zero-allocation)
# ---------------------------------------------------------------

cdef void _c_decode_component(
    const uint8_t* data, int data_len,
    CRfxQuant shift,
    const int16_t* prev_dequant,
    bint is_diff,
    bint extrapolate,
    int16_t* out_pixels,
    int16_t* out_dequant,
    int16_t* out_raw_sign
) noexcept nogil:
    _c_rlgr1_decode(data, data_len, out_pixels, 4096)
    memcpy(out_raw_sign, out_pixels, 4096 * sizeof(int16_t))

    cdef int i, acc, s_idx, cnt, pos
    cdef int sub_shifts[10]
    sub_shifts[0] = shift.HL1; sub_shifts[1] = shift.LH1; sub_shifts[2] = shift.HH1
    sub_shifts[3] = shift.HL2; sub_shifts[4] = shift.LH2; sub_shifts[5] = shift.HH2
    sub_shifts[6] = shift.HL3; sub_shifts[7] = shift.LH3; sub_shifts[8] = shift.HH3
    sub_shifts[9] = shift.LL3

    if not extrapolate:
        acc = 0
        for i in range(4032, 4096):
            acc += out_pixels[i]
            out_pixels[i] = <int16_t>acc

        pos = 0
        for s_idx in range(10):
            cnt = C_SUBBAND_SIZES[s_idx]
            if sub_shifts[s_idx] > 0:
                for i in range(cnt):
                    out_pixels[pos + i] = <int16_t>(out_pixels[pos + i] << sub_shifts[s_idx])
            pos += cnt
    else:
        acc = 0
        for i in range(4015, 4096):
            acc += out_pixels[i]
            out_pixels[i] = <int16_t>acc

        pos = 0
        for s_idx in range(10):
            cnt = C_SUBBAND_SIZES_EXT[s_idx]
            if sub_shifts[s_idx] > 0:
                for i in range(cnt):
                    out_pixels[pos + i] = <int16_t>(out_pixels[pos + i] << sub_shifts[s_idx])
            pos += cnt

    if is_diff and prev_dequant != NULL:
        for i in range(4096):
            out_pixels[i] = _clampi16(<int>out_pixels[i] + <int>prev_dequant[i])

    memcpy(out_dequant, out_pixels, 4096 * sizeof(int16_t))
    if not extrapolate:
        _cy_inverse_dwt_2d(out_pixels)
    else:
        _cy_inverse_dwt_extrapolate(out_pixels)


cdef void _c_upgrade_component(
    const uint8_t* srl_data, int srl_len,
    const uint8_t* raw_data, int raw_len,
    CRfxQuant shift,
    CRfxQuant num_bits,
    int16_t* current,
    int16_t* sign,
    bint extrapolate,
    int16_t* out_pixels
) noexcept nogil:
    _c_upgrade_component_subbands(srl_data, srl_len, raw_data, raw_len,
                                  shift, num_bits, current, sign, extrapolate)
    memcpy(out_pixels, current, 4096 * sizeof(int16_t))
    if not extrapolate:
        _cy_inverse_dwt_2d(out_pixels)
    else:
        _cy_inverse_dwt_extrapolate(out_pixels)


# ---------------------------------------------------------------
# Tile State Cache Structure
# ---------------------------------------------------------------

DEF MAX_TILE_GRID = 16384  # Up to 128x128 tiles (8192x8192 screen)

cdef struct CRfxTileState:
    int16_t y[4096]
    int16_t cb[4096]
    int16_t cr[4096]
    int16_t sign_y[4096]
    int16_t sign_cb[4096]
    int16_t sign_cr[4096]
    CRfxQuant y_bit_pos
    CRfxQuant cb_bit_pos
    CRfxQuant cr_bit_pos
    bint valid


# ---------------------------------------------------------------
# Progressive stream parser & tile decoder (Cython class)
# ---------------------------------------------------------------

cdef class RfxProgressiveDecoder:
    """Decodes RFX Progressive codec data from RDPGFX WIRE_TO_SURFACE_2."""
    cdef CRfxTileState** _tiles

    def __cinit__(self):
        self._tiles = <CRfxTileState**>calloc(MAX_TILE_GRID, sizeof(CRfxTileState*))

    def __dealloc__(self):
        cdef int i
        if self._tiles != NULL:
            for i in range(MAX_TILE_GRID):
                if self._tiles[i] != NULL:
                    free(self._tiles[i])
            free(self._tiles)
            self._tiles = NULL

    cdef inline CRfxTileState* _get_tile_state(self, int x_idx, int y_idx) noexcept nogil:
        cdef int idx = ((y_idx & 0x7F) << 7) | (x_idx & 0x7F)
        if self._tiles[idx] == NULL:
            self._tiles[idx] = <CRfxTileState*>calloc(1, sizeof(CRfxTileState))
        return self._tiles[idx]

    def reset(self):
        """Reset tile state (called on DELETEENCODINGCONTEXT / RESET_GRAPHICS)."""
        cdef int i
        if self._tiles != NULL:
            for i in range(MAX_TILE_GRID):
                if self._tiles[i] != NULL:
                    self._tiles[i].valid = 0

    def decode(self, data, surf_data, int width, int height):
        """Decode progressive bitmap stream, render tiles onto surf_data (BGRA buffer).
        Returns list of (x, y, w, h) bounding rectangles."""
        if not data or width <= 0 or height <= 0:
            return []

        cdef bytes b_data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        cdef const uint8_t* p_data = <const uint8_t*>b_data
        cdef int data_len = len(b_data)

        cdef uint8_t[:] surf_mv
        if isinstance(surf_data, (bytes, bytearray, memoryview, np.ndarray)):
            surf_mv = np.frombuffer(surf_data, dtype=np.uint8)
        else:
            surf_mv = surf_data
        cdef uint8_t* p_surf = &surf_mv[0]

        cdef int offset = 0
        cdef uint16_t block_type
        cdef uint32_t block_len
        rects = []

        while offset + 6 <= data_len:
            block_type = (<const uint16_t*>(p_data + offset))[0]
            block_len = (<const uint32_t*>(p_data + offset + 2))[0]
            if block_len < 6 or offset + block_len > data_len:
                break

            if block_type == PROG_WBT_REGION:
                region_rects = self._decode_region_c(p_data + offset + 6, block_len - 6,
                                                     p_surf, width, height)
                rects.extend(region_rects)

            offset += block_len

        return rects

    cdef object _decode_region_c(self, const uint8_t* data, int data_len,
                                 uint8_t* p_surf, int width, int height):
        if data_len < 12:
            return []

        cdef uint16_t num_rects = (<const uint16_t*>(data + 1))[0]
        cdef uint8_t num_quant = data[3]
        cdef uint8_t num_prog_quant = data[4]
        cdef uint8_t flags = data[5]
        cdef uint16_t num_tiles = (<const uint16_t*>(data + 6))[0]
        cdef bint extrapolate = (flags & 0x01) != 0

        cdef int offset = 12
        cdef uint16_t rx, ry, rw, rh, tx, ty
        cdef int i

        rects = []
        for i in range(num_rects):
            if offset + 8 > data_len:
                return []
            rx = (<const uint16_t*>(data + offset))[0]
            ry = (<const uint16_t*>(data + offset + 2))[0]
            rw = (<const uint16_t*>(data + offset + 4))[0]
            rh = (<const uint16_t*>(data + offset + 6))[0]
            rects.append((rx, ry, rw, rh))
            offset += 8

        cdef CRfxQuant quants[64]
        cdef int q_count = num_quant if num_quant < 64 else 64
        for i in range(q_count):
            if offset + 5 > data_len:
                return []
            quants[i] = _c_parse_quant(data + offset)
            offset += 5

        cdef CRfxProgQuant prog_quants[64]
        cdef int pq_count = num_prog_quant if num_prog_quant < 64 else 64
        for i in range(pq_count):
            if offset + 16 > data_len:
                return []
            prog_quants[i] = _c_parse_prog_quant(data + offset)
            offset += 16

        tile_positions = []
        cdef uint16_t tile_type
        cdef uint32_t tile_len

        for i in range(num_tiles):
            if offset + 6 > data_len:
                break
            tile_type = (<const uint16_t*>(data + offset))[0]
            tile_len = (<const uint32_t*>(data + offset + 2))[0]
            if tile_len < 6 or offset + tile_len > data_len:
                break

            if tile_len >= 13:
                tx = (<const uint16_t*>(data + offset + 9))[0]
                ty = (<const uint16_t*>(data + offset + 11))[0]
                tile_positions.append((tx, ty))

            if tile_type == PROG_WBT_TILE_SIMPLE:
                self._c_tile_simple(data + offset + 6, tile_len - 6,
                                    quants, q_count, p_surf, width, height, extrapolate)
            elif tile_type == PROG_WBT_TILE_FIRST:
                self._c_tile_first(data + offset + 6, tile_len - 6,
                                   quants, q_count, prog_quants, pq_count,
                                   p_surf, width, height, extrapolate)
            elif tile_type == PROG_WBT_TILE_UPGRADE:
                self._c_tile_upgrade(data + offset + 6, tile_len - 6,
                                     quants, q_count, prog_quants, pq_count,
                                     p_surf, width, height, extrapolate)

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

    cdef void _c_tile_simple(self, const uint8_t* data, int data_len,
                             const CRfxQuant* quants, int num_quant,
                             uint8_t* output, int out_w, int out_h, bint extrapolate) noexcept nogil:
        if data_len < 16:
            return
        cdef uint8_t q_idx_y = data[0]
        cdef uint8_t q_idx_cb = data[1]
        cdef uint8_t q_idx_cr = data[2]
        cdef uint16_t x_idx = (<const uint16_t*>(data + 3))[0]
        cdef uint16_t y_idx = (<const uint16_t*>(data + 5))[0]
        cdef uint8_t flags = data[7]
        cdef uint16_t y_len = (<const uint16_t*>(data + 8))[0]
        cdef uint16_t cb_len = (<const uint16_t*>(data + 10))[0]
        cdef uint16_t cr_len = (<const uint16_t*>(data + 12))[0]
        cdef bint is_diff = (flags & 0x01) != 0

        cdef int off = 16
        cdef const uint8_t* y_data = (data + off) if (y_len > 0 and off + y_len <= data_len) else NULL
        off += y_len
        cdef const uint8_t* cb_data = (data + off) if (cb_len > 0 and off + cb_len <= data_len) else NULL
        off += cb_len
        cdef const uint8_t* cr_data = (data + off) if (cr_len > 0 and off + cr_len <= data_len) else NULL

        cdef CRfxQuant q_y = quants[q_idx_y] if q_idx_y < num_quant else _c_default_quant()
        cdef CRfxQuant q_cb = quants[q_idx_cb] if q_idx_cb < num_quant else _c_default_quant()
        cdef CRfxQuant q_cr = quants[q_idx_cr] if q_idx_cr < num_quant else _c_default_quant()

        cdef CRfxQuant shift_y = _c_prog_quant_shift(q_y, q_y, 0)
        cdef CRfxQuant shift_cb = _c_prog_quant_shift(q_cb, q_cb, 0)
        cdef CRfxQuant shift_cr = _c_prog_quant_shift(q_cr, q_cr, 0)

        cdef CRfxTileState* state = self._get_tile_state(x_idx, y_idx)
        cdef const int16_t* prev_y = state.y if (is_diff and state.valid) else NULL
        cdef const int16_t* prev_cb = state.cb if (is_diff and state.valid) else NULL
        cdef const int16_t* prev_cr = state.cr if (is_diff and state.valid) else NULL

        cdef int16_t y_pixels[4096]
        cdef int16_t cb_pixels[4096]
        cdef int16_t cr_pixels[4096]

        _c_decode_component(y_data, y_len, shift_y, prev_y, is_diff, extrapolate,
                            y_pixels, state.y, state.sign_y)
        _c_decode_component(cb_data, cb_len, shift_cb, prev_cb, is_diff, extrapolate,
                            cb_pixels, state.cb, state.sign_cb)
        _c_decode_component(cr_data, cr_len, shift_cr, prev_cr, is_diff, extrapolate,
                            cr_pixels, state.cr, state.sign_cr)

        _cy_place_tile_abs(y_pixels, cb_pixels, cr_pixels,
                           <int>x_idx * C_RFX_TILE_SIZE, <int>y_idx * C_RFX_TILE_SIZE,
                           output, out_w, out_h)

        state.y_bit_pos = q_y
        state.cb_bit_pos = q_cb
        state.cr_bit_pos = q_cr
        state.valid = 1

    cdef void _c_tile_first(self, const uint8_t* data, int data_len,
                            const CRfxQuant* quants, int num_quant,
                            const CRfxProgQuant* prog_quants, int num_prog_quant,
                            uint8_t* output, int out_w, int out_h, bint extrapolate) noexcept nogil:
        if data_len < 17:
            return
        cdef uint8_t q_idx_y = data[0]
        cdef uint8_t q_idx_cb = data[1]
        cdef uint8_t q_idx_cr = data[2]
        cdef uint16_t x_idx = (<const uint16_t*>(data + 3))[0]
        cdef uint16_t y_idx = (<const uint16_t*>(data + 5))[0]
        cdef uint8_t flags = data[7]
        cdef uint8_t quality = data[8]
        cdef uint16_t y_len = (<const uint16_t*>(data + 9))[0]
        cdef uint16_t cb_len = (<const uint16_t*>(data + 11))[0]
        cdef uint16_t cr_len = (<const uint16_t*>(data + 13))[0]
        cdef bint is_diff = (flags & 0x01) != 0

        cdef int off = 17
        cdef const uint8_t* y_data = (data + off) if (y_len > 0 and off + y_len <= data_len) else NULL
        off += y_len
        cdef const uint8_t* cb_data = (data + off) if (cb_len > 0 and off + cb_len <= data_len) else NULL
        off += cb_len
        cdef const uint8_t* cr_data = (data + off) if (cr_len > 0 and off + cr_len <= data_len) else NULL

        cdef CRfxQuant q_y = quants[q_idx_y] if q_idx_y < num_quant else _c_default_quant()
        cdef CRfxQuant q_cb = quants[q_idx_cb] if q_idx_cb < num_quant else _c_default_quant()
        cdef CRfxQuant q_cr = quants[q_idx_cr] if q_idx_cr < num_quant else _c_default_quant()

        cdef bint has_pq = quality < num_prog_quant
        cdef CRfxQuant pq_y = prog_quants[quality].y if has_pq else _c_default_quant()
        cdef CRfxQuant pq_cb = prog_quants[quality].cb if has_pq else _c_default_quant()
        cdef CRfxQuant pq_cr = prog_quants[quality].cr if has_pq else _c_default_quant()

        cdef CRfxQuant shift_y = _c_prog_quant_shift(q_y, pq_y, has_pq)
        cdef CRfxQuant shift_cb = _c_prog_quant_shift(q_cb, pq_cb, has_pq)
        cdef CRfxQuant shift_cr = _c_prog_quant_shift(q_cr, pq_cr, has_pq)

        cdef CRfxQuant bit_pos_y = _c_quant_add(q_y, pq_y) if has_pq else q_y
        cdef CRfxQuant bit_pos_cb = _c_quant_add(q_cb, pq_cb) if has_pq else q_cb
        cdef CRfxQuant bit_pos_cr = _c_quant_add(q_cr, pq_cr) if has_pq else q_cr

        cdef CRfxTileState* state = self._get_tile_state(x_idx, y_idx)
        cdef const int16_t* prev_y = state.y if (is_diff and state.valid) else NULL
        cdef const int16_t* prev_cb = state.cb if (is_diff and state.valid) else NULL
        cdef const int16_t* prev_cr = state.cr if (is_diff and state.valid) else NULL

        cdef int16_t y_pixels[4096]
        cdef int16_t cb_pixels[4096]
        cdef int16_t cr_pixels[4096]

        _c_decode_component(y_data, y_len, shift_y, prev_y, is_diff, extrapolate,
                            y_pixels, state.y, state.sign_y)
        _c_decode_component(cb_data, cb_len, shift_cb, prev_cb, is_diff, extrapolate,
                            cb_pixels, state.cb, state.sign_cb)
        _c_decode_component(cr_data, cr_len, shift_cr, prev_cr, is_diff, extrapolate,
                            cr_pixels, state.cr, state.sign_cr)

        _cy_place_tile_abs(y_pixels, cb_pixels, cr_pixels,
                           <int>x_idx * C_RFX_TILE_SIZE, <int>y_idx * C_RFX_TILE_SIZE,
                           output, out_w, out_h)

        state.y_bit_pos = bit_pos_y
        state.cb_bit_pos = bit_pos_cb
        state.cr_bit_pos = bit_pos_cr
        state.valid = 1

    cdef void _c_tile_upgrade(self, const uint8_t* data, int data_len,
                              const CRfxQuant* quants, int num_quant,
                              const CRfxProgQuant* prog_quants, int num_prog_quant,
                              uint8_t* output, int out_w, int out_h, bint extrapolate) noexcept nogil:
        if data_len < 20:
            return
        cdef uint8_t q_idx_y = data[0]
        cdef uint8_t q_idx_cb = data[1]
        cdef uint8_t q_idx_cr = data[2]
        cdef uint16_t x_idx = (<const uint16_t*>(data + 3))[0]
        cdef uint16_t y_idx = (<const uint16_t*>(data + 5))[0]
        cdef uint8_t quality = data[7]
        cdef uint16_t y_srl_len = (<const uint16_t*>(data + 8))[0]
        cdef uint16_t y_raw_len = (<const uint16_t*>(data + 10))[0]
        cdef uint16_t cb_srl_len = (<const uint16_t*>(data + 12))[0]
        cdef uint16_t cb_raw_len = (<const uint16_t*>(data + 14))[0]
        cdef uint16_t cr_srl_len = (<const uint16_t*>(data + 16))[0]
        cdef uint16_t cr_raw_len = (<const uint16_t*>(data + 18))[0]

        cdef int off = 20
        cdef const uint8_t* y_srl = (data + off) if (y_srl_len > 0 and off + y_srl_len <= data_len) else NULL
        off += y_srl_len
        cdef const uint8_t* y_raw = (data + off) if (y_raw_len > 0 and off + y_raw_len <= data_len) else NULL
        off += y_raw_len
        cdef const uint8_t* cb_srl = (data + off) if (cb_srl_len > 0 and off + cb_srl_len <= data_len) else NULL
        off += cb_srl_len
        cdef const uint8_t* cb_raw = (data + off) if (cb_raw_len > 0 and off + cb_raw_len <= data_len) else NULL
        off += cb_raw_len
        cdef const uint8_t* cr_srl = (data + off) if (cr_srl_len > 0 and off + cr_srl_len <= data_len) else NULL
        off += cr_srl_len
        cdef const uint8_t* cr_raw = (data + off) if (cr_raw_len > 0 and off + cr_raw_len <= data_len) else NULL

        cdef CRfxTileState* state = self._get_tile_state(x_idx, y_idx)
        if not state.valid:
            return

        cdef CRfxQuant q_y = quants[q_idx_y] if q_idx_y < num_quant else _c_default_quant()
        cdef CRfxQuant q_cb = quants[q_idx_cb] if q_idx_cb < num_quant else _c_default_quant()
        cdef CRfxQuant q_cr = quants[q_idx_cr] if q_idx_cr < num_quant else _c_default_quant()

        cdef bint has_pq = quality < num_prog_quant
        cdef CRfxQuant pq_y = prog_quants[quality].y if has_pq else _c_default_quant()
        cdef CRfxQuant pq_cb = prog_quants[quality].cb if has_pq else _c_default_quant()
        cdef CRfxQuant pq_cr = prog_quants[quality].cr if has_pq else _c_default_quant()

        cdef CRfxQuant new_y_bit_pos = _c_quant_add(q_y, pq_y) if has_pq else q_y
        cdef CRfxQuant new_cb_bit_pos = _c_quant_add(q_cb, pq_cb) if has_pq else q_cb
        cdef CRfxQuant new_cr_bit_pos = _c_quant_add(q_cr, pq_cr) if has_pq else q_cr

        cdef CRfxQuant num_bits_y = _c_quant_sub(state.y_bit_pos, new_y_bit_pos)
        cdef CRfxQuant num_bits_cb = _c_quant_sub(state.cb_bit_pos, new_cb_bit_pos)
        cdef CRfxQuant num_bits_cr = _c_quant_sub(state.cr_bit_pos, new_cr_bit_pos)

        cdef CRfxQuant shift_y = _c_prog_quant_shift(new_y_bit_pos, new_y_bit_pos, 0)
        cdef CRfxQuant shift_cb = _c_prog_quant_shift(new_cb_bit_pos, new_cb_bit_pos, 0)
        cdef CRfxQuant shift_cr = _c_prog_quant_shift(new_cr_bit_pos, new_cr_bit_pos, 0)

        state.y_bit_pos = new_y_bit_pos
        state.cb_bit_pos = new_cb_bit_pos
        state.cr_bit_pos = new_cr_bit_pos

        cdef int16_t y_pixels[4096]
        cdef int16_t cb_pixels[4096]
        cdef int16_t cr_pixels[4096]

        _c_upgrade_component(y_srl, y_srl_len, y_raw, y_raw_len, shift_y, num_bits_y,
                             state.y, state.sign_y, extrapolate, y_pixels)
        _c_upgrade_component(cb_srl, cb_srl_len, cb_raw, cb_raw_len, shift_cb, num_bits_cb,
                             state.cb, state.sign_cb, extrapolate, cb_pixels)
        _c_upgrade_component(cr_srl, cr_srl_len, cr_raw, cr_raw_len, shift_cr, num_bits_cr,
                             state.cr, state.sign_cr, extrapolate, cr_pixels)

        _cy_place_tile_abs(y_pixels, cb_pixels, cr_pixels,
                           <int>x_idx * C_RFX_TILE_SIZE, <int>y_idx * C_RFX_TILE_SIZE,
                           output, out_w, out_h)


# ---------------------------------------------------------------
# Python Wrapper functions for external compatibility
# ---------------------------------------------------------------

def _inverse_dwt_2d(np.ndarray[int16_t, ndim=1] coeffs):
    _cy_inverse_dwt_2d(<int16_t*>coeffs.data)


def _inverse_dwt_extrapolate(np.ndarray[int16_t, ndim=1] buffer):
    _cy_inverse_dwt_extrapolate(<int16_t*>buffer.data)


def _place_tile_abs(y_coeffs, cb_coeffs, cr_coeffs,
                    int tile_x, int tile_y,
                    output, int out_w, int out_h):
    if isinstance(y_coeffs, (tuple, list)): y_coeffs = y_coeffs[0]
    if isinstance(cb_coeffs, (tuple, list)): cb_coeffs = cb_coeffs[0]
    if isinstance(cr_coeffs, (tuple, list)): cr_coeffs = cr_coeffs[0]

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


def _decode_component(data, quant_obj, prev_dequant=None, is_diff=False, extrapolate=False):
    cdef CRfxQuant q
    if isinstance(quant_obj, RfxQuant):
        q.LL3 = quant_obj.LL3; q.LH3 = quant_obj.LH3; q.HL3 = quant_obj.HL3; q.HH3 = quant_obj.HH3
        q.LH2 = quant_obj.LH2; q.HL2 = quant_obj.HL2; q.HH2 = quant_obj.HH2
        q.LH1 = quant_obj.LH1; q.HL1 = quant_obj.HL1; q.HH1 = quant_obj.HH1
    else:
        q = _c_default_quant()

    cdef CRfxQuant shift = _c_prog_quant_shift(q, q, 0)
    cdef np.ndarray[int16_t, ndim=1] out_pixels = np.empty(4096, dtype=np.int16)
    cdef np.ndarray[int16_t, ndim=1] out_dequant = np.empty(4096, dtype=np.int16)
    cdef np.ndarray[int16_t, ndim=1] out_raw_sign = np.empty(4096, dtype=np.int16)

    cdef bytes b_data = bytes(data) if data else b''
    cdef const uint8_t* p_data = <const uint8_t*>b_data if b_data else NULL
    cdef int dlen = len(b_data)

    cdef const int16_t* p_prev = NULL
    cdef np.ndarray[int16_t, ndim=1] prev_arr
    if prev_dequant is not None:
        prev_arr = prev_dequant
        p_prev = <const int16_t*>prev_arr.data

    _c_decode_component(p_data, dlen, shift, p_prev, 1 if is_diff else 0, 1 if extrapolate else 0,
                        <int16_t*>out_pixels.data,
                        <int16_t*>out_dequant.data,
                        <int16_t*>out_raw_sign.data)
    return out_pixels, out_dequant, out_raw_sign
