#
# RFX Progressive Codec decoder (MS-RDPRFX / MS-RDPEGFX 2.2.4).
# Handles RDPGFX_CODECID_CAPROGRESSIVE (0x0009) in WIRE_TO_SURFACE_PDU_2.
# Ported from grdp Go implementation (commit 6d4735c).
#

import struct
import functools
import numpy as np
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

# Pre-built index array for full 64×64 tiles — avoids np.arange() per tile call.
_FULL_TILE_COEFF_IDX = np.arange(RFX_TILE_SIZE * RFX_TILE_SIZE, dtype=np.int32)

# Sub-band sizes in RLGR decode order:
# HL1(1024) LH1(1024) HH1(1024) HL2(256) LH2(256) HH2(256) HL3(64) LH3(64) HH3(64) LL3(64)
_SUBBAND_SIZES = [1024, 1024, 1024, 256, 256, 256, 64, 64, 64, 64]
# Sub-band sizes for extrapolated DWT (RFX_DWT_REDUCE_EXTRAPOLATE):
_SUBBAND_SIZES_EXT = [1023, 1023, 961, 272, 272, 256, 72, 72, 64, 81]


# ---------------------------------------------------------------
# RLGR1 Decoder
# ---------------------------------------------------------------

# _BitReader is kept here for use by _upgrade_component.
# The same class also lives in rlgr1_decode.py for use by the Python rlgr1_decode.
class _BitReader:
    __slots__ = ('_data', '_pos', '_total', '_accum', '_accum_bits')

    def __init__(self, data):
        self._data = bytes(data) if not isinstance(data, bytes) else data
        self._pos = 0
        self._total = len(data) * 8
        self._accum = 0       # bit accumulator (up to 64 bits)
        self._accum_bits = 0  # valid bits in accumulator

    def remaining(self):
        # _pos tracks bits loaded into accumulator; _accum_bits are loaded but unconsumed
        return self._total - self._pos + self._accum_bits

    def _refill(self, need):
        """Ensure at least `need` bits in the accumulator.
        Loads up to 8 bytes per call to minimize Python loop overhead."""
        data = self._data
        byte_pos = self._pos >> 3
        data_len = len(data)
        accum = self._accum
        accum_bits = self._accum_bits
        pos = self._pos
        # Load multiple bytes in a tight loop
        while accum_bits < need:
            if byte_pos < data_len:
                # Load up to 8 bytes at once (limited by available data)
                load = min(8, data_len - byte_pos, (64 - accum_bits) >> 3)
                if load <= 0:
                    load = 1
                for i in range(load):
                    accum = (accum << 8) | data[byte_pos]
                    byte_pos += 1
                accum_bits += load * 8
                pos += load * 8
            else:
                accum <<= 8
                accum_bits += 8
                pos += 8
        self._accum = accum
        self._accum_bits = accum_bits
        self._pos = pos

    def read_bit(self):
        if self._accum_bits < 1:
            self._refill(8)
        self._accum_bits -= 1
        return (self._accum >> self._accum_bits) & 1

    def read_bits(self, n):
        if n == 0:
            return 0
        if self._accum_bits < n:
            self._refill(n)
        self._accum_bits -= n
        val = (self._accum >> self._accum_bits) & ((1 << n) - 1)
        return val

    def count_leading_bits(self, target):
        """Count consecutive bits matching `target`. Scans multiple bits at
        a time within the accumulator to avoid per-bit Python overhead."""
        count = 0
        while self.remaining() > 0:
            if self._accum_bits < 8:
                self._refill(16)
            # Scan available bits in chunks — check 8 bits at a time
            while self._accum_bits >= 8:
                top8 = (self._accum >> (self._accum_bits - 8)) & 0xFF
                expected = 0xFF if target else 0x00
                if top8 == expected:
                    count += 8
                    self._accum_bits -= 8
                else:
                    # Scan remaining bits in this byte individually
                    for _ in range(8):
                        self._accum_bits -= 1
                        bit = (self._accum >> self._accum_bits) & 1
                        if bit == target:
                            count += 1
                        else:
                            return count
                    break
            else:
                continue
            break
        # Handle leftover bits (< 8) at end of stream
        while self.remaining() > 0 and self._accum_bits > 0:
            self._accum_bits -= 1
            bit = (self._accum >> self._accum_bits) & 1
            if bit == target:
                count += 1
            else:
                return count
        return count


# ---------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------

class RfxQuant:
    """TS_RFX_CODEC_QUANT: 5-byte packed quantization values per MS-RDPRFX 2.2.2.1.5."""
    __slots__ = ('LL3', 'LH3', 'HL3', 'HH3', 'LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1')

    def __init__(self, data):
        # byte 0: LL3 (low), HL3 (high)
        self.LL3 = data[0] & 0x0F
        self.HL3 = data[0] >> 4
        # byte 1: LH3 (low), HH3 (high)
        self.LH3 = data[1] & 0x0F
        self.HH3 = data[1] >> 4
        # byte 2: HL2 (low), LH2 (high)
        self.HL2 = data[2] & 0x0F
        self.LH2 = data[2] >> 4
        # byte 3: HH2 (low), HL1 (high)
        self.HH2 = data[3] & 0x0F
        self.HL1 = data[3] >> 4
        # byte 4: LH1 (low), HH1 (high)
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
# Inverse 2D DWT (3 levels)
# ---------------------------------------------------------------

def _idwt_2d_level(buf, n):
    """One level of inverse 2D DWT.
    buf contains [HL(n²)|LH(n²)|HH(n²)|LL(n²)] → (2n)×(2n) result.
    Uses MS-RDPRFX lifting scheme matching FreeRDP/grdp."""
    nn = n * n
    size = 2 * n

    # Extract subbands as int32 2D arrays
    hl = buf[0:nn].astype(np.int32).reshape(n, n)
    lh = buf[nn:2*nn].astype(np.int32).reshape(n, n)
    hh = buf[2*nn:3*nn].astype(np.int32).reshape(n, n)
    ll = buf[3*nn:4*nn].astype(np.int32).reshape(n, n)

    # --- Step 1: Horizontal IDWT (all rows at once) ---
    # Stack L-part and H-part: rows 0..n-1 use (ll,hl), rows n..2n-1 use (lh,hh)
    lo = np.vstack([ll, lh])   # (2n, n)
    hi = np.vstack([hl, hh])   # (2n, n)

    # Even positions (undo update step); compute in int32, then truncate to int16 (matching grdp)
    even_h = np.empty((2 * n, n), dtype=np.int32)
    even_h[:, 0] = lo[:, 0] - ((hi[:, 0] * 2 + 1) >> 1)
    even_h[:, 1:] = lo[:, 1:] - ((hi[:, :-1] + hi[:, 1:] + 1) >> 1)
    even_h_i16 = even_h.astype(np.int16)
    even_h_i32 = even_h_i16.astype(np.int32)  # sign-extended truncated values

    # Odd positions — both neighbours are read from int16 tmp (both truncated, matching grdp)
    odd_h = np.empty((2 * n, n), dtype=np.int32)
    odd_h[:, :-1] = (hi[:, :-1] << 1) + ((even_h_i32[:, :-1] + even_h_i32[:, 1:]) >> 1)
    odd_h[:, -1] = (hi[:, -1] << 1) + even_h_i32[:, -1]

    # Interleave even/odd into tmp as int16 (matching grdp's int16 tmp buffer)
    tmp = np.empty((2 * n, size), dtype=np.int16)
    tmp[:, 0::2] = even_h_i16
    tmp[:, 1::2] = odd_h.astype(np.int16)

    # --- Step 2: Vertical IDWT (all columns at once) ---
    lo_v = tmp[:n, :].astype(np.int32)    # L rows (sign-extended from int16)
    hi_v = tmp[n:, :].astype(np.int32)    # H rows (sign-extended from int16)

    # Even rows; compute in int32, then truncate to int16 (matching grdp)
    even_v = np.empty((n, size), dtype=np.int32)
    even_v[0, :] = lo_v[0, :] - ((hi_v[0, :] * 2 + 1) >> 1)
    even_v[1:, :] = lo_v[1:, :] - ((hi_v[:-1, :] + hi_v[1:, :] + 1) >> 1)
    even_v_i16 = even_v.astype(np.int16)
    even_v_i32 = even_v_i16.astype(np.int32)  # sign-extended truncated values

    # Odd rows — prevEven (even[row-1]) is truncated; current even[row] is not yet truncated
    # (matches grdp: prevEven = int32(buf[prev]) is truncated, even = local int32 is not)
    odd_v = np.empty((n, size), dtype=np.int32)
    odd_v[:-1, :] = (hi_v[:-1, :] << 1) + ((even_v_i32[:-1, :] + even_v[1:, :]) >> 1)
    odd_v[-1, :] = (hi_v[-1, :] << 1) + even_v_i32[-1, :]

    # Interleave even/odd into output as int16
    out = np.empty((size, size), dtype=np.int16)
    out[0::2, :] = even_v_i16
    out[1::2, :] = odd_v.astype(np.int16)

    buf[:size * size] = out.reshape(-1)


def _clampi16(val):
    if val < -32768:
        return -32768
    if val > 32767:
        return 32767
    return val


def _trunc_div2(x):
    """Integer division by 2 truncating towards zero, matching C/Go (x / 2)."""
    return (x // 2) if x >= 0 else -(-x // 2)


def _idwt_x(p_low, p_high, p_dst, n_low_step, n_high_step, n_dst_step, n_low_count, n_high_count, n_dst_count):
    for i in range(n_dst_count):
        p_l = i * n_low_step
        p_h = i * n_high_step
        p_x = i * n_dst_step

        H0 = int(p_high[p_h])
        p_h += 1
        L0 = int(p_low[p_l])
        p_l += 1

        X0 = _clampi16(L0 - H0)
        X2 = X0

        for j in range(n_high_count - 1):
            H1 = int(p_high[p_h])
            p_h += 1
            L0 = int(p_low[p_l])
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
                L0 = int(p_low[p_l])
                p_l += 1
                X0 = _clampi16(L0 - H0)
                p_dst[p_x] = X2
                p_dst[p_x + 1] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
                p_dst[p_x + 2] = X0
        else:
            L0 = int(p_low[p_l])
            p_l += 1
            X0 = _clampi16(L0 - _trunc_div2(H0))
            p_dst[p_x] = X2
            p_dst[p_x + 1] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
            p_dst[p_x + 2] = X0
            L0 = int(p_low[p_l])
            p_dst[p_x + 3] = _clampi16(_trunc_div2(X0 + L0))


def _idwt_y(p_low, p_high, p_dst, n_low_step, n_high_step, n_dst_step, n_low_count, n_high_count, n_dst_count):
    for i in range(n_dst_count):
        p_l = i
        p_h = i
        p_x = i

        H0 = int(p_high[p_h])
        p_h += n_high_step
        L0 = int(p_low[p_l])
        p_l += n_low_step

        X0 = _clampi16(L0 - H0)
        X2 = X0

        for j in range(n_high_count - 1):
            H1 = int(p_high[p_h])
            p_h += n_high_step
            L0 = int(p_low[p_l])
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
                L0 = int(p_low[p_l])
                p_l += n_low_step
                X0 = _clampi16(L0 - H0)
                p_dst[p_x] = X2
                p_x += n_dst_step
                p_dst[p_x] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
                p_x += n_dst_step
                p_dst[p_x] = X0
        else:
            L0 = int(p_low[p_l])
            p_l += n_low_step
            X0 = _clampi16(L0 - _trunc_div2(H0))
            p_dst[p_x] = X2
            p_x += n_dst_step
            p_dst[p_x] = _clampi16(_trunc_div2(X0 + X2) + 2 * H0)
            p_x += n_dst_step
            p_dst[p_x] = X0
            p_x += n_dst_step
            L0 = int(p_low[p_l])
            p_dst[p_x] = _clampi16(_trunc_div2(X0 + L0))


def _dwt_2d_decode_block(buffer, temp, level):
    n_band_l = (64 >> level) + 1
    if level == 1:
        n_band_h = (64 >> 1) - 1
    else:
        n_band_h = (64 + (1 << (level - 1))) >> level

    offset = 0
    hl = buffer[offset:offset + n_band_h * n_band_l]
    offset += n_band_h * n_band_l
    lh = buffer[offset:offset + n_band_l * n_band_h]
    offset += n_band_l * n_band_h
    hh = buffer[offset:offset + n_band_h * n_band_h]
    offset += n_band_h * n_band_h
    ll = buffer[offset:offset + n_band_l * n_band_l]

    n_dst_step = n_band_l + n_band_h

    temp_l = temp[0:n_band_l * n_dst_step]
    temp_h = temp[n_band_l * n_dst_step:(n_band_l + n_band_h) * n_dst_step]

    # horizontal (LL + HL -> L)
    _idwt_x(ll, hl, temp_l, n_band_l, n_band_h, n_dst_step, n_band_l, n_band_h, n_band_l)

    # horizontal (LH + HH -> H)
    _idwt_x(lh, hh, temp_h, n_band_l, n_band_h, n_dst_step, n_band_l, n_band_h, n_band_h)

    # vertical (L + H -> buffer)
    _idwt_y(temp_l, temp_h, buffer, n_dst_step, n_dst_step, n_dst_step, n_band_l, n_band_h, n_band_l + n_band_h)


def _inverse_dwt_extrapolate(buffer):
    temp = np.zeros(4096, dtype=np.int16)
    _dwt_2d_decode_block(buffer[3807:], temp, 3)
    _dwt_2d_decode_block(buffer[3007:], temp, 2)
    _dwt_2d_decode_block(buffer[0:], temp, 1)


def _inverse_dwt_2d(coeffs):
    """3-level inverse 2D DWT in-place."""
    _idwt_2d_level(coeffs[3840:], 8)   # Level 3: 8×8 → 16×16
    _idwt_2d_level(coeffs[3072:], 16)  # Level 2: 16×16 → 32×32
    _idwt_2d_level(coeffs[0:], 32)     # Level 1: 32×32 → 64×64


# ---------------------------------------------------------------
# Component decoding
# ---------------------------------------------------------------

def _decode_component(data, shift, prev_dequant=None, is_diff=False, extrapolate=False):
    """Decode one color component (Y/Cb/Cr) for a 64×64 tile."""
    TILE_PIXELS = RFX_TILE_SIZE * RFX_TILE_SIZE  # 4096

    if data:
        coeffs = rlgr1_decode(data, TILE_PIXELS)
        raw_sign = coeffs.copy()
    else:
        coeffs = np.zeros(TILE_PIXELS, dtype=np.int16)
        raw_sign = np.zeros(TILE_PIXELS, dtype=np.int16)

    if not extrapolate:
        # Differential decode LL3 (positions 4032..4095)
        coeffs[4032:4096] = np.cumsum(coeffs[4032:4096], dtype=np.int16)
        shift_arr = np.repeat(np.array(shift.subband_shifts(), dtype=np.int16), _SUBBAND_SIZES)
        coeffs <<= shift_arr
    else:
        # Differential decode LL3 (positions 4015..4095)
        coeffs[4015:4096] = np.cumsum(coeffs[4015:4096], dtype=np.int16)
        shift_arr = np.repeat(np.array(shift.subband_shifts(), dtype=np.int16), _SUBBAND_SIZES_EXT)
        coeffs <<= shift_arr

    if is_diff and prev_dequant is not None:
        coeffs += prev_dequant

    dequant = coeffs.copy()
    work = coeffs.copy()
    if not extrapolate:
        _inverse_dwt_2d(work)
    else:
        _inverse_dwt_extrapolate(work)

    return work, dequant, raw_sign


# ---------------------------------------------------------------
# YCbCr → BGRA color conversion and tile placement
# ---------------------------------------------------------------

def _place_tile_abs(y_coeffs, cb_coeffs, cr_coeffs, tile_x, tile_y, output, out_w, out_h):
    """Convert YCbCr tile to BGRA at absolute pixel coordinates (tile_x, tile_y)."""
    tile_w = min(RFX_TILE_SIZE, out_w - tile_x)
    tile_h = min(RFX_TILE_SIZE, out_h - tile_y)
    if tile_w <= 0 or tile_h <= 0:
        return

    # Build flat index into 64-wide coefficient arrays
    if tile_w == RFX_TILE_SIZE and tile_h == RFX_TILE_SIZE:
        y_arr = y_coeffs.astype(np.int32)
        cb_arr = cb_coeffs.astype(np.int32)
        cr_arr = cr_coeffs.astype(np.int32)
    else:
        rows = np.arange(tile_h, dtype=np.int32)
        cols = np.arange(tile_w, dtype=np.int32)
        coeff_idx = (rows[:, np.newaxis] * RFX_TILE_SIZE + cols[np.newaxis, :]).ravel()
        y_arr = y_coeffs[coeff_idx].astype(np.int32)
        cb_arr = cb_coeffs[coeff_idx].astype(np.int32)
        cr_arr = cr_coeffs[coeff_idx].astype(np.int32)

    # ICT (BT.601) inverse colour transform
    y_scaled = (y_arr + 4096) << 16
    r = (cr_arr * 91916 + y_scaled) >> 21
    g = (y_scaled - cb_arr * 22527 - cr_arr * 46819) >> 21
    b = (cb_arr * 115992 + y_scaled) >> 21

    np.clip(r, 0, 255, out=r)
    np.clip(g, 0, 255, out=g)
    np.clip(b, 0, 255, out=b)

    # Assemble BGRA tile
    bgra = np.empty((tile_h, tile_w, 4), dtype=np.uint8)
    bgra_flat = bgra.reshape(-1, 4)
    bgra_flat[:, 0] = b
    bgra_flat[:, 1] = g
    bgra_flat[:, 2] = r
    bgra_flat[:, 3] = 0xFF

    # Write into output using numpy 2D view (avoids Python row loop)
    try:
        out_arr = np.frombuffer(output, dtype=np.uint8).reshape(out_h, out_w * 4)
        out_arr[tile_y:tile_y + tile_h, tile_x * 4:(tile_x + tile_w) * 4] = \
            bgra.reshape(tile_h, tile_w * 4)
    except (ValueError, IndexError):
        stride = tile_w * 4
        out_stride = out_w * 4
        out_mv = memoryview(output)
        bgra_bytes = bgra.tobytes()
        for row in range(tile_h):
            out_start = ((tile_y + row) * out_w + tile_x) * 4
            if out_start + stride <= len(output):
                out_mv[out_start:out_start + stride] = bgra_bytes[row * stride:(row + 1) * stride]


def _place_tile(y_coeffs, cb_coeffs, cr_coeffs, x_idx, y_idx, output, out_w, out_h):
    """Convert YCbCr tile to BGRA using tile grid indices."""
    _place_tile_abs(y_coeffs, cb_coeffs, cr_coeffs,
                    x_idx * RFX_TILE_SIZE, y_idx * RFX_TILE_SIZE,
                    output, out_w, out_h)


# ---------------------------------------------------------------
# SRL + RAW bit-level reader for progressive upgrades
# ---------------------------------------------------------------

class _BitStream:
    def __init__(self, data):
        self.data = data or b''
        self.pos = 0
        self.bits = 0
        self.bit_len = 0
        self._fill()

    def _fill(self):
        while self.bit_len <= 56 and self.pos < len(self.data):
            self.bits |= (self.data[self.pos] << (56 - self.bit_len))
            self.pos += 1
            self.bit_len += 8

    def read_bit(self):
        if self.bit_len < 1:
            self._fill()
            if self.bit_len < 1:
                return 0
        bit = (self.bits >> 63) & 1
        self.bits = (self.bits << 1) & 0xFFFFFFFFFFFFFFFF
        self.bit_len -= 1
        return bit

    def read_bits(self, n):
        if n <= 0:
            return 0
        if self.bit_len < n:
            self._fill()
            if self.bit_len < n:
                n = self.bit_len
                if n <= 0:
                    return 0
        val = (self.bits >> (64 - n)) & ((1 << n) - 1)
        self.bits = (self.bits << n) & 0xFFFFFFFFFFFFFFFF
        self.bit_len -= n
        return val


class _SRLState:
    def __init__(self, srl_data, raw_data):
        self.srl = _BitStream(srl_data)
        self.raw = _BitStream(raw_data)
        self.kp = 8
        self.nz = 0
        self.mode = False

    def read_srl(self, num_bits):
        if self.nz > 0:
            self.nz -= 1
            return 0
        k = self.kp // 8
        if not self.mode:
            bit = self.srl.read_bit()
            if bit == 0:
                self.nz = (1 << k) - 1
                self.kp += 4
                if self.kp > 80:
                    self.kp = 80
                return 0
            else:
                self.nz = 0
                self.mode = True
                if k > 0:
                    self.nz = self.srl.read_bits(k)
                if self.nz > 0:
                    self.nz -= 1
                    return 0
        self.mode = False
        sign = self.srl.read_bit()
        if self.kp < 6:
            self.kp = 0
        else:
            self.kp -= 6
        if num_bits == 1:
            return -1 if sign != 0 else 1
        mag = 1
        max_val = (1 << num_bits) - 1
        while mag < max_val:
            bit = self.srl.read_bit()
            if bit != 0:
                break
            mag += 1
        return -mag if sign != 0 else mag

    def upgrade_block(self, current, sign, length, shift, num_bits, is_ll):
        if num_bits == 0 or length > len(current) or length > len(sign):
            return
        if is_ll:
            for i in range(length):
                input_val = self.raw.read_bits(num_bits)
                current[i] += np.int16(input_val << shift)
            return
        for i in range(length):
            if sign[i] > 0:
                input_val = self.raw.read_bits(num_bits)
            elif sign[i] < 0:
                input_val = -self.raw.read_bits(num_bits)
            else:
                srl_val = self.read_srl(num_bits)
                sign[i] = srl_val
                input_val = srl_val
            current[i] += np.int16(input_val << shift)


def _upgrade_component(srl_data, raw_data, shift, num_bits, current, sign, extrapolate=False):
    """Apply progressive upgrade delta to stored coefficients."""
    state = _SRLState(srl_data, raw_data)
    cur = current
    sgn = sign
    if not extrapolate:
        state.upgrade_block(cur[0:1024], sgn[0:1024], 1024, shift.HL1, num_bits.HL1, False)
        state.upgrade_block(cur[1024:2048], sgn[1024:2048], 1024, shift.LH1, num_bits.LH1, False)
        state.upgrade_block(cur[2048:3072], sgn[2048:3072], 1024, shift.HH1, num_bits.HH1, False)
        state.upgrade_block(cur[3072:3328], sgn[3072:3328], 256, shift.HL2, num_bits.HL2, False)
        state.upgrade_block(cur[3328:3584], sgn[3328:3584], 256, shift.LH2, num_bits.LH2, False)
        state.upgrade_block(cur[3584:3840], sgn[3584:3840], 256, shift.HH2, num_bits.HH2, False)
        state.upgrade_block(cur[3840:3904], sgn[3840:3904], 64, shift.HL3, num_bits.HL3, False)
        state.upgrade_block(cur[3904:3968], sgn[3904:3968], 64, shift.LH3, num_bits.LH3, False)
        state.upgrade_block(cur[3968:4032], sgn[3968:4032], 64, shift.HH3, num_bits.HH3, False)
        state.upgrade_block(cur[4032:4096], sgn[4032:4096], 64, shift.LL3, num_bits.LL3, True)
    else:
        state.upgrade_block(cur[0:1023], sgn[0:1023], 1023, shift.HL1, num_bits.HL1, False)
        state.upgrade_block(cur[1023:2046], sgn[1023:2046], 1023, shift.LH1, num_bits.LH1, False)
        state.upgrade_block(cur[2046:3007], sgn[2046:3007], 961, shift.HH1, num_bits.HH1, False)
        state.upgrade_block(cur[3007:3279], sgn[3007:3279], 272, shift.HL2, num_bits.HL2, False)
        state.upgrade_block(cur[3279:3551], sgn[3279:3551], 272, shift.LH2, num_bits.LH2, False)
        state.upgrade_block(cur[3551:3807], sgn[3551:3807], 256, shift.HH2, num_bits.HH2, False)
        state.upgrade_block(cur[3807:3879], sgn[3807:3879], 72, shift.HL3, num_bits.HL3, False)
        state.upgrade_block(cur[3879:3951], sgn[3879:3951], 72, shift.LH3, num_bits.LH3, False)
        state.upgrade_block(cur[3951:4015], sgn[3951:4015], 64, shift.HH3, num_bits.HH3, False)
        state.upgrade_block(cur[4015:4096], sgn[4015:4096], 81, shift.LL3, num_bits.LL3, True)

    work = cur.copy()
    if not extrapolate:
        _inverse_dwt_2d(work)
    else:
        _inverse_dwt_extrapolate(work)
    return work


# ---------------------------------------------------------------
# Progressive stream parser
# ---------------------------------------------------------------

class RfxProgressiveDecoder:
    """Decodes RFX Progressive codec data from RDPGFX WIRE_TO_SURFACE_2."""

    def __init__(self):
        self._debug_tile_count = 0
        # Per-tile state: (x_idx, y_idx) -> dict with y/cb/cr raw coefficients and sign
        self._tileState = {}

    def reset(self):
        """Reset tile state (called on DELETEENCODINGCONTEXT)."""
        self._tileState.clear()
        self._debug_tile_count = 0

    def decode(self, data, surf_data, width, height):
        """Decode progressive bitmap stream, render tiles onto surf_data (BGRA buffer).
        Returns list of (x, y, w, h) bounding rectangles."""
        rects = []
        offset = 0

        while offset + 6 <= len(data):
            block_type, block_len = struct.unpack_from('<HI', data, offset)

            if block_len < 6 or offset + block_len > len(data):
                break

            block_data = data[offset + 6:offset + block_len]

            if block_type == PROG_WBT_REGION:
                rects.extend(self._decode_region(block_data, surf_data, width, height))
            elif block_type in (PROG_WBT_SYNC, PROG_WBT_FRAME_BEGIN,
                                PROG_WBT_FRAME_END, PROG_WBT_CONTEXT):
                pass  # handled implicitly

            offset += block_len

        return rects

    def _decode_region(self, data, output, width, height):
        if len(data) < 12:
            return []

        num_rects = struct.unpack_from('<H', data, 1)[0]
        num_quant = data[3]
        num_prog_quant = data[4]
        flags = data[5]
        num_tiles = struct.unpack_from('<H', data, 6)[0]

        offset = 12

        # Parse rects
        rects = []
        for _ in range(num_rects):
            if offset + 8 > len(data):
                return []
            rx, ry, rw, rh = struct.unpack_from('<HHHH', data, offset)
            rects.append((rx, ry, rw, rh))
            offset += 8

        # Parse quant values (5 bytes each)
        quants = []
        for _ in range(num_quant):
            if offset + 5 > len(data):
                return []
            quants.append(RfxQuant(data[offset:offset + 5]))
            offset += 5

        # Parse progressive quant values (16 bytes each)
        prog_quants = []
        for _ in range(num_prog_quant):
            if offset + 16 > len(data):
                return []
            prog_quants.append(ProgQuant(data[offset:offset + 16]))
            offset += 16

        log.debug("RFX: region %d tiles, %d quants, %d rects, flags=0x%02X" %
                  (num_tiles, num_quant, num_rects, flags))
        for i, (rx, ry, rw, rh) in enumerate(rects):
            log.debug("RFX:   rect[%d]=(%d,%d,%d,%d)" % (i, rx, ry, rw, rh))
        for i, q in enumerate(quants):
            log.debug("RFX:   quant[%d] LL3=%d LH3=%d HL3=%d HH3=%d LH2=%d HL2=%d HH2=%d LH1=%d HL1=%d HH1=%d" %
                      (i, q.LL3, q.LH3, q.HL3, q.HH3, q.LH2, q.HL2, q.HH2, q.LH1, q.HL1, q.HH1))

        extrapolate = (flags & 0x01) != 0

        tile_positions = []
        for _ in range(num_tiles):
            if offset + 6 > len(data):
                break
            tile_type, tile_len = struct.unpack_from('<HI', data, offset)

            if tile_len < 6 or offset + tile_len > len(data):
                break

            tile_data = data[offset + 6:offset + tile_len]

            # Log tile position from header (xIdx at offset 3, yIdx at offset 5)
            if len(tile_data) >= 7:
                tx, ty = struct.unpack_from('<HH', tile_data, 3)
                tile_positions.append((tx, ty))

            if tile_type == PROG_WBT_TILE_SIMPLE:
                self._decode_tile_simple(tile_data, quants, output, width, height, extrapolate)
            elif tile_type == PROG_WBT_TILE_FIRST:
                self._decode_tile_first(tile_data, quants, prog_quants, output, width, height, extrapolate)
            elif tile_type == PROG_WBT_TILE_UPGRADE:
                self._decode_tile_upgrade(tile_data, quants, prog_quants,
                                          output, width, height, extrapolate)

            offset += tile_len

        if tile_positions:
            log.debug("RFX: tile positions (idx): %s" %
                      ", ".join("(%d,%d)" % (tx, ty) for tx, ty in tile_positions[:10]))
            if len(tile_positions) > 10:
                log.debug("RFX:   ... and %d more tiles" % (len(tile_positions) - 10))

        # MS-RDPEGFX: numRects==0 means the tile data represents a full-surface
        # refresh — compute dirty bounds from decoded tile positions.
        if num_rects == 0 and tile_positions:
            min_x = min(tx * 64 for (tx, ty) in tile_positions)
            min_y = min(ty * 64 for (tx, ty) in tile_positions)
            max_x = min(max(tx * 64 + 64 for (tx, ty) in tile_positions), width)
            max_y = min(max(ty * 64 + 64 for (tx, ty) in tile_positions), height)
            rects = [(min_x, min_y, max_x - min_x, max_y - min_y)]
        elif num_rects == 0:
            rects = [(0, 0, width, height)]

        return rects

    def _get_quant(self, quants, idx):
        if idx < len(quants):
            return quants[idx]
        return RfxQuant.default()

    def _decode_tile_simple(self, data, quants, output, out_w, out_h, extrapolate=False):
        if len(data) < 16:
            return
        quant_idx_y = data[0]
        quant_idx_cb = data[1]
        quant_idx_cr = data[2]
        x_idx, y_idx = struct.unpack_from('<HH', data, 3)
        flags = data[7]
        y_len, cb_len, cr_len = struct.unpack_from('<HHH', data, 8)

        is_diff = (flags & 0x01) != 0

        off = 16
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

        if self._debug_tile_count < 3:
            log.debug("RFX: TILE_SIMPLE (%d,%d) isDiff=%s yLen=%d cbLen=%d crLen=%d qY=%d qCb=%d qCr=%d" %
                      (x_idx, y_idx, is_diff, y_len, cb_len, cr_len,
                       quant_idx_y, quant_idx_cb, quant_idx_cr))
        self._debug_tile_count += 1

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

    def _decode_tile_first(self, data, quants, prog_quants, output, out_w, out_h, extrapolate=False):
        if len(data) < 17:
            return
        quant_idx_y = data[0]
        quant_idx_cb = data[1]
        quant_idx_cr = data[2]
        x_idx, y_idx = struct.unpack_from('<HH', data, 3)
        flags = data[7]
        quality = data[8]
        y_len, cb_len, cr_len = struct.unpack_from('<HHH', data, 9)

        is_diff = (flags & 0x01) != 0

        off = 17
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

        if self._debug_tile_count < 3:
            log.debug("RFX: TILE_FIRST (%d,%d) isDiff=%s quality=%d yLen=%d cbLen=%d crLen=%d qY=%d qCb=%d qCr=%d" %
                      (x_idx, y_idx, is_diff, quality, y_len, cb_len, cr_len,
                       quant_idx_y, quant_idx_cb, quant_idx_cr))
        self._debug_tile_count += 1

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

    def _decode_tile_upgrade(self, data, quants, prog_quants, output, out_w, out_h, extrapolate=False):
        """Decode a TILE_UPGRADE: apply progressive delta to stored tile state."""
        if len(data) < 20:
            return
        quant_idx_y = data[0]
        quant_idx_cb = data[1]
        quant_idx_cr = data[2]
        x_idx, y_idx = struct.unpack_from('<HH', data, 3)
        quality = data[7]
        y_srl_len, y_raw_len, cb_srl_len, cb_raw_len, cr_srl_len, cr_raw_len = struct.unpack_from('<HHHHHH', data, 8)

        off = 20
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

    def _get_prog_quant(self, prog_quants, quality):
        """Get progressive quant entry by quality index."""
        if quality < len(prog_quants):
            return prog_quants[quality]
        if prog_quants:
            return prog_quants[-1]
        return None
