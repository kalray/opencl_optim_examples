/**
 *****************************************************************************
 * Copyright (C) 2021 Kalray
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @file sobel_compute_block.c
 *
 * Native C Sobel kernel for Device
 *
 * @author Minh Quan HO <mqho@kalrayinc.com>
 *
 ******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "sobel_config.h"

typedef unsigned char uchar8 __attribute__((vector_size(8 * sizeof(unsigned char))));
typedef short         short8 __attribute__((vector_size(8 * sizeof(short))));
typedef int           int8   __attribute__((vector_size(8 * sizeof(int))));
typedef float         float8 __attribute__((vector_size(8 * sizeof(float))));

#define min(a, b)        ((a) < (b) ? (a) : (b))
#define max(a, b)        ((a) > (b) ? (a) : (b))

#define BLOCK_IN_INDEX(y, x) \
    ((min(y, block_height_halo-1) * block_in_row_stride) + min(x, block_width_halo-1))

#define BLOCK_OUT_INDEX(y, x) \
    ((min(y, block_height_halo-1) * block_out_row_stride) + min(x, block_width_halo-1))


// ============================================================================
// v1
// ============================================================================
static inline unsigned char convert_uchar_sat(const float mag)
{
    const float mag_sat = __builtin_kvx_fmaxw(0.0f, __builtin_kvx_fminw(mag, (float)UCHAR_MAX));
    return (unsigned char) mag_sat;
}

static inline float hypot_float(const float magx, const float magy)
{
    const float sum_pow_xy = (magx * magx) + (magy * magy);
    // enabling -ffast-math will allow compiler emitting inverse-square-root
    // ISA instruction. Note: this is not IEEE-compliant, so we consider only
    // non-zero positive floating-point numbers here.
    return ((sum_pow_xy <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy));
}

// ============================================================================
// v8
// ============================================================================
static inline short8 convert_short8(const uchar8 u8)
{
    return (short8){u8[0], u8[1], u8[2], u8[3], u8[4], u8[5], u8[6], u8[7]};
}

static inline int8 convert_int8(const short8 u8)
{
    return (int8){u8[0], u8[1], u8[2], u8[3], u8[4], u8[5], u8[6], u8[7]};
}

static inline float8 convert_float8(const short8 u8)
{
    return __builtin_kvx_floatwo(convert_int8(u8), 0, ".rn");
}

static inline float8 hypot_float8(const float8 magx, const float8 magy)
{
    const float8 sum_pow_xy = (magx * magx) + (magy * magy);
    // enabling -ffast-math will allow compiler emitting inverse-square-root
    // ISA instruction. Note: this is not IEEE-compliant, so we consider only
    // non-zero positive floating-point numbers here.
    return (float8){((sum_pow_xy[0] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[0])),
                    ((sum_pow_xy[1] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[1])),
                    ((sum_pow_xy[2] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[2])),
                    ((sum_pow_xy[3] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[3])),
                    ((sum_pow_xy[4] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[4])),
                    ((sum_pow_xy[5] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[5])),
                    ((sum_pow_xy[6] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[6])),
                    ((sum_pow_xy[7] <= FLT_EPSILON) ? 0.0f : sqrtf(sum_pow_xy[7]))};
}

static inline uchar8 convert_uchar8_sat(const float8 mag)
{
    const float8 zero      = {0.0f};
    const float8 uchar_max = {UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
                              UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX};

    const float8 mag_sat = __builtin_kvx_fmaxwo(zero, __builtin_kvx_fminwo(mag, uchar_max));
    const uchar8 mag_sat_ret = (uchar8) {mag_sat[0], mag_sat[1], mag_sat[2], mag_sat[3],
                                         mag_sat[4], mag_sat[5], mag_sat[6], mag_sat[7]};
    return mag_sat_ret;
}


void native_sobel_compute_block(const unsigned char *block_in_local,
                                unsigned char *block_out_local,
                                const int block_in_row_stride, const int block_out_row_stride,
                                const int block_width, const int block_height,
                                const int block_width_halo, const int block_height_halo,
                                const float scale,
                                const int num_threads,
                                const int thread_id)
{
    // dispatch rows of block block_height x block_width on workitems
    const div_t division = div(block_height, num_threads);
    const int num_rows_per_thread = division.quot;
    const int num_rows_trailing   = division.rem;

    const int irow_begin = thread_id * num_rows_per_thread + min(thread_id, num_rows_trailing);
    const int irow_end   = irow_begin + num_rows_per_thread + ((thread_id < num_rows_trailing) ? 1 : 0);

    if (irow_begin == irow_end) { return; }

    int icol = 0;

    // take care to not exceed 'block_width_halo' when doing
    // vectorized neighbor reads
    for (; icol + 8 + HALO_SIZE <= block_width_halo; icol += 8)
    {
        int irow = irow_begin;

        // load neighbors
        short8 c0 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+0, icol+0)]));
        short8 c1 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+0, icol+1)]));
        short8 c2 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+0, icol+2)]));

        short8 n0 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+1, icol+0)]));
        short8 n2 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+1, icol+2)]));

        short8 t0 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+2, icol+0)]));
        short8 t1 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+2, icol+1)]));
        short8 t2 = convert_short8(*((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+2, icol+2)]));

        for (; irow < irow_end; irow++)
        {
            const uchar8 n1 = *((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+1, icol+1)]);

            const uchar8 t0_next = *((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+3, icol+0)]);
            const uchar8 t1_next = *((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+3, icol+1)]);
            const uchar8 t2_next = *((uchar8 *)&block_in_local[BLOCK_IN_INDEX(irow+3, icol+2)]);

            // compute
            const float8 magx = convert_float8(((short)2 * (n2 - n0)) + (c2 - c0 + t2 - t0));
            const float8 magy = convert_float8(((short)2 * (t1 - c1)) + (t0 - c0 + t2 - c2));
            const float8 mag  = hypot_float8(magx, magy) * scale;

            // store pixel
            *((uchar8 *)&block_out_local[BLOCK_OUT_INDEX(irow, icol)]) = convert_uchar8_sat(mag);

            c0 = n0;
            c1 = convert_short8(n1);
            c2 = n2;

            n0 = t0;
            n2 = t2;

            t0 = convert_short8(t0_next);
            t1 = convert_short8(t1_next);
            t2 = convert_short8(t2_next);
        }
    }

    for (; icol < block_width; icol ++)
    {
        int irow = irow_begin;

        // load neighbors
        short c0 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+0)];
        short c1 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+1)];
        short c2 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+2)];

        short n0 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+0)];
        short n2 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+2)];

        short t0 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+0)];
        short t1 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+1)];
        short t2 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+2)];

        for (; irow < irow_end; irow++)
        {
            const unsigned char n1 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+1)];

            const unsigned char t0_next = block_in_local[BLOCK_IN_INDEX(irow+3, icol+0)];
            const unsigned char t1_next = block_in_local[BLOCK_IN_INDEX(irow+3, icol+1)];
            const unsigned char t2_next = block_in_local[BLOCK_IN_INDEX(irow+3, icol+2)];

            // compute
            const float magx = (float)(((short)2 * (n2 - n0)) + (c2 - c0 + t2 - t0));
            const float magy = (float)(((short)2 * (t1 - c1)) + (t0 - c0 + t2 - c2));
            const float mag  = hypot_float(magx, magy) * scale;

            // store pixel
            block_out_local[BLOCK_OUT_INDEX(irow, icol)] = convert_uchar_sat(mag);

            c0 = n0;
            c1 = n1;
            c2 = n2;

            n0 = t0;
            n2 = t2;

            t0 = t0_next;
            t1 = t1_next;
            t2 = t2_next;
        }
    }
}
