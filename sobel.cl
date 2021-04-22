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
 * @file sobel.cl
 *
 * Device Sobel kernels step-by-step
 *
 * @author Minh Quan HO <mqho@kalrayinc.com>
 *
 ******************************************************************************
 */

#include "sobel_config.h"

#define IMAGE_INDEX(y, x) \
    ((min(y, image_height-1) * image_width) + min(x, image_width-1))

// ============================================================================
// Step 0: Reference GPU-friendly kernel.
//         This kernel is considered as baseline for further steps.

// NOTE: For the sake of simplicity of further optimization steps, we implement
// here a "shifted" Sobel filter, in which the output pixel (y+1, x+1) will
// be stored to the index (y, x).
// ============================================================================
__kernel void sobel_step_0(__global uchar *image_in, __global uchar *image_out,
                           int image_width, int image_height, float scale)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= image_width || y >= image_height) { return; }

    // load neighbors
    float c0 = image_in[IMAGE_INDEX(y+0, x+0)];
    float c1 = image_in[IMAGE_INDEX(y+0, x+1)];
    float c2 = image_in[IMAGE_INDEX(y+0, x+2)];

    float n0 = image_in[IMAGE_INDEX(y+1, x+0)];
    float n2 = image_in[IMAGE_INDEX(y+1, x+2)];

    float t0 = image_in[IMAGE_INDEX(y+2, x+0)];
    float t1 = image_in[IMAGE_INDEX(y+2, x+1)];
    float t2 = image_in[IMAGE_INDEX(y+2, x+2)];

    // compute
    float magx = mad(2.0f, (n2 - n0), (c2 - c0 + t2 - t0));
    float magy = mad(2.0f, (t1 - c1), (t0 - c0 + t2 - c2));
    float mag  = hypot(magx, magy) * scale;

    // store pixel
    image_out[IMAGE_INDEX(y, x)] = convert_uchar_sat(mag);
}


// ============================================================================
// Step 1: Explicit Tiling
//
// Note:
// - This is different from the implicit tiling via 2D local workgroup size in
//   clEnqueueNDRangeKernel()
// - This is explicit tiling in the kernel code (via macros and loops), the
//   workgroup deployment(*) and the way compute is dispatched on all workitems
//   within the workgroup
// - Explicit (large) tiling provides better data-locality and cache-hit ratio
//   on stencil kernels
//
// (*) One workgroup computes one tile with pre-defined size
// TILE_HEIGHT x TILE_WIDTH, regardless its number of local workitems
// ============================================================================

// NOTE: "Shifted" Sobel filter, whose the output tile is shifted back to
// top-left by one pixel, yielding the below stencil block with HALO_SIZE == 2:
//
//                    |<------ TILE_WIDTH ------->|  HALO_SIZE (== 2)
//         -----------+---------------------------+----+
//            ʌ       |                           |    |
//            |       |                           |    |
//            |       |                           |    |
//            |       |                           |    |
//       TILE_HEIGHT  |           Tile            |    |
//            |       |                           |    |
//            |       |                           |    |
//            v       |                           |    |
//         -----------+---------------------------+    |
//         HALO_SIZE  |                                |
//                    +--------------------------------+
//
// HALO_SIZE will be pruned off on edge tiles to avoid out-of-bound memory access
//

#define BLOCK_INDEX(y, x) \
    ((min(y, block_height_halo-1) * block_row_stride) + min(x, block_width_halo-1))

/**
 * @brief      Compute a tile (aka block) (block_height x block_width)
 *
 * @param      block_in           Input block
 * @param      block_out          Output block
 * @param[in]  block_row_stride   Row stride in uchar-pixel of input block
 * @param[in]  block_width        Block width
 * @param[in]  block_height       Block height
 * @param[in]  block_width_halo   Maximal block width with halo (used for clamp)
 * @param[in]  block_height_halo  Maximal block height with halo (used for clamp)
 * @param[in]  scale              Magnitude scale
 */
static void sobel_compute_block_step_1(__global uchar *block_in,
                                       __global uchar *block_out,
                                       int block_row_stride,
                                       int block_width, int block_height,
                                       int block_width_halo, int block_height_halo,
                                       float scale)
{
    const int lsizex = get_local_size(0);
    const int lsizey = get_local_size(1);

    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);

    // number of workitems in workgroup
    const int num_wi = lsizex * lsizey;

    // linearized workitem id in workgroup
    const int wid = lidx + lidy * lsizex;

    // dispatch rows of block block_height x block_width on workitems
    const int num_rows_per_wi   = block_height / num_wi;
    const int num_rows_trailing = block_height % num_wi;

    const int irow_begin = wid * num_rows_per_wi + min(wid, num_rows_trailing);
    const int irow_end   = irow_begin + num_rows_per_wi + ((wid < num_rows_trailing) ? 1 : 0);

    for (int irow = irow_begin; irow < irow_end; irow++)
    {
        for (int icol = 0; icol < block_width; icol++)
        {
            // load neighbors
            float c0 = block_in[BLOCK_INDEX(irow+0, icol+0)];
            float c1 = block_in[BLOCK_INDEX(irow+0, icol+1)];
            float c2 = block_in[BLOCK_INDEX(irow+0, icol+2)];

            float n0 = block_in[BLOCK_INDEX(irow+1, icol+0)];
            float n2 = block_in[BLOCK_INDEX(irow+1, icol+2)];

            float t0 = block_in[BLOCK_INDEX(irow+2, icol+0)];
            float t1 = block_in[BLOCK_INDEX(irow+2, icol+1)];
            float t2 = block_in[BLOCK_INDEX(irow+2, icol+2)];

            // compute
            float magx = mad(2.0f, (n2 - n0), (c2 - c0 + t2 - t0));
            float magy = mad(2.0f, (t1 - c1), (t0 - c0 + t2 - c2));
            float mag  = hypot(magx, magy) * scale;

            // store pixel
            block_out[BLOCK_INDEX(irow, icol)] = convert_uchar_sat(mag);
        }
    }
}

__kernel void sobel_step_1(__global uchar *image_in, __global uchar *image_out,
                           int image_width, int image_height, float scale)
{
    const int group_idx = get_group_id(0);
    const int group_idy = get_group_id(1);

    const int block_idx = group_idx * TILE_WIDTH;
    const int block_idy = group_idy * TILE_HEIGHT;

    if (block_idx >= image_width || block_idy >= image_height) { return; }

    const ulong block_offset = (block_idy * image_width) + block_idx;

    __global uchar *block_in  = image_in  + block_offset;
    __global uchar *block_out = image_out + block_offset;

    const int block_row_stride = image_width;

    const int block_width  = min(TILE_WIDTH, (image_width-block_idx));
    const int block_height = min(TILE_HEIGHT, (image_height-block_idy));

    const int block_width_halo  = min((TILE_WIDTH+HALO_SIZE), (image_width-block_idx));
    const int block_height_halo = min((TILE_HEIGHT+HALO_SIZE), (image_height-block_idy));

    sobel_compute_block_step_1(block_in, block_out,
                               block_row_stride,
                               block_width, block_height,
                               block_width_halo, block_height_halo,
                               scale);
}



// ============================================================================
// Step 2: Explicit Tiling + __local on input
//
// - Same as Step-1, but use __local to preload input block into local memory
//   before processing.
// - Some parameters such as `block_row_stride` will change because we are
//   working from local memory, instead of global memory in step-1.
// - Note that this step does not necessarily outperform the step-1, typically
//   when tiling has been done and $L2 cache enabled ($L2 cache is
//   in SMEM, same as __local). The goal of this step is to introduce using of
//   __local and basic async-copy (no overlapping, yet), preparing for later
//   optimization steps (3, 4, 5, ...).
// ============================================================================

// We differentiate now between block_in_row_stride and
// block_out_row_stride, since blocks now may reside in different address
// spaces, between __local and __global.

#define BLOCK_IN_INDEX(y, x) \
    ((min(y, block_height_halo-1) * block_in_row_stride) + min(x, block_width_halo-1))

#define BLOCK_OUT_INDEX(y, x) \
    ((min(y, block_height_halo-1) * block_out_row_stride) + min(x, block_width_halo-1))

/**
 * @brief      Compute a tile (aka block) (block_height x block_width)
 *
 * @param      block_in_local           Input block in __local
 * @param      block_out                Output block
 * @param[in]  block_in_row_stride      Row stride in uchar-pixel of __local input block
 * @param[in]  block_out_row_stride     Row stride in uchar-pixel of __global output block
 * @param[in]  block_width              Block width
 * @param[in]  block_height             Block height
 * @param[in]  block_width_halo         Maximal block width with halo (used for clamp)
 * @param[in]  block_height_halo        Maximal block height with halo (used for clamp)
 * @param[in]  scale                    Magnitude scale
 */
static void sobel_compute_block_step_2(__local  uchar *block_in_local,
                                       __global uchar *block_out,
                                       int block_in_row_stride, int block_out_row_stride,
                                       int block_width, int block_height,
                                       int block_width_halo, int block_height_halo,
                                       float scale)
{
    const int lsizex = get_local_size(0);
    const int lsizey = get_local_size(1);

    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);

    // number of workitems in workgroup
    const int num_wi = lsizex * lsizey;

    // linearized workitem id in workgroup
    const int wid = lidx + lidy * lsizex;

    // dispatch rows of block block_height x block_width on workitems
    const int num_rows_per_wi   = block_height / num_wi;
    const int num_rows_trailing = block_height % num_wi;

    const int irow_begin = wid * num_rows_per_wi + min(wid, num_rows_trailing);
    const int irow_end   = irow_begin + num_rows_per_wi + ((wid < num_rows_trailing) ? 1 : 0);

    for (int irow = irow_begin; irow < irow_end; irow++)
    {
        for (int icol = 0; icol < block_width; icol++)
        {
            // load neighbors
            float c0 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+0)];
            float c1 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+1)];
            float c2 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+2)];

            float n0 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+0)];
            float n2 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+2)];

            float t0 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+0)];
            float t1 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+1)];
            float t2 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+2)];

            // compute
            float magx = mad(2.0f, (n2 - n0), (c2 - c0 + t2 - t0));
            float magy = mad(2.0f, (t1 - c1), (t0 - c0 + t2 - c2));
            float mag  = hypot(magx, magy) * scale;

            // store pixel
            block_out[BLOCK_OUT_INDEX(irow, icol)] = convert_uchar_sat(mag);
        }
    }
}

__kernel void sobel_step_2(__global uchar *image_in, __global uchar *image_out,
                           int image_width, int image_height, float scale)
{
    __local uchar block_in_local[(TILE_HEIGHT + HALO_SIZE) * (TILE_WIDTH + HALO_SIZE)];

    const int group_idx = get_group_id(0);
    const int group_idy = get_group_id(1);

    const int block_idx = group_idx * TILE_WIDTH;
    const int block_idy = group_idy * TILE_HEIGHT;

    if (block_idx >= image_width || block_idy >= image_height) { return; }

    const ulong block_offset = (block_idy * image_width) + block_idx;

    __global uchar *block_in  = image_in  + block_offset;
    __global uchar *block_out = image_out + block_offset;

    const int block_width  = min(TILE_WIDTH, (image_width-block_idx));
    const int block_height = min(TILE_HEIGHT, (image_height-block_idy));

    const int block_width_halo  = min((TILE_WIDTH+HALO_SIZE), (image_width-block_idx));
    const int block_height_halo = min((TILE_HEIGHT+HALO_SIZE), (image_height-block_idy));

    const int block_in_row_stride  = block_width_halo;
    const int block_out_row_stride = image_width;

    // ------------------------------------------------------------
    // Before computing, copy data to __local
    //     (block_height_halo x block_width_halo)
    // ------------------------------------------------------------
    event_t event;

    // To perform a 2D block-copy, we can either use:
    //    - for-loop with traditional load/store
    //    - for-loop with 1D async-copy primitive (OpenCL 1.2)
    //    - Kalray extension 2D async-copy primitive
    //
    // Let's use 2D copy for better performance than for-loop.

#define HAVE_ASYNC_COPY_2D2D

#ifndef HAVE_ASYNC_COPY_2D2D
    for (int irow = 0; irow < block_height_halo; irow++)
    {
        event = async_work_group_copy(block_in_local + (irow * block_in_row_stride),
                                      block_in + (irow * block_out_row_stride),
                                      block_width_halo, 0);
    }
#else
    int2 block_to_copy = (int2)(block_width_halo, block_height_halo);
    int4 local_point   = (int4)(    0    ,     0    , block_width_halo, block_height_halo);
    int4 global_point  = (int4)(block_idx, block_idy, image_width     , image_height     );
    event = async_work_group_copy_block_2D2D(
                    block_in_local,   // __local buffer
                    image_in,         // __global image
                    1,                // num_gentype_per_pixel
                    block_to_copy,    // block to copy
                    local_point,      // local_point
                    global_point,     // global_point
                    0);
#endif

    // Wait immediately. There is almost no overlapping. This will be tackled
    // in the future steps
    wait_group_events(1, &event);

    // ------------------------------------------------------------
    // Compute (same as sobel_compute_block_step_1() but with
    // block_in_local[])
    // ------------------------------------------------------------
    sobel_compute_block_step_2(block_in_local, block_out,
                               block_in_row_stride, block_out_row_stride,
                               block_width, block_height,
                               block_width_halo, block_height_halo,
                               scale);
}
