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



// ============================================================================
// Step 3: Explicit Tiling + __local on both input/output
//
// - Same as Step-2, but use __local to store output block before sending to
//   __global
// ============================================================================

/**
 * @brief      Compute a tile (aka block) (block_height x block_width)
 *
 * @param      block_in_local           Input block in __local
 * @param      block_out_local          Output block in __local
 * @param[in]  block_in_row_stride      Row stride in uchar-pixel of __local input block
 * @param[in]  block_out_row_stride     Row stride in uchar-pixel of __local output block
 * @param[in]  block_width              Block width
 * @param[in]  block_height             Block height
 * @param[in]  block_width_halo         Maximal block width with halo (used for clamp)
 * @param[in]  block_height_halo        Maximal block height with halo (used for clamp)
 * @param[in]  scale                    Magnitude scale
 */
static void sobel_compute_block_step_3(__local uchar *block_in_local,
                                       __local uchar *block_out_local,
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
            block_out_local[BLOCK_OUT_INDEX(irow, icol)] = convert_uchar_sat(mag);
        }
    }

    // sync to gather result from all WI
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void sobel_step_3(__global uchar *image_in, __global uchar *image_out,
                           int image_width, int image_height, float scale)
{
    __local uchar block_in_local [(TILE_HEIGHT + HALO_SIZE) * (TILE_WIDTH + HALO_SIZE)];
    __local uchar block_out_local[       TILE_HEIGHT        *        TILE_WIDTH       ];

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
    const int block_out_row_stride = block_width;

    // ------------------------------------------------------------
    // Before computing, copy data to __local
    //     (block_height_halo x block_width_halo)
    // ------------------------------------------------------------
    event_t event;

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

    // Wait immediately. There is almost no overlapping. This will be tackled
    // in the future steps
    wait_group_events(1, &event);

    // ------------------------------------------------------------
    // Compute (same as sobel_compute_block_step_1() but with
    // block_in_local[])
    // ------------------------------------------------------------
    sobel_compute_block_step_3(block_in_local, block_out_local,
                               block_in_row_stride, block_out_row_stride,
                               block_width, block_height,
                               block_width_halo, block_height_halo,
                               scale);

    // ------------------------------------------------------------
    // After computing, send result to __global
    //     (block_height x block_width)
    // ------------------------------------------------------------
    block_to_copy = (int2)(block_width, block_height);
    local_point   = (int4)(    0    ,     0    , block_width, block_height);
    event = async_work_group_copy_block_2D2D(
                    image_out,        // __global image
                    block_out_local,  // __local buffer
                    1,                // num_gentype_per_pixel
                    block_to_copy,    // block to copy
                    local_point,      // local_point
                    global_point,     // global_point
                    0);

    // Wait immediately. There is almost no overlapping. This will be tackled
    // in the future steps
    wait_group_events(1, &event);
}



// ============================================================================
// Step 4: Explicit Tiling + __local on both input/output + double-buffering
//
// - Same as Step-3, but with double-buffering and overlapping, this will be
//   the ultimate optimization step on data transfer.
// - Unlike previous steps 1/2/3, in which each WG computes one tile, and
//   there are as many WG as the number of tiles in image. In this step, there
//   will be only 5 WG spawned (CL_DEVICE_MAX_COMPUTE_UNITS). Each WG will be
//   in charge of multiple tiles and perform overlapping through
//   double-buffering and async-copy.
// ============================================================================

// We reuse the same sobel_compute_block_step_3() function for step-4
static void sobel_compute_block_step_4(__local uchar *block_in_local,
                                       __local uchar *block_out_local,
                                       int block_in_row_stride, int block_out_row_stride,
                                       int block_width, int block_height,
                                       int block_width_halo, int block_height_halo,
                                       float scale)
__attribute__((alias("sobel_compute_block_step_3")));


#define OCL_KERNEL_DMA_TILING_ENGINE_SOBEL(KERNEL_NAME, COMPUTE_BLOCK_FUNC)                       \
                                                                                                  \
__kernel void KERNEL_NAME(__global uchar *image_in, __global uchar *image_out,                    \
                          int image_width, int image_height, float scale)                         \
{                                                                                                 \
    __local uchar block_in_local [2][(TILE_HEIGHT + HALO_SIZE) * (TILE_WIDTH + HALO_SIZE)];       \
    __local uchar block_out_local[2][       TILE_HEIGHT        *        TILE_WIDTH       ];       \
                                                                                                  \
    event_t event_read[2]  = {0, 0};                                                              \
    event_t event_write[2] = {0, 0};                                                              \
                                                                                                  \
    const int group_idx = get_group_id(0);                                                        \
    const int group_idy = get_group_id(1);                                                        \
                                                                                                  \
    const int num_groups = get_num_groups(0) * get_num_groups(1);                                 \
    const int group_id = (group_idy * get_num_groups(0)) + group_idx;                             \
                                                                                                  \
    const int num_blocks_x = (int)ceil(((float)image_width)  / TILE_WIDTH);                       \
    const int num_blocks_y = (int)ceil(((float)image_height) / TILE_HEIGHT);                      \
    const int num_blocks_total = num_blocks_x * num_blocks_y;                                     \
                                                                                                  \
    const int block_dispatch_step = num_groups;                                                   \
    const int iblock_begin        = group_id;                                                     \
    const int iblock_end          = num_blocks_total;                                             \
                                                                                                  \
    /* ===================================================================== */                   \
    /* PROLOGUE: prefetch first block                                        */                   \
    /* ===================================================================== */                   \
    int2 block_to_copy;                                                                           \
    int4 local_point;                                                                             \
    int4 global_point;                                                                            \
                                                                                                  \
    int iblock_x_next = iblock_begin % num_blocks_x;                                              \
    int iblock_y_next = iblock_begin / num_blocks_x;                                              \
                                                                                                  \
    int block_idx_next = iblock_x_next * TILE_WIDTH;                                              \
    int block_idy_next = iblock_y_next * TILE_HEIGHT;                                             \
                                                                                                  \
    int block_width_next  = min(TILE_WIDTH, (image_width-block_idx_next));                        \
    int block_height_next = min(TILE_HEIGHT, (image_height-block_idy_next));                      \
                                                                                                  \
    int block_width_halo_next  = min((TILE_WIDTH+HALO_SIZE), (image_width-block_idx_next));       \
    int block_height_halo_next = min((TILE_HEIGHT+HALO_SIZE), (image_height-block_idy_next));     \
                                                                                                  \
    /* prefetch first block */                                                                    \
    int block_counter = 0;                                                                        \
    block_to_copy = (int2)(block_width_halo_next, block_height_halo_next);                        \
    local_point  = (int4)(0, 0, block_width_halo_next, block_height_halo_next);                   \
    global_point = (int4)(block_idx_next, block_idy_next, image_width, image_height);             \
                                                                                                  \
    event_read[block_counter & 1] = async_work_group_copy_block_2D2D(                             \
                    block_in_local[block_counter & 1],     /* __local buffer         */           \
                    image_in,                              /* __global image         */           \
                    1,                                     /* num_gentype_per_pixel  */           \
                    block_to_copy,                         /* block to copy          */           \
                    local_point,                           /* local_point            */           \
                    global_point,                          /* global_point           */           \
                    0);                                                                           \
                                                                                                  \
    /* ===================================================================== */                   \
    /* FOR-LOOP: Compute all blocks                                          */                   \
    /* ===================================================================== */                   \
    for (int iblock = iblock_begin; iblock < iblock_end; iblock += block_dispatch_step,           \
                                                         block_counter++)                         \
    {                                                                                             \
        /* ------------------------------------------------------------ */                        \
        /* current block to be processed                                */                        \
        /* ------------------------------------------------------------ */                        \
        const int iblock_parity        = block_counter & 1;                                       \
                                                                                                  \
        const int block_idx            = block_idx_next;                                          \
        const int block_idy            = block_idy_next;                                          \
                                                                                                  \
        const int block_width          = block_width_next;                                        \
        const int block_height         = block_height_next;                                       \
                                                                                                  \
        const int block_width_halo     = block_width_halo_next;                                   \
        const int block_height_halo    = block_height_halo_next;                                  \
                                                                                                  \
        const int block_in_row_stride  = block_width_halo;                                        \
        const int block_out_row_stride = block_width;                                             \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* prefetch next block (if any)                                 */                        \
        /* ------------------------------------------------------------ */                        \
        const int iblock_next = iblock + block_dispatch_step;                                     \
                                                                                                  \
        if (iblock_next < iblock_end)                                                             \
        {                                                                                         \
            const int iblock_next_parity = (block_counter+1) & 1;                                 \
                                                                                                  \
            iblock_x_next = iblock_next % num_blocks_x;                                           \
            iblock_y_next = iblock_next / num_blocks_x;                                           \
            block_idx_next = iblock_x_next * TILE_WIDTH;                                          \
            block_idy_next = iblock_y_next * TILE_HEIGHT;                                         \
                                                                                                  \
            block_width_next  = min(TILE_WIDTH, (image_width-block_idx_next));                    \
            block_height_next = min(TILE_HEIGHT, (image_height-block_idy_next));                  \
                                                                                                  \
            block_width_halo_next  = min((TILE_WIDTH+HALO_SIZE), (image_width-block_idx_next));   \
            block_height_halo_next = min((TILE_HEIGHT+HALO_SIZE), (image_height-block_idy_next)); \
                                                                                                  \
            block_to_copy = (int2)(block_width_halo_next, block_height_halo_next);                \
            local_point  = (int4)(0, 0, block_width_halo_next, block_height_halo_next);           \
            global_point = (int4)(block_idx_next, block_idy_next, image_width, image_height);     \
                                                                                                  \
            event_read[iblock_next_parity] = async_work_group_copy_block_2D2D(                    \
                        block_in_local[iblock_next_parity],    /* __local buffer         */       \
                        image_in,                              /* __global image         */       \
                        1,                                     /* num_gentype_per_pixel  */       \
                        block_to_copy,                         /* block to copy          */       \
                        local_point,                           /* local_point            */       \
                        global_point,                          /* global_point           */       \
                        0);                                                                       \
        }                                                                                         \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* wait for prefetch of current block                           */                        \
        /* ------------------------------------------------------------ */                        \
        wait_group_events(1, &event_read[iblock_parity]);                                         \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* wait for previous put of the 2D block from local to global   */                        \
        /* to avoid data race: writing result to a being-put buffer     */                        \
        /* ------------------------------------------------------------ */                        \
        if (block_counter >= 2) {                                                                 \
            wait_group_events(1, &event_write[iblock_parity]);                                    \
        }                                                                                         \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* now ready to compute the current block                       */                        \
        /* ------------------------------------------------------------ */                        \
        COMPUTE_BLOCK_FUNC(block_in_local[iblock_parity],                                         \
                           block_out_local[iblock_parity],                                        \
                           block_in_row_stride, block_out_row_stride,                             \
                           block_width, block_height,                                             \
                           block_width_halo, block_height_halo,                                   \
                           scale);                                                                \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* put result to global memory                                  */                        \
        /* ------------------------------------------------------------ */                        \
        block_to_copy = (int2)(block_width, block_height);                                        \
        local_point   = (int4)(    0    ,     0    , block_width, block_height);                  \
        global_point  = (int4)(block_idx, block_idy, image_width, image_height);                  \
        event_write[iblock_parity] = async_work_group_copy_block_2D2D(                            \
                        image_out,                       /* __global image          */            \
                        block_out_local[iblock_parity],  /* __local buffer          */            \
                        1,                               /* num_gentype_per_pixel   */            \
                        block_to_copy,                   /* block to copy           */            \
                        local_point,                     /* local_point             */            \
                        global_point,                    /* global_point            */            \
                        0);                                                                       \
                                                                                                  \
    }                                                                                             \
                                                                                                  \
    /* ===================================================================== */                   \
    /* End of compute, fence all outstanding put                             */                   \
    /* ===================================================================== */                   \
    async_work_group_copy_fence(CLK_GLOBAL_MEM_FENCE);                                            \
}

OCL_KERNEL_DMA_TILING_ENGINE_SOBEL(sobel_step_4, sobel_compute_block_step_4)



// ============================================================================
// Step 4-PAPI: PAPI Profiling
//
// - Now we have async-ed all the things, let's try identifying potential
//   bottlenecks with some PAPI instrumentation.
//
// - Depending on image-processing algorithms, performance bottlenecks can be
//   in data transfer, in pre/post-processing, or in compute_block() etc.
//
// ============================================================================

#define MAX_PAPI_EVENTS 2

#define PAPI_LOG_REAL_CYCLE(counter) \
if (group_id < 5 && lidx == 0 && lidy == 0) { counter = PAPI_get_real_cyc(); }

#define PAPI_ACCUMULATE_REAL_CYCLE(counter, acc) \
if (group_id < 5 && lidx == 0 && lidy == 0) { acc += PAPI_get_real_cyc() - counter; }

#define PAPI_LOG_REAL_USEC(counter) \
if (group_id < 5 && lidx == 0 && lidy == 0) { counter = PAPI_get_real_usec(); }

#define PAPI_ACCUMULATE_REAL_USEC(counter, acc) \
if (group_id < 5 && lidx == 0 && lidy == 0) { acc += PAPI_get_real_usec() - counter; }


#define OCL_KERNEL_DMA_TILING_ENGINE_SOBEL_PAPI(KERNEL_NAME, COMPUTE_BLOCK_FUNC)                  \
                                                                                                  \
__kernel void KERNEL_NAME(__global uchar *image_in, __global uchar *image_out,                    \
                          int image_width, int image_height, float scale)                         \
{                                                                                                 \
    __local uchar block_in_local [2][(TILE_HEIGHT + HALO_SIZE) * (TILE_WIDTH + HALO_SIZE)];       \
    __local uchar block_out_local[2][       TILE_HEIGHT        *        TILE_WIDTH       ];       \
                                                                                                  \
    event_t event_read[2]  = {0, 0};                                                              \
    event_t event_write[2] = {0, 0};                                                              \
                                                                                                  \
    const int lidx = get_local_id(0);                                                             \
    const int lidy = get_local_id(1);                                                             \
                                                                                                  \
    const int group_idx = get_group_id(0);                                                        \
    const int group_idy = get_group_id(1);                                                        \
                                                                                                  \
    const int num_groups = get_num_groups(0) * get_num_groups(1);                                 \
    const int group_id = (group_idy * get_num_groups(0)) + group_idx;                             \
                                                                                                  \
    const int num_blocks_x = (int)ceil(((float)image_width)  / TILE_WIDTH);                       \
    const int num_blocks_y = (int)ceil(((float)image_height) / TILE_HEIGHT);                      \
    const int num_blocks_total = num_blocks_x * num_blocks_y;                                     \
                                                                                                  \
    const int block_dispatch_step = num_groups;                                                   \
    const int iblock_begin        = group_id;                                                     \
    const int iblock_end          = num_blocks_total;                                             \
                                                                                                  \
    /* ===================================================================== */                   \
    /* PROLOGUE: prefetch first block                                        */                   \
    /* ===================================================================== */                   \
    int2 block_to_copy;                                                                           \
    int4 local_point;                                                                             \
    int4 global_point;                                                                            \
                                                                                                  \
    int iblock_x_next = iblock_begin % num_blocks_x;                                              \
    int iblock_y_next = iblock_begin / num_blocks_x;                                              \
                                                                                                  \
    int block_idx_next = iblock_x_next * TILE_WIDTH;                                              \
    int block_idy_next = iblock_y_next * TILE_HEIGHT;                                             \
                                                                                                  \
    int block_width_next  = min(TILE_WIDTH, (image_width-block_idx_next));                        \
    int block_height_next = min(TILE_HEIGHT, (image_height-block_idy_next));                      \
                                                                                                  \
    int block_width_halo_next  = min((TILE_WIDTH+HALO_SIZE), (image_width-block_idx_next));       \
    int block_height_halo_next = min((TILE_HEIGHT+HALO_SIZE), (image_height-block_idy_next));     \
                                                                                                  \
    /* prefetch first block */                                                                    \
    int block_counter = 0;                                                                        \
    block_to_copy = (int2)(block_width_halo_next, block_height_halo_next);                        \
    local_point  = (int4)(0, 0, block_width_halo_next, block_height_halo_next);                   \
    global_point = (int4)(block_idx_next, block_idy_next, image_width, image_height);             \
                                                                                                  \
    event_read[block_counter & 1] = async_work_group_copy_block_2D2D(                             \
                    block_in_local[block_counter & 1],     /* __local buffer        */            \
                    image_in,                              /* __global image        */            \
                    1,                                     /* num_gentype_per_pixel */            \
                    block_to_copy,                         /* block to copy         */            \
                    local_point,                           /* local_point           */            \
                    global_point,                          /* global_point          */            \
                    0);                                                                           \
                                                                                                  \
    /* ===================================================================== */                   \
    /* Setting up PAPI                                                       */                   \
    /* ===================================================================== */                   \
    /* Hardware Performance Monitoring Counters (PMC)                        */                   \
    /* With PAPI, each core can use upto two PMCs to track events.           */                   \
    /* For simplicity, we use PE0 of each cluster to measure two differents  */                   \
    /* events. User can refer to the Kalray PAPI manual and/or Kalray VLIW   */                   \
    /* architecture document for details on these PMC events.                */                   \
    __constant char *pmc_names[10]  = {                                                           \
        /* Work-group 0 */ "PCC",   "EBE",                                                        \
        /* Work-group 1 */ "DCLHE", "DCLME",                                                      \
        /* Work-group 2 */ "DARSC", "LDSC",                                                       \
        /* Work-group 3 */ "DMAE",  "DDSC",                                                       \
        /* Work-group 4 */ "PSC",   "SC"                                                          \
    };                                                                                            \
    long pmc_values[MAX_PAPI_EVENTS] = {0};                                                       \
    long tmp_cycle = 0;                                                                           \
    long tmp_usec = 0;                                                                            \
    long sobel_cycles = 0;                                                                        \
    long sobel_usec = 0;                                                                          \
    int  papi_evenSet = PAPI_NULL;                                                                \
    if (group_id < 5 && lidx == 0 && lidy == 0) {                                                 \
        int ret;                                                                                  \
        ret = PAPI_create_eventset(&papi_evenSet);                                                \
        if (ret != PAPI_OK) {                                                                     \
           printf("Failed to PAPI_create_eventset\n");                                            \
        }                                                                                         \
        for (int i = 0; i < MAX_PAPI_EVENTS; i++) {                                               \
            ret = PAPI_add_named_event(papi_evenSet, pmc_names[group_id*2 + i]);                  \
            if (ret != PAPI_OK) {                                                                 \
               printf("Failed to PAPI_add_named_event(%s)\n", pmc_names[group_id*2 + i]);         \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    /* ===================================================================== */                   \
    /* FOR-LOOP: Compute all blocks                                          */                   \
    /* ===================================================================== */                   \
    for (int iblock = iblock_begin; iblock < iblock_end; iblock += block_dispatch_step,           \
                                                         block_counter++)                         \
    {                                                                                             \
        /* ------------------------------------------------------------ */                        \
        /* current block to be processed                                */                        \
        /* ------------------------------------------------------------ */                        \
        const int iblock_parity        = block_counter & 1;                                       \
                                                                                                  \
        const int block_idx            = block_idx_next;                                          \
        const int block_idy            = block_idy_next;                                          \
                                                                                                  \
        const int block_width          = block_width_next;                                        \
        const int block_height         = block_height_next;                                       \
                                                                                                  \
        const int block_width_halo     = block_width_halo_next;                                   \
        const int block_height_halo    = block_height_halo_next;                                  \
                                                                                                  \
        const int block_in_row_stride  = block_width_halo;                                        \
        const int block_out_row_stride = block_width;                                             \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* prefetch next block (if any)                                 */                        \
        /* ------------------------------------------------------------ */                        \
        const int iblock_next = iblock + block_dispatch_step;                                     \
                                                                                                  \
        if (iblock_next < iblock_end)                                                             \
        {                                                                                         \
            const int iblock_next_parity = (block_counter+1) & 1;                                 \
                                                                                                  \
            iblock_x_next = iblock_next % num_blocks_x;                                           \
            iblock_y_next = iblock_next / num_blocks_x;                                           \
            block_idx_next = iblock_x_next * TILE_WIDTH;                                          \
            block_idy_next = iblock_y_next * TILE_HEIGHT;                                         \
                                                                                                  \
            block_width_next  = min(TILE_WIDTH, (image_width-block_idx_next));                    \
            block_height_next = min(TILE_HEIGHT, (image_height-block_idy_next));                  \
                                                                                                  \
            block_width_halo_next  = min((TILE_WIDTH+HALO_SIZE), (image_width-block_idx_next));   \
            block_height_halo_next = min((TILE_HEIGHT+HALO_SIZE), (image_height-block_idy_next)); \
                                                                                                  \
            block_to_copy = (int2)(block_width_halo_next, block_height_halo_next);                \
            local_point  = (int4)(0, 0, block_width_halo_next, block_height_halo_next);           \
            global_point = (int4)(block_idx_next, block_idy_next, image_width, image_height);     \
                                                                                                  \
            event_read[iblock_next_parity] = async_work_group_copy_block_2D2D(                    \
                        block_in_local[iblock_next_parity],    /* __local buffer        */        \
                        image_in,                              /* __global image        */        \
                        1,                                     /* num_gentype_per_pixel */        \
                        block_to_copy,                         /* block to copy         */        \
                        local_point,                           /* local_point           */        \
                        global_point,                          /* global_point          */        \
                        0);                                                                       \
        }                                                                                         \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* wait for prefetch of current block                           */                        \
        /* ------------------------------------------------------------ */                        \
        wait_group_events(1, &event_read[iblock_parity]);                                         \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* wait for previous put of the 2D block from local to global   */                        \
        /* to avoid data race: writing result to a being-put buffer     */                        \
        /* ------------------------------------------------------------ */                        \
        if (block_counter >= 2) {                                                                 \
            wait_group_events(1, &event_write[iblock_parity]);                                    \
        }                                                                                         \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* now ready to compute the current block                       */                        \
        /* ------------------------------------------------------------ */                        \
        PAPI_LOG_REAL_CYCLE(tmp_cycle);                                                           \
        PAPI_LOG_REAL_USEC(tmp_usec);                                                             \
                                                                                                  \
        long pmc_unit[MAX_PAPI_EVENTS] = {0};                                                     \
        if (group_id < 5 && lidx == 0 && lidy == 0) {                                             \
            int ret;                                                                              \
            ret = PAPI_reset(papi_evenSet);                                                       \
            if (ret != PAPI_OK) {                                                                 \
                printf("Failed to PAPI_reset\n");                                                 \
            }                                                                                     \
            ret = PAPI_start(papi_evenSet);                                                       \
            if (ret != PAPI_OK) {                                                                 \
                printf("Failed to PAPI_start\n");                                                 \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        COMPUTE_BLOCK_FUNC(block_in_local[iblock_parity],                                         \
                           block_out_local[iblock_parity],                                        \
                           block_in_row_stride, block_out_row_stride,                             \
                           block_width, block_height,                                             \
                           block_width_halo, block_height_halo,                                   \
                           scale);                                                                \
                                                                                                  \
        if (group_id < 5 && lidx == 0 && lidy == 0) {                                             \
            int ret = PAPI_stop(papi_evenSet, pmc_unit);                                          \
            if (ret != PAPI_OK) {                                                                 \
                printf("Failed to PAPI_stop\n");                                                  \
            }                                                                                     \
                                                                                                  \
            /* accumulate */                                                                      \
            for (int i = 0; i < MAX_PAPI_EVENTS; i++) {                                           \
                pmc_values[i] += pmc_unit[i];                                                     \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        PAPI_ACCUMULATE_REAL_CYCLE(tmp_cycle, sobel_cycles);                                      \
        PAPI_ACCUMULATE_REAL_USEC(tmp_usec, sobel_usec);                                          \
                                                                                                  \
        /* ------------------------------------------------------------ */                        \
        /* put result to global memory                                  */                        \
        /* ------------------------------------------------------------ */                        \
        block_to_copy = (int2)(block_width, block_height);                                        \
        local_point   = (int4)(    0    ,     0    , block_width, block_height);                  \
        global_point  = (int4)(block_idx, block_idy, image_width, image_height);                  \
        event_write[iblock_parity] = async_work_group_copy_block_2D2D(                            \
                        image_out,                       /* __global image        */              \
                        block_out_local[iblock_parity],  /* __local buffer        */              \
                        1,                               /* num_gentype_per_pixel */              \
                        block_to_copy,                   /* block to copy         */              \
                        local_point,                     /* local_point           */              \
                        global_point,                    /* global_point          */              \
                        0);                                                                       \
                                                                                                  \
    }                                                                                             \
                                                                                                  \
    /* ===================================================================== */                   \
    /* Reporting PAPI                                                        */                   \
    /* ===================================================================== */                   \
    if (group_id < 5 && lidx == 0 && lidy == 0) {                                                 \
        /* Note: These printf will slow-down this kernel. */                                      \
        printf("[PAPI] sobel_step_4_PAPI(): WG %d: Processed %d blocks"                           \
               "  sobel_cycles %ld  sobel_usec %ld = %.3f ms. %s %ld  %s %ld\n",                  \
                group_id, block_counter,                                                          \
                sobel_cycles, sobel_usec, (sobel_usec * 1E-3),                                    \
                pmc_names[group_id*2 + 0], pmc_values[0],                                         \
                pmc_names[group_id*2 + 1], pmc_values[1]);                                        \
                                                                                                  \
        PAPI_cleanup_eventset(papi_evenSet);                                                      \
        PAPI_destroy_eventset(&papi_evenSet);                                                     \
    }                                                                                             \
                                                                                                  \
    /* ===================================================================== */                   \
    /* End of compute, fence all outstanding put                             */                   \
    /* ===================================================================== */                   \
    async_work_group_copy_fence(CLK_GLOBAL_MEM_FENCE);                                            \
}

OCL_KERNEL_DMA_TILING_ENGINE_SOBEL_PAPI(sobel_step_4_PAPI, sobel_compute_block_step_4)



// ============================================================================
// Step 5: Compute_block optimization
//
// - Now the Step 4 has been instrumented. You should notice that, depending on
//   the input image size, the compute_block() function may take from
//   80% to 98% of kernel execution time.
// - It's now time to perform some optimization on it:
//   + Change compute type from float to short
//   + Vectorization by manual loop-unrolling
//   + You can also optimize the compute_block() function in a separate C or
//     assembly kernel and plug it into this OpenCL via the MPPA Native
//     extension (see Kalray OpenCL manual)
// ============================================================================

static void sobel_compute_block_step_5(__local uchar *block_in_local,
                                       __local uchar *block_out_local,
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
        int icol = 0;

        // take care to not exceed 'block_width_halo' when doing
        // vectorized neighbor reads
        for (; icol + 8 + HALO_SIZE <= block_width_halo; icol += 8)
        {
            // load neighbors
            short8 c0 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+0, icol+0)]));
            short8 c1 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+0, icol+1)]));
            short8 c2 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+0, icol+2)]));

            short8 n0 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+1, icol+0)]));
            short8 n2 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+1, icol+2)]));

            short8 t0 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+2, icol+0)]));
            short8 t1 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+2, icol+1)]));
            short8 t2 = convert_short8(vload8(0, &block_in_local[BLOCK_IN_INDEX(irow+2, icol+2)]));

            // compute
            float8 magx = convert_float8(((short)2 * (n2 - n0)) + (c2 - c0 + t2 - t0));
            float8 magy = convert_float8(((short)2 * (t1 - c1)) + (t0 - c0 + t2 - c2));
            float8 mag  = hypot(magx, magy) * scale;

            // store pixel
            vstore8(convert_uchar8_sat(mag), 0, &block_out_local[BLOCK_OUT_INDEX(irow, icol)]);
        }

        for (; icol + 4 + HALO_SIZE <= block_width_halo; icol += 4)
        {
            // load neighbors
            short4 c0 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+0, icol+0)]));
            short4 c1 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+0, icol+1)]));
            short4 c2 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+0, icol+2)]));

            short4 n0 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+1, icol+0)]));
            short4 n2 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+1, icol+2)]));

            short4 t0 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+2, icol+0)]));
            short4 t1 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+2, icol+1)]));
            short4 t2 = convert_short4(vload4(0, &block_in_local[BLOCK_IN_INDEX(irow+2, icol+2)]));

            // compute
            float4 magx = convert_float4(((short)2 * (n2 - n0)) + (c2 - c0 + t2 - t0));
            float4 magy = convert_float4(((short)2 * (t1 - c1)) + (t0 - c0 + t2 - c2));
            float4 mag  = hypot(magx, magy) * scale;

            // store pixel
            vstore4(convert_uchar4_sat(mag), 0, &block_out_local[BLOCK_OUT_INDEX(irow, icol)]);
        }

        for (; icol < block_width; icol++)
        {
            // load neighbors
            short c0 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+0)];
            short c1 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+1)];
            short c2 = block_in_local[BLOCK_IN_INDEX(irow+0, icol+2)];

            short n0 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+0)];
            short n2 = block_in_local[BLOCK_IN_INDEX(irow+1, icol+2)];

            short t0 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+0)];
            short t1 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+1)];
            short t2 = block_in_local[BLOCK_IN_INDEX(irow+2, icol+2)];

            // compute
            float magx = ((short)2 * (n2 - n0)) + (c2 - c0 + t2 - t0);
            float magy = ((short)2 * (t1 - c1)) + (t0 - c0 + t2 - c2);
            float mag  = hypot(magx, magy) * scale;

            // store pixel
            block_out_local[BLOCK_OUT_INDEX(irow, icol)] = convert_uchar_sat(mag);
        }
    }

    // sync to gather result from all WI
    barrier(CLK_LOCAL_MEM_FENCE);
}

OCL_KERNEL_DMA_TILING_ENGINE_SOBEL(sobel_step_5, sobel_compute_block_step_5)
