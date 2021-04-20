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
