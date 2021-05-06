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
 * @file ocl_utils.h
 *
 * Some OpenCL helper functions on Host
 *
 * @author Minh Quan HO <mqho@kalrayinc.com>
 *
 ******************************************************************************
 */

#include <stdio.h>
#include <CL/cl.h>

#define OCL_CHECK_ERROR(err, fmt, ...)                                     \
do {                                                                       \
    if (CL_SUCCESS != err) {                                               \
        fprintf(stderr, "%s():%d: error %2d - " fmt "\n", __FUNCTION__,    \
                __LINE__, err, ##__VA_ARGS__);                             \
    }                                                                      \
} while (0)

#define OCL_CHECK_ERROR_QUIT(err, fmt, ...)                                \
do {                                                                       \
    if (CL_SUCCESS != err) {                                               \
        fprintf(stderr, "%s():%d: error %2d - " fmt "\n", __FUNCTION__,    \
                __LINE__, err, ##__VA_ARGS__);                             \
        goto quit;                                                         \
    }                                                                      \
} while (0)

#define ALIGN_MULT_UP(a, b) (((a+b-1) / b) * b)

/**
 * @brief      Create a cl_program from given binary
 *
 * @param[in]  context      The context
 * @param[in]  device       The device
 * @param[in]  binary_path  The binary path
 *
 * @return     Created cl_program
 */
cl_program ocl_CreateProgramFromBinary(cl_context context,
                                       cl_device_id device,
                                       const char* binary_path);

size_t memdiff(const unsigned char *ref, const unsigned char *buf, const size_t size,
               const int tolerance);
