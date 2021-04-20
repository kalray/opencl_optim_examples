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
 * @file ocl_utils.c
 *
 * Some OpenCL helper functions on Host
 *
 * @author Minh Quan HO <mqho@kalrayinc.com>
 *
 ******************************************************************************
 */

#include <stdlib.h>
#include "ocl_utils.h"

#define BUILD_LOG_LENGTH (16*1024)

cl_program ocl_CreateProgramFromBinary(cl_context context,
                                       cl_device_id device,
                                       const char* binary_path)
{
    cl_int err = 0;
    cl_int binary_status = CL_SUCCESS;
    cl_program program = NULL;

    FILE *fp = fopen(binary_path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening %s\n", binary_path);
        goto quit;
    }

    printf("File opened %s\n", binary_path);

    // get the size of binary file
    size_t binary_size;
    fseek(fp, 0, SEEK_END);
    binary_size = ftell(fp);
    rewind(fp);

    unsigned char *program_binary = malloc(binary_size * sizeof(unsigned char));
    if (program_binary == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes for program_binary\n",
            binary_size);
        goto quit;
    }

    size_t read_size = fread(program_binary, 1, binary_size, fp);
    fclose(fp);
    if (read_size != binary_size) { err = CL_INVALID_BINARY; }
    OCL_CHECK_ERROR_QUIT(err, "Failed to read binary %s", binary_path);

    program = clCreateProgramWithBinary(context,
                1,
                &device,
                &binary_size,
                (const unsigned char**)&program_binary,
                &binary_status,
                &err);

    free(program_binary);

    OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateProgramWithBinary");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        char *build_log = malloc(BUILD_LOG_LENGTH * sizeof(char));

        // get the failure reason
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                BUILD_LOG_LENGTH, build_log, NULL);

        OCL_CHECK_ERROR_QUIT(err, "Program build failed: %s", build_log);

        clReleaseProgram(program);

        free(build_log);
        goto quit;
    }

quit:
    return program;
}
