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
 * @file host.c
 *
 * Main program on Host
 *
 * @author Minh Quan HO <mqho@kalrayinc.com>
 *
 ******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include <CL/cl.h>

#include "ocl_utils.h"
#include "png_utils.h"
#include "sobel_config.h"

int main(int argc, char* argv[])
{
    int ret = 0;

    // ===================================================================
    // Argument processing. Keep it simple stupid
    // ===================================================================
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_PNG_image> [sobel_scale]\n", argv[0]);
        ret = -1;
        goto quit;
    }

    const char *image_filename = argv[1];
    const float sobel_scale    = (argc >= 3) ? (float)atof(argv[2]) : DEFAULT_SOBEL_SCALE;

    // ===================================================================
    // Read input PNG image given by user
    // ===================================================================
    png_image_t img;
    const bool convert_to_gray = true;
    printf("[HOST] Reading input PNG %s %s\n", image_filename, convert_to_gray ? "and convert to GRAY" : "");
    read_png_file(image_filename, &img, convert_to_gray);

    // image information
    const int    image_width  = img.w;
    const int    image_height = img.h;
    const size_t image_size   = image_width * image_height * sizeof(unsigned char);

    // ===================================================================
    // OpenCL stuffs
    // ===================================================================
    cl_int err = CL_SUCCESS;

    cl_platform_id   platform;         // OpenCL platform
    cl_device_id     device_id;        // device ID
    cl_context       context;          // context
    cl_command_queue queue;            // command queue
    cl_program       program;          // program
    cl_mem           ocl_image_input;  // Device buffer for input image
    size_t           max_workgroup_size; // CL_DEVICE_MAX_WORK_GROUP_SIZE

    // ===================================================================
    // Device detection
    // ===================================================================
    err = clGetPlatformIDs(1, &platform, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetDeviceIDs");

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(max_workgroup_size), &max_workgroup_size, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)");

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    assert(context);

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    assert(queue);

    // ===================================================================
    // List of kernels
    // ===================================================================
    struct kernel_desc_s {
        const char    *name;
        size_t        globalSize[2];
        size_t        localSize[2];
        // additional fields to handle multi-kernels
        cl_kernel     ocl_kernel;
        cl_mem        ocl_image_output;
        unsigned char *host_image_output;
        cl_event      ocl_event[2];
        float         host_elapsed_ms[2];
        float         device_elapsed_ms[2];
        bool          ocl_have_native_kernel;
    } kernel_desc[] = {
        {
            .name       = "sobel_step_0",
            .globalSize = {ALIGN_MULT_UP(image_width, max_workgroup_size), image_height},
            .localSize  = {max_workgroup_size, 1},
        },
        {
            .name       = "sobel_step_1",
            .globalSize = {(ceil(((double)image_width)  / TILE_WIDTH))  * max_workgroup_size,
                           (ceil(((double)image_height) / TILE_HEIGHT)) * 1},
            .localSize  = {max_workgroup_size, 1},
        },
        {
            .name       = "sobel_step_2",
            .globalSize = {(ceil(((double)image_width)  / TILE_WIDTH))  * max_workgroup_size,
                           (ceil(((double)image_height) / TILE_HEIGHT)) * 1},
            .localSize  = {max_workgroup_size, 1},
        },
    };

    const int nb_kernels = sizeof(kernel_desc) / sizeof(kernel_desc[0]);
    assert(nb_kernels > 0 && "No kernel available");

    // ===================================================================
    // Buffer creation, Program creation & Kernel arguments
    // ===================================================================
    // Create READ-ONLY input image
    ocl_image_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     image_size, img.row_pointers[0], &err);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateBuffer for input image");

    // free allocated buffer for image reading (img.row_pointers[])
    free_img_row_pointers(&img);

    // Create program
    program = ocl_CreateProgramFromBinary(context, device_id, "output/opencl_kernels/sobel.cl.pocl");
    assert(program);

    // From program, create all kernels
    for (int i = 0; i < nb_kernels; i++)
    {
        // create output buffer for each kernel, on Host and Device
        kernel_desc[i].host_image_output = (unsigned char *)malloc(image_size * sizeof(unsigned char));
        assert(kernel_desc[i].host_image_output && "Failed to allocate host_image_output");

        kernel_desc[i].ocl_image_output  = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                          image_size, NULL, &err);
        OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateBuffer for %d-th output image", i);

        // create kernel
        kernel_desc[i].ocl_kernel = clCreateKernel(program, kernel_desc[i].name, &err);
        OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateKernel %s", kernel_desc[i].name);

        // set arguments
        cl_uint nb_arguments = 0;
        err  = clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++, sizeof(cl_mem), &ocl_image_input);
        err |= clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++, sizeof(cl_mem), &kernel_desc[i].ocl_image_output);
        err |= clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++, sizeof(cl_int), &image_width);
        err |= clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++, sizeof(cl_int), &image_height);
        err |= clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++, sizeof(float),  &sobel_scale);
        OCL_CHECK_ERROR_QUIT(err, "Failed to clSetKernelArg %s", kernel_desc[i].name);
    }

    // ===================================================================
    // Run all kernels
    // ===================================================================
    for (int i = 0; i < nb_kernels; i++)
    {
        for (int hot = 0; hot < 2; hot++)
        {
            // time_spent on host
            struct timespec host_start, host_end;
            clock_gettime(CLOCK_MONOTONIC, &host_start);

            err = clEnqueueNDRangeKernel(queue, kernel_desc[i].ocl_kernel, 2, NULL,
                                         kernel_desc[i].globalSize, kernel_desc[i].localSize,
                                         0, NULL, &kernel_desc[i].ocl_event[hot]);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clEnqueueNDRangeKernel %s", kernel_desc[i].name);

            cl_event event_read;
            err = clEnqueueReadBuffer(queue, kernel_desc[i].ocl_image_output, CL_FALSE, 0,
                                      image_size, kernel_desc[i].host_image_output,
                                      1, &kernel_desc[i].ocl_event[hot], &event_read);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clEnqueueReadBuffer %s", kernel_desc[i].name);

            err = clWaitForEvents(1, &event_read);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clWaitForEvents kernel %s", kernel_desc[i].name);

            clock_gettime(CLOCK_MONOTONIC, &host_end);

            err = clReleaseEvent(event_read);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clReleaseEvent kernel %s", kernel_desc[i].name);

            // ----------------------------------------------------------------
            // get profiling info
            // ----------------------------------------------------------------
            // time_spent on Host
            const double host_elapsed_ns = (host_end.tv_sec - host_start.tv_sec) * 1E9 +
                                           (host_end.tv_nsec - host_start.tv_nsec);
            kernel_desc[i].host_elapsed_ms[hot] = (float)(host_elapsed_ns * 1E-6);

            // time_spent on Device
            cl_ulong start = 0;
            cl_ulong end   = 0;

            err = clGetEventProfilingInfo(kernel_desc[i].ocl_event[hot],
                                          CL_PROFILING_COMMAND_START,
                                          sizeof(cl_ulong), &start, NULL);
            OCL_CHECK_ERROR_QUIT(err, "Failed to get CL_PROFILING_COMMAND_START of kernel %s",
                                 kernel_desc[i].name);

            err = clGetEventProfilingInfo(kernel_desc[i].ocl_event[hot],
                                          CL_PROFILING_COMMAND_END,
                                          sizeof(cl_ulong), &end, NULL);
            OCL_CHECK_ERROR_QUIT(err, "Failed to get CL_PROFILING_COMMAND_END of kernel %s",
                                 kernel_desc[i].name);

            kernel_desc[i].device_elapsed_ms[hot] = (double)(end - start) * 1E-06;
        }
    }

    for (int i = 0; i < nb_kernels; i++)
    {
        // correctness check against the step-0 reference kernel
        bool passed = true;
        if (i > 0) {
            passed = (0 == memcmp(kernel_desc[i].host_image_output,
                                  kernel_desc[0].host_image_output,
                                  image_size));
        }

        if (kernel_desc[i].ocl_have_native_kernel) {
            printf("[HOST] Kernel %19s(): Host cold %6.3f ms hot %6.3f ms"
                   " - Device cold %6.3f ms hot %6.3f ms"
                   " - Speedup vs. Step-0 %5.2f  %s (HAVE_FAST_MATH = %d)\n",
                   kernel_desc[i].name,
                   kernel_desc[i].host_elapsed_ms[0], kernel_desc[i].host_elapsed_ms[1],
                   kernel_desc[i].device_elapsed_ms[0], kernel_desc[i].device_elapsed_ms[1],
                   (kernel_desc[0].device_elapsed_ms[1] / kernel_desc[i].device_elapsed_ms[1]),
                   passed ? "[PASSED]" : "[FAILED]", HAVE_FAST_MATH);
        } else {
            printf("[HOST] Kernel %19s(): Host cold %6.3f ms hot %6.3f ms"
                   " - Device cold %6.3f ms hot %6.3f ms"
                   " - Speedup vs. Step-0 %5.2f  %s\n",
                   kernel_desc[i].name,
                   kernel_desc[i].host_elapsed_ms[0], kernel_desc[i].host_elapsed_ms[1],
                   kernel_desc[i].device_elapsed_ms[0], kernel_desc[i].device_elapsed_ms[1],
                   (kernel_desc[0].device_elapsed_ms[1] / kernel_desc[i].device_elapsed_ms[1]),
                   passed ? "[PASSED]" : "[FAILED]");
        }

        ret |= !passed;
    }

#if PERF_TRACKING
    // Generate Cheetah entries
    for (int i = 0; i < nb_kernels; i++)
    {
        printf("#HOST_MPPA_OCL_optim_%s=%.3f time_ms\n",   kernel_desc[i].name,
               kernel_desc[i].host_elapsed_ms[1]);

        printf("#KERNEL_MPPA_OCL_optim_%s_FAST_MATH_%d=%.3f time_ms\n", kernel_desc[i].name,
               HAVE_FAST_MATH, kernel_desc[i].device_elapsed_ms[1]);

        printf("#KERNEL_MPPA_OCL_optim_%s_FAST_MATH_%d=%.3f speedup\n", kernel_desc[i].name,
               HAVE_FAST_MATH,
               (kernel_desc[0].device_elapsed_ms[1] / kernel_desc[i].device_elapsed_ms[1]));
    }
#endif  // PERF_TRACKING

    // ===================================================================
    // Write output images to disk
    // ===================================================================
#if OUTPUT_IMAGE_TO_DISK
    #define MAX_FILENAME_LENGTH 1024
    static char output_image_name[MAX_FILENAME_LENGTH];
    for (int i = 0; i < nb_kernels; i++)
    {
        snprintf(output_image_name, MAX_FILENAME_LENGTH, "%s.%s.png",
                 image_filename, kernel_desc[i].name);

        printf("[HOST] Writing output PNG %s\n", output_image_name);
        write_png_file_attached_buffer(output_image_name, &img,
                                       kernel_desc[i].host_image_output);

        // display
        #if OUTPUT_DISPLAY
        static char cmd[MAX_FILENAME_LENGTH];
        snprintf(cmd, MAX_FILENAME_LENGTH, "display %s &", output_image_name);
        err = system(cmd);
        if (err) {
            printf("[HOST] Impossible to launch command %s\n", cmd);
            exit(1);
        }
        #endif // OUTPUT_DISPLAY
    }
#endif  // OUTPUT_IMAGE_TO_DISK

    // ===================================================================
    // Cleanup
    // ===================================================================
    clReleaseMemObject(ocl_image_input);

    for (int i = 0; i < nb_kernels; i++)
    {
        clReleaseMemObject(kernel_desc[i].ocl_image_output);
        clReleaseKernel(kernel_desc[i].ocl_kernel);
        clReleaseEvent(kernel_desc[i].ocl_event[0]);
        clReleaseEvent(kernel_desc[i].ocl_event[1]);
        free(kernel_desc[i].host_image_output);
    }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device_id);

quit:
    return ret;
}
