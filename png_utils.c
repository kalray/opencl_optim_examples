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
 * @file png_utils.c
 *
 * Some libPNG helper functions
 *
 ******************************************************************************
 */

/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#include "png_utils.h"

#define IMAGE_ALLOC_CONTIGUOUS 1

void read_png_file(const char *file_name, png_image_t *img, bool convert_to_gray)
{
    char header[8];        // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        abort_("[read_png_file] File %s could not be opened for reading", file_name);
    }
    if (fread(header, 1, 8, fp) != 8) {
        abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);
    }
    if (png_sig_cmp((png_const_bytep)header, 0, 8)) {
        abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);
    }

    /* initialize stuff */
    img->png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!img->png_ptr) {
        abort_("[read_png_file] png_create_read_struct failed");
    }

    img->info_ptr = png_create_info_struct(img->png_ptr);
    if (!img->info_ptr) {
        abort_("[read_png_file] png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(img->png_ptr))) {
        png_destroy_read_struct(&img->png_ptr, &img->info_ptr, NULL);
        fclose(fp);
        abort_("[read_png_file] Error during init_io");
    }

    png_init_io(img->png_ptr, fp);
    png_set_sig_bytes(img->png_ptr, 8);

    png_read_info(img->png_ptr, img->info_ptr);

    img->w = png_get_image_width(img->png_ptr, img->info_ptr);
    img->h = png_get_image_height(img->png_ptr, img->info_ptr);
    img->color_type = png_get_color_type(img->png_ptr, img->info_ptr);
    img->bit_depth = png_get_bit_depth(img->png_ptr, img->info_ptr);
    img->number_of_passes = png_set_interlace_handling(img->png_ptr);

    /* Convert to GRAY */
    if (convert_to_gray) {
        const int color_type = png_get_color_type(img->png_ptr, img->info_ptr);
        const int bit_depth  = png_get_bit_depth(img->png_ptr, img->info_ptr);
        /* setup to convert every images to 8-bit grayscale */
        if(color_type & PNG_COLOR_TYPE_PALETTE) {
            png_set_palette_to_rgb(img->png_ptr);
        }
        if(bit_depth == 16) {
            png_set_strip_16(img->png_ptr);
        }
        if(color_type & PNG_COLOR_MASK_ALPHA) {
            png_set_strip_alpha(img->png_ptr);
        }
        if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
            png_set_expand_gray_1_2_4_to_8(img->png_ptr);
        }
        if(color_type & PNG_COLOR_MASK_COLOR) {
            const int error_action = 1;
            const int red_weight   = -1;
            const int green_weight = -1;
            png_set_rgb_to_gray_fixed(img->png_ptr, error_action, red_weight, green_weight);
        }

        // re-read info
        png_read_update_info(img->png_ptr, img->info_ptr);
        img->color_type = png_get_color_type(img->png_ptr, img->info_ptr);
        img->bit_depth = png_get_bit_depth(img->png_ptr, img->info_ptr);
        img->number_of_passes = png_set_interlace_handling(img->png_ptr);
    }

    /* allocate image buffer */
    img->row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * img->h);
    if (!img->row_pointers) {
        abort_("[read_png_file] Error allocating the row array");
    }

    #ifdef IMAGE_ALLOC_CONTIGUOUS
    const png_uint_32 rowbytes = png_get_rowbytes(img->png_ptr, img->info_ptr);
    png_byte *image_buffer = (png_byte*) malloc(img->h * rowbytes);
    for (int y = 0; y < img->h; y++)
    {
        img->row_pointers[y] = &image_buffer[y * rowbytes];
    }
    printf("Rowbytes %d - Width %d Height %d\n", rowbytes, img->w, img->h);
    #else
    for (int y = 0; y < img->h; y++)
    {
        img->row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(img->png_ptr, img->info_ptr));
        if (!img->row_pointers[y]) {
            abort_("[read_png_file] Error allocating memory for a row");
        }
    }
    #endif

    png_read_image(img->png_ptr, img->row_pointers);

    png_destroy_read_struct(&img->png_ptr, &img->info_ptr, NULL);
    fclose(fp);

}

void write_png_file(const char *file_name, png_image_t *img)
{
    /* create file */
    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        abort_("[write_png_file] File %s could not be opened for writing", file_name);
    }


    /* initialize stuff */
    img->png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!img->png_ptr) {
        abort_("[write_png_file] png_create_write_struct failed");
    }

    img->info_ptr = png_create_info_struct(img->png_ptr);
    if (!img->info_ptr) {
        abort_("[write_png_file] png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during init_io");
    }

    png_init_io(img->png_ptr, fp);

    /* write header */
    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during writing header");
    }

    png_set_IHDR(img->png_ptr, img->info_ptr, img->w, img->h,
        img->bit_depth, img->color_type, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(img->png_ptr, img->info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during writing bytes");
    }

    png_write_image(img->png_ptr, img->row_pointers);

    /* end write */
    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during end of write");
    }

    png_write_end(img->png_ptr, NULL);

    /* cleanup heap allocation */
    #ifdef IMAGE_ALLOC_CONTIGUOUS
    free(img->row_pointers[0]);
    #else
    for (int y = 0; y < img->h; y++) {
        free(img->row_pointers[y]);
    }
    #endif

    free(img->row_pointers);

    png_destroy_write_struct(&img->png_ptr, &img->info_ptr);

    fclose(fp);
}


void write_png_file_attached_buffer(const char *file_name, png_image_t *img, unsigned char *image_buffer)
{
    /* create file */
    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        abort_("[write_png_file] File %s could not be opened for writing", file_name);
    }

    /* initialize stuff */
    img->png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!img->png_ptr) {
        abort_("[write_png_file] png_create_write_struct failed");
    }

    img->info_ptr = png_create_info_struct(img->png_ptr);
    if (!img->info_ptr) {
        abort_("[write_png_file] png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during init_io");
    }

    png_init_io(img->png_ptr, fp);

    /* write header */
    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during writing header");
    }

    png_set_IHDR(img->png_ptr, img->info_ptr, img->w, img->h,
        img->bit_depth, img->color_type, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(img->png_ptr, img->info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during writing bytes");
    }

    png_bytep *volatile used_row_pointers = img->row_pointers;

    if (image_buffer) {
        const png_uint_32 rowbytes = png_get_rowbytes(img->png_ptr, img->info_ptr);
        used_row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * img->h);
        if (!used_row_pointers) {
            abort_("[write_png_file] Error allocating the row array with attached buffer");
        };
        for (int y = 0; y < img->h; y++)
        {
            used_row_pointers[y] = &image_buffer[y * rowbytes];
        }
    }

    png_write_image(img->png_ptr, used_row_pointers);

    /* end write */
    if (setjmp(png_jmpbuf(img->png_ptr))) {
        abort_("[write_png_file] Error during end of write");
    }

    png_write_end(img->png_ptr, NULL);

    if (image_buffer) {
        free(used_row_pointers);
    } else {
        /* cleanup heap allocation */
        #ifdef IMAGE_ALLOC_CONTIGUOUS
        free(img->row_pointers[0]);
        #else
        for (int y = 0; y < img->h; y++) {
            free(img->row_pointers[y]);
        }
        #endif

        free(img->row_pointers);
    }

    png_destroy_write_struct(&img->png_ptr, &img->info_ptr);

    fclose(fp);
}

void free_img_row_pointers(png_image_t *img)
{
    if (img) {
        /* cleanup heap allocation */
        #ifdef IMAGE_ALLOC_CONTIGUOUS
        free(img->row_pointers[0]);
        #else
        for (int y = 0; y < img->h; y++) {
            free(img->row_pointers[y]);
        }
        #endif

        free(img->row_pointers);
    }
}
