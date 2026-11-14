#include "png_io.h"

#include <png.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct png_state {
    FILE* file;
    png_structp png;
    png_infop info;
    unsigned char* pixels;
    png_bytep* rows;
    char* error;
    size_t error_size;
};

static void report_error(struct png_state* state, const char* message) {
    if (state->error_size > 0) {
        snprintf(state->error, state->error_size, "%s", message);
    }
}

static void png_error_callback(png_structp png, png_const_charp message) {
    struct png_state* state = (struct png_state*)png_get_error_ptr(png);
    report_error(state, message);
    png_longjmp(png, 1);
}

static void cleanup_read(struct png_state* state) {
    if (state->png) png_destroy_read_struct(&state->png, state->info ? &state->info : NULL, NULL);
    if (state->file) fclose(state->file);
    free(state->rows);
    free(state->pixels);
    free(state);
}

int fusion_read_png(const char* path, unsigned char** pixels, size_t* width,
                    size_t* height, char* error, size_t error_size) {
    struct png_state* state = (struct png_state*)calloc(1, sizeof(*state));
    if (!state) {
        if (error_size > 0) snprintf(error, error_size, "Out of memory");
        return 0;
    }
    state->error = error;
    state->error_size = error_size;
    state->file = fopen(path, "rb");
    if (!state->file) {
        report_error(state, "Cannot open PNG for reading");
        cleanup_read(state);
        return 0;
    }
    state->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, state, png_error_callback, NULL);
    if (!state->png) {
        report_error(state, "Cannot create PNG reader");
        cleanup_read(state);
        return 0;
    }
    state->info = png_create_info_struct(state->png);
    if (!state->info) {
        report_error(state, "Cannot create PNG info");
        cleanup_read(state);
        return 0;
    }
    if (setjmp(png_jmpbuf(state->png))) {
        cleanup_read(state);
        return 0;
    }
    png_init_io(state->png, state->file);
    png_read_info(state->png, state->info);
    const png_uint_32 w = png_get_image_width(state->png, state->info);
    const png_uint_32 h = png_get_image_height(state->png, state->info);
    if (png_get_bit_depth(state->png, state->info) != 8 ||
        png_get_color_type(state->png, state->info) != PNG_COLOR_TYPE_RGB ||
        png_get_valid(state->png, state->info, PNG_INFO_tRNS)) {
        png_error(state->png, "Only 8-bit RGB PNG images are supported");
    }
    if (w == 0 || h == 0 || (size_t)w > SIZE_MAX / 3 ||
        (size_t)h > SIZE_MAX / ((size_t)w * 3) ||
        (size_t)h > SIZE_MAX / sizeof(*state->rows)) {
        png_error(state->png, "Invalid PNG dimensions");
    }
    png_set_interlace_handling(state->png);
    png_read_update_info(state->png, state->info);
    if (png_get_rowbytes(state->png, state->info) != (size_t)w * 3) {
        png_error(state->png, "Unexpected PNG row size");
    }
    state->pixels = (unsigned char*)malloc((size_t)w * h * 3);
    state->rows = (png_bytep*)malloc((size_t)h * sizeof(*state->rows));
    if (!state->pixels || !state->rows) png_error(state->png, "Out of memory");
    for (size_t y = 0; y < h; ++y) state->rows[y] = state->pixels + y * (size_t)w * 3;
    png_read_image(state->png, state->rows);
    png_read_end(state->png, state->info);
    *pixels = state->pixels;
    *width = w;
    *height = h;
    state->pixels = NULL;
    cleanup_read(state);
    return 1;
}

static void cleanup_write(struct png_state* state) {
    if (state->png) png_destroy_write_struct(&state->png, state->info ? &state->info : NULL);
    if (state->file) fclose(state->file);
    free(state->rows);
    free(state);
}

int fusion_write_png(const char* path, const unsigned char* pixels, size_t width,
                     size_t height, char* error, size_t error_size) {
    struct png_state* state = (struct png_state*)calloc(1, sizeof(*state));
    if (!state) {
        if (error_size > 0) snprintf(error, error_size, "Out of memory");
        return 0;
    }
    state->error = error;
    state->error_size = error_size;
    if (width == 0 || height == 0 || width > PNG_UINT_31_MAX || height > PNG_UINT_31_MAX ||
        width > SIZE_MAX / 3 || height > SIZE_MAX / (width * 3) ||
        height > SIZE_MAX / sizeof(*state->rows)) {
        report_error(state, "Invalid PNG dimensions");
        cleanup_write(state);
        return 0;
    }
    state->file = fopen(path, "wb");
    if (!state->file) {
        report_error(state, "Cannot open PNG for writing");
        cleanup_write(state);
        return 0;
    }
    state->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, state, png_error_callback, NULL);
    if (!state->png) {
        report_error(state, "Cannot create PNG writer");
        cleanup_write(state);
        return 0;
    }
    state->info = png_create_info_struct(state->png);
    if (!state->info) {
        report_error(state, "Cannot create PNG info");
        cleanup_write(state);
        return 0;
    }
    if (setjmp(png_jmpbuf(state->png))) {
        cleanup_write(state);
        return 0;
    }
    png_init_io(state->png, state->file);
    png_set_IHDR(state->png, state->info, (png_uint_32)width, (png_uint_32)height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(state->png, state->info);
    state->rows = (png_bytep*)malloc(height * sizeof(*state->rows));
    if (!state->rows) png_error(state->png, "Out of memory");
    for (size_t y = 0; y < height; ++y) {
        state->rows[y] = (png_bytep)(pixels + y * width * 3);
    }
    png_write_image(state->png, state->rows);
    png_write_end(state->png, state->info);
    if (fclose(state->file) != 0) {
        state->file = NULL;
        report_error(state, "Cannot finish writing PNG");
        cleanup_write(state);
        return 0;
    }
    state->file = NULL;
    cleanup_write(state);
    return 1;
}
