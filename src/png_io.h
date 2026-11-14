#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int fusion_read_png(const char* path, unsigned char** pixels, size_t* width,
                    size_t* height, char* error, size_t error_size);
int fusion_write_png(const char* path, const unsigned char* pixels, size_t width,
                     size_t height, char* error, size_t error_size);

#ifdef __cplusplus
}
#endif
