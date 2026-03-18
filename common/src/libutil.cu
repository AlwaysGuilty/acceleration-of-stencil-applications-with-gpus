// General includes

#include <stdio.h>
#include <math.h>
#include <assert.h>

// CUDA related includes

#include <cuda.h>
#include <cuda_runtime.h>

// Custom

#include <libutil.cuh>

/// CLI input sanitization

int sanitize_int(char* arg) {
    int result = atoi(arg);
    // No arg is expected to be negative
    assert((result >= 0) && "Negative values are not allowed");
    return result;
}

/// Error handling

/**
 * Exit with an error msg printed to stdout.
*/
void error_exit(int err_code, const char* msg) {
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, msg);
    exit(err_code);
}

/// Memory handling

/**
 * Free and null a ptr.
*/
void free_null(void* ptr) {
    if (!ptr) return;
    free(ptr);
    ptr = NULL;
}

/**
 * Free CUDA allocated memory and null the ptr.
*/
void cu_free_null(void* ptr) {
    if (!ptr) return;

    cudaError_t retval = cudaFree(ptr);
    assert(retval == cudaSuccess);
    ptr = NULL;
}

/// Debug

/**
 * @param arr 1D array representing 2D world to print
 */
void print_2d(float* arr, int h, int w) {
    // TODO: Test with [1, 2, 3, 4, 5, 6, 7, 8, 9], h=3, w=3
    // Double loop to print as a 2D array
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", arr[i * h + j]);
        }
        printf("\n");
    }
}

/**
 * @param arr 1D array representing 3D world to print
 */
void print_3d(float* arr, int h, int w, int d) {
    // TODO: Test with [1..27], h=3, w=3, d=3
    for (int i = 0; i < h; i++) {
        printf("Layer %d\n", i);
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < d; k++) {
                printf("%f ", arr[i * h + j * w + k]);
            }
            printf("\n");
        }
    }
}

/// DEVICE CODE

__device__ void cu_print_2d(float* arr, int h, int w) {
    // TODO: Test with [1, 2, 3, 4, 5, 6, 7, 8, 9], h=3, w=3
    // Double loop to print as a 2D array
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", arr[i * w + j]);
        }
        printf("\n");
    }
}

__device__ void cu_print_3d(float* arr, int h, int w, int d) {
    // TODO: Test with [1..27], h=3, w=3, d=3
    for (int i = 0; i < h; i++) {
        printf("Layer %d\n", i);
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < d; k++) {
                printf("%f ", arr[i * h + j * w + k]);
            }
            printf("\n");
        }
    }
}

/**
 * 2D convolution kernel for no caching
 *
 * @param world array of cells representing the 2D world
 * @param conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param cell_idxs int2 vector of current cell global indices (x and y) inside the world
 * @param R neighbourhood radius
*/
__inline_hint__ __host__ __device__ float convolve_2d(float* world, float* conv_kernel, dim3 world_dims, int2 cell_idxs, int R) {
    // neighbour global indices
    int n_x;
    int n_y;

    float potential = 0.0f;
    int D = R * 2 + 1;

    // This does convolution with the kernel
    // Iterate over the neighbourhood in a square shape
    #pragma unroll
    for (int off_y = -R; off_y < (R + 1); off_y++) {
        n_y = cell_idxs.y + off_y;
        n_y = (n_y + world_dims.y) & (world_dims.y - 1);  // wrap-around

        #pragma unroll
        for (int off_x = -R; off_x < (R + 1); off_x++) {
            // Get indices in 1d world of the current cell we are looking at
            n_x = cell_idxs.x + off_x;
            n_x = (n_x + world_dims.x) & (world_dims.x - 1);  // wrap-around

            potential += world[world_dims.x * n_y + n_x] * conv_kernel[(off_y + R) * D + (off_x + R)];
        }
    }
    return potential;
}

/**
 * 2D convolution kernel for caching in c2 or c3 modes
 * row major
 *
 * @param cache array of cells representing the 2D cache
 * @param conv_kernel convolution kernel
 * @param cache_dims dim3 vector of cache dim sizes
 * @param cell_idxs int2 vector of current cell indices (x and y) inside the cache
 * @param R neighbourhood radius
*/
__inline_hint__ __host__ __device__ float convolve_2d_cache(float* cache, float* conv_kernel, dim3 cache_dims, int2 cell_idxs, int R) {
    // neighbour global indices
    int n_x;
    int n_y;

    float potential = 0.0f;
    int D = R * 2 + 1;

    // This does convolution with the kernel
    // Iterate over the neighbourhood in a square shape
    #pragma unroll
    for (int off_y = -R; off_y < (R + 1); off_y++) {
        n_y = cell_idxs.y + off_y;

        #pragma unroll
        for (int off_x = -R; off_x < (R + 1); off_x++) {
            // Get indices in 1d cache of the current cell we are looking at
            n_x = cell_idxs.x + off_x;
            // Access the cell value with 1D index
            potential += cache[cache_dims.x * n_y + n_x] * conv_kernel[(off_y + R) * D + (off_x + R)];
            // potential += cache[cache_dims.y * n_x + n_y] * conv_kernel[(off_y + R) * D + (off_x + R)];
        }
    }
    return potential;
}

/**
 * 2D convolution kernel for caching in c2 or c3 modes
 * col major
 *
 * @param cache array of cells representing the 2D cache
 * @param conv_kernel convolution kernel
 * @param cache_dims dim3 vector of cache dim sizes
 * @param cell_idxs int2 vector of current cell indices (x and y) inside the cache
 * @param R neighbourhood radius
*/
__inline_hint__ __host__ __device__ float convolve_2d_cache_col_major(float* cache, float* conv_kernel, dim3 cache_dims, int2 cell_idxs, int R) {
    // neighbour global indices
    int n_x;
    int n_y;

    float potential = 0.0f;
    int D = R * 2 + 1;

    // This does convolution with the kernel
    // Iterate over the neighbourhood in a square shape
    #pragma unroll
    for (int off_x = -R; off_x < (R + 1); off_x++) {
        n_x = cell_idxs.x + off_x;

        #pragma unroll
        for (int off_y = -R; off_y < (R + 1); off_y++) {
            n_y = cell_idxs.y + off_y;
            // Get indices in 1d cache of the current cell we are looking at
            // Access the cell value with 1D index
            // potential += cache[cache_dims.y * n_x + n_y] * conv_kernel[(off_x + R) * D + (off_y + R)];
            potential += cache[cache_dims.y * n_x + n_y] * conv_kernel[(off_y + R) * D + (off_x + R)];
        }
    }
    return potential;
}

/**
 * 3D convolution kernel for no caching
 *
 * @param world 1D array of cells representing the 3D world
 * @param conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param cell_idxs int3 vector of current cell global indices (x, y and z) inside the world
 * @param R neighbourhood radius
*/
__inline_hint__ __host__ __device__ float convolve_3d(float* world, float* conv_kernel, dim3 world_dims, int3 cell_idxs, int R) {
    // neighbour global indices
    int n_x;
    int n_y;
    int n_z;

    float potential = 0.0f;
    int D = R * 2 + 1;

    // This does convolution with the kernel
    #pragma unroll
    for (int off_z = -R; off_z < (R + 1); off_z++) {
        n_z = cell_idxs.z + off_z;
        n_z = (n_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

        #pragma unroll
        for (int off_y = -R; off_y < (R + 1); off_y++) {
            n_y = cell_idxs.y + off_y;
            n_y = (n_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int off_x = -R; off_x < (R + 1); off_x++) {
                // Get indices in 1d world of the current cell we are looking at
                n_x = cell_idxs.x + off_x;
                n_x = (n_x + world_dims.x) & (world_dims.x - 1);  // wrap-around

                // Access the cell value with 1D index
                potential += world[world_dims.x * world_dims.y * n_z + world_dims.x * n_y + n_x]
                    * conv_kernel[D * D * (off_z + R) + (off_y + R) * D + (off_x + R)];
            }
        }
    }
    return potential;
}

/**
 * 3D convolution kernel for c2 and c3 caching
 * row major
 *
 * @param cache 1D array of cells representing the 3D cache
 * @param conv_kernel convolution kernel
 * @param cache_dims dim3 vector of cache dim sizes
 * @param cell_idxs int3 vector of current cell indices (x, y and z) inside the cache
 * @param R neighbourhood radius
*/
__inline_hint__ __host__ __device__ float convolve_3d_cache(float* cache, float* conv_kernel, dim3 cache_dims, int3 cell_idxs, int R) {
    // neighbour global indices
    int n_x;
    int n_y;
    int n_z;

    float potential = 0.0f;
    int D = R * 2 + 1;

    // This does convolution with the kernel
    #pragma unroll
    for (int off_z = -R; off_z < (R + 1); off_z++) {
        n_z = cell_idxs.z + off_z;

        #pragma unroll
        for (int off_y = -R; off_y < (R + 1); off_y++) {
            n_y = cell_idxs.y + off_y;

            #pragma unroll
            for (int off_x = -R; off_x < (R + 1); off_x++) {
                // Get indices in 1d cache of the current cell we are looking at
                n_x = cell_idxs.x + off_x;

                // Access the cell value with 1D index
                potential += cache[cache_dims.x * cache_dims.y * n_z + cache_dims.x * n_y + n_x]
                    * conv_kernel[D * D * (off_z + R) + (off_y + R) * D + (off_x + R)];
            }
        }
    }
    return potential;
}

/**
 * 3D convolution kernel for c2 and c3 caching
 * col major
 *
 * @param cache 1D array of cells representing the 3D cache
 * @param conv_kernel convolution kernel
 * @param cache_dims dim3 vector of cache dim sizes
 * @param cell_idxs int3 vector of current cell indices (x, y and z) inside the cache
 * @param R neighbourhood radius
*/
__inline_hint__ __host__ __device__ float convolve_3d_cache_col_major(float* cache, float* conv_kernel, dim3 cache_dims, int3 cell_idxs, int R) {
    // neighbour global indices
    int n_x;
    int n_y;
    int n_z;

    float potential = 0.0f;
    int D = R * 2 + 1;

    // This does convolution with the kernel
    #pragma unroll
    for (int off_x = -R; off_x < (R + 1); off_x++) {
        n_x = cell_idxs.x + off_x;

        #pragma unroll
        for (int off_y = -R; off_y < (R + 1); off_y++) {
            n_y = cell_idxs.y + off_y;

            #pragma unroll
            for (int off_z = -R; off_z < (R + 1); off_z++) {
                n_z = cell_idxs.z + off_z;

                // Access the cell value with 1D index
                potential += cache[cache_dims.z * cache_dims.y * n_x + cache_dims.z * n_y + n_z]
                    * conv_kernel[D * D * (off_z + R) + (off_y + R) * D + (off_x + R)];
            }
        }
    }
    return potential;
}

/**
 * Loads shared memory with contents of a block+halo from global memory.
 * 2D
 *
 * @param cache shared memory cache
 * @param world 1d repr of 2d world
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 */
__device__ __inline_hint__ void load_cache_2d_c2(float* cache, float* world, dim3 world_dims, int R) {
    int curr_cell_x;
    int curr_cell_y;
    int curr_index_1D = blockDim.x * threadIdx.y + threadIdx.x;
    int cache_start_x = blockIdx.x * blockDim.x - R;
    int cache_start_y = blockIdx.y * blockDim.y - R;
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.y + 2 * R);
    // int n_cache_cells = cache_dims.x * cache_dims.y;
    // int N = n_cache_cells + n_cache_cells % (blockDim.x * blockDim.y);

    // Problem: this causes branching within warps
    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y) {
        curr_cell_y = cache_start_y + curr_index_1D / cache_dims.x;
        curr_cell_x = cache_start_x + curr_index_1D % cache_dims.x;

        // wrap-around, assume world dims are powers of 2
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[world_dims.x * curr_cell_y + curr_cell_x];

        // jump to the next cell
        curr_index_1D += blockDim.y * blockDim.x;
    }
}

__inline_hint__ __device__ void load_cache_2d_c2_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.y + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.y) {
        curr_cell_y = blockIdx.y * blockDim.y - R + y;
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

        #pragma unroll
        for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
            curr_cell_x = blockIdx.x * blockDim.x - R + x;
            curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

            // Use read-only data cache where available for slightly better throughput
            // cache[y * cache_dims.x + x] = __ldg(&world[world_dims.x * curr_cell_y + curr_cell_x]);
            cache[y * cache_dims.x + x] = world[world_dims.x * curr_cell_y + curr_cell_x];
        }
    }
}

__inline_hint__ __device__ void load_cache_2d_c2_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.y + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.y) {
            curr_cell_y = blockIdx.y * blockDim.y - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            // cache[x * cache_dims.y + y] = world[world_dims.y * curr_cell_x + curr_cell_y];
            cache[x * cache_dims.y + y] = world[world_dims.x * curr_cell_y + curr_cell_x];
        }
    }
}

/**
 * Cache size is (blockDim.x + 2*R) ^ 2
 */
__inline_hint__ __device__ void load_cache_2d_c2_p1(float* cache, float* world, dim3 world_dims, int R) {
    int curr_cell_x;
    int curr_cell_y;
    int curr_index_1D = threadIdx.x;
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R);

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y) {
        curr_cell_y = blockIdx.y * blockDim.x - R + curr_index_1D / cache_dims.x;
        curr_cell_x = blockIdx.x * blockDim.x - R + curr_index_1D % cache_dims.x;

        // wrap-around
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[world_dims.x * curr_cell_y + curr_cell_x];

        // jump to the next cell
        curr_index_1D += blockDim.x;
    }
}

/**
 * Row majow
 *
 * Cache size is (blockDim.x + 2*R) ^ 2
 */
__inline_hint__ __device__ void load_cache_2d_c2_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int y = 0; y < cache_dims.y; y++) {
        curr_cell_y = blockIdx.y * blockDim.x - R + y;
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

        #pragma unroll
        for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
            curr_cell_x = blockIdx.x * blockDim.x - R + x;
            curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

            cache[y * cache_dims.x + x] = world[world_dims.x * curr_cell_y + curr_cell_x];
        }
    }
}

/**
 * Col majow
 *
 * Cache size is (blockDim.x + 2*R) ^ 2
 */
__inline_hint__ __device__ void load_cache_2d_c2_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    // for (int x = 0; x < cache_dims.x; x++) {
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        // for (int y = threadIdx.x; y < cache_dims.y; y += blockDim.x) {
        for (int y = 0; y < cache_dims.y; y++) {
            curr_cell_y = blockIdx.y * blockDim.x - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            // cache[y * cache_dims.x + x] = world[world_dims.x * curr_cell_y + curr_cell_x];
            cache[x * cache_dims.y + y] = world[world_dims.x * curr_cell_y + curr_cell_x];
        }
    }
}

/**
 * Cache size is (blockDim.x + 2*R) ^ 3
 */
__inline_hint__ __device__ void load_cache_3d_c2_p1(float* cache, float* world, dim3 world_dims, int R) {
    int curr_cell_x;
    int curr_cell_y;
    int curr_cell_z;
    int curr_index_1D = blockDim.x * threadIdx.y + threadIdx.x;
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R, // assume cube
        blockDim.x + 2 * R
    );
    int remainder;

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y * cache_dims.z) {
        remainder = curr_index_1D % (cache_dims.x * cache_dims.y);

        curr_cell_z = (blockIdx.z * blockDim.x - R) + curr_index_1D / (cache_dims.x * cache_dims.y);
        curr_cell_y = (blockIdx.y * blockDim.x - R) + remainder / cache_dims.x;
        curr_cell_x = (blockIdx.x * blockDim.x - R) + remainder % cache_dims.x;

        // Wrap around
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[
            world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
        ];

        // jump to the next cell
        curr_index_1D += blockDim.x * blockDim.x;
    }
}

/**
 * Cache size is (blockDim.x + 2*R) ^ 3
 * row major
 */
__inline_hint__ __device__ void load_cache_3d_c2_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R, blockDim.x + 2 * R);

    int curr_cell_z;
    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int z = 0; z < cache_dims.z; z++) {
        curr_cell_z = blockIdx.z * blockDim.x - R + z;
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.x) {
            curr_cell_y = blockIdx.y * blockDim.x - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
                curr_cell_x = blockIdx.x * blockDim.x - R + x;
                curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

                cache[cache_dims.x * cache_dims.y * z + y * cache_dims.x + x] = world[
                    world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
                ];
            }
        }
    }
}

/**
 * Cache size is (blockDim.x + 2*R) ^ 3
 * col major
 */
__inline_hint__ __device__ void load_cache_3d_c2_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R, blockDim.x + 2 * R);

    int curr_cell_z;
    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.x) {
            curr_cell_y = blockIdx.y * blockDim.x - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int z = 0; z < cache_dims.z; z++) {
                curr_cell_z = blockIdx.z * blockDim.x - R + z;
                curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

                cache[cache_dims.z * cache_dims.y * x + y * cache_dims.z + z] = world[
                    world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
                ];
            }
        }
    }
}

/**
 * Cache size is (blockDim.x + 2*R) * (1 + 2*R)
 */
__inline_hint__ __device__ void preload_cache_2d_c3_p1(float* cache, float* world, dim3 world_dims, int R) {
    int curr_cell_x;
    int curr_cell_y;
    int curr_index_1D = threadIdx.x;
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y) {
        curr_cell_y = blockIdx.y * blockDim.x - R - 1 + curr_index_1D / cache_dims.x;
        curr_cell_x = blockIdx.x * blockDim.x - R + curr_index_1D % cache_dims.x;

        // wrap-around
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[world_dims.x * curr_cell_y + curr_cell_x];

        // jump to the next cell
        curr_index_1D += blockDim.x;
    }
}

/**
 * Row major
 *
 * Cache size is (blockDim.x + 2*R) * (1 + 2*R)
 */
__inline_hint__ __device__ void preload_cache_2d_c3_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int y = 0; y < cache_dims.y; y++) {
        curr_cell_y = blockIdx.y * blockDim.x - R - 1 + y;
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

        #pragma unroll
        for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
            curr_cell_x = blockIdx.x * blockDim.x - R + x;
            curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

            cache[y * cache_dims.x + x] = world[world_dims.x * curr_cell_y + curr_cell_x];
        }
    }
}

/**
 * Col major
 *
 * Cache size is (blockDim.x + 2*R) * (1 + 2*R)
 */
__inline_hint__ __device__ void preload_cache_2d_c3_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    // for (int x = 0; x < cache_dims.x; x++) {
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        // for (int y = threadIdx.x; y < cache_dims.y; y += blockDim.x) {
        for (int y = 0; y < cache_dims.y; y++) {
            curr_cell_y = blockIdx.y * blockDim.x - R - 1 + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            // cache[y * cache_dims.x + x] = world[world_dims.x * curr_cell_y + curr_cell_x];
            cache[x * cache_dims.y + y] = world[world_dims.x * curr_cell_y + curr_cell_x];
        }
    }
}

/**
 * Cache size is (blockDim.x + 2*R) * (blockDim.x + 2*R) * (1 + 2*R)
 */
__inline_hint__ __device__ void preload_cache_3d_c3_p1(float* cache, float* world, dim3 world_dims, int R) {
    int curr_cell_x;
    int curr_cell_y;
    int curr_cell_z;
    int curr_index_1D = blockDim.x * threadIdx.y + threadIdx.x;
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );
    int remainder;

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y * cache_dims.z) {
        remainder = curr_index_1D % (cache_dims.x * cache_dims.y);

        curr_cell_z = (blockIdx.z * blockDim.x - R - 1) + curr_index_1D / (cache_dims.x * cache_dims.y);
        // curr_cell_z = (blockIdx.z * blockDim.x - R) + curr_index_1D / (cache_dims.x * cache_dims.y);
        curr_cell_y = (blockIdx.y * blockDim.x - R) + remainder / cache_dims.x;
        curr_cell_x = (blockIdx.x * blockDim.x - R) + remainder % cache_dims.x;

        // wrap-around
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[
            world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
        ];

        // jump to the next cell
        curr_index_1D += blockDim.x * blockDim.x;
    }
}

/**
 * Cache size is (blockDim.x + 2*R) * (blockDim.x + 2*R) * (1 + 2*R)
 * row major
 */
__inline_hint__ __device__ void preload_cache_3d_c3_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );

    int curr_cell_z;
    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int z = 0; z < cache_dims.z; z++) {
        curr_cell_z = blockIdx.z * blockDim.x - R + z - 1;
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.x) {
            curr_cell_y = blockIdx.y * blockDim.x - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
                curr_cell_x = blockIdx.x * blockDim.x - R + x;
                curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

                cache[cache_dims.x * cache_dims.y * z + y * cache_dims.x + x] = world[
                    world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
                ];
            }
        }
    }
}

/**
 * Cache size is (blockDim.x + 2*R) * (blockDim.x + 2*R) * (1 + 2*R)
 * col major
 */
__inline_hint__ __device__ void preload_cache_3d_c3_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );

    int curr_cell_z;
    int curr_cell_y;
    int curr_cell_x;

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.x) {
            curr_cell_y = blockIdx.y * blockDim.x - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int z = 0; z < cache_dims.z; z++) {
                curr_cell_z = blockIdx.z * blockDim.x - R + z - 1;
                curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

                cache[cache_dims.z * cache_dims.y * x + y * cache_dims.z + z] = world[
                    world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
                ];
            }
        }
    }
}

/**
 * Loads shared memory with contents of a block+halo from global memory.
 * 3D
 *
 * @param cache shared memory cache
 * @param world 1d repr of 3d world
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 */
__inline_hint__ __device__ void load_cache_3d_c2(float* cache, float* world, dim3 world_dims, int R) {
    int l_x = threadIdx.x;
    int l_y = threadIdx.y;
    int l_z = threadIdx.z;
    int curr_index_1D = blockDim.x * blockDim.y * l_z + blockDim.x * l_y + l_x;
    int curr_cell_x;
    int curr_cell_y;
    int curr_cell_z;
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.y + 2 * R,
        blockDim.z + 2 * R
    );
    int cache_start_x = blockIdx.x * blockDim.x - R;
    int cache_start_y = blockIdx.y * blockDim.y - R;
    int cache_start_z = blockIdx.z * blockDim.z - R;
    int remainder;

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y * cache_dims.z) {
        remainder = curr_index_1D % (cache_dims.x * cache_dims.y);

        curr_cell_z = cache_start_z + curr_index_1D / (cache_dims.x * cache_dims.y);
        curr_cell_y = cache_start_y + remainder / cache_dims.x;
        curr_cell_x = cache_start_x + remainder % cache_dims.x;

        // wrap-around
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[
            world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
        ];

        // jump to the next cell
        curr_index_1D += blockDim.x * blockDim.y * blockDim.z;
    }
}

/**
 * Loads shared memory with contents of a block+halo from global memory.
 * 3D
 * Multi-for-loop, row major
 *
 * @param cache shared memory cache
 * @param world 1d repr of 3d world
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 */
__inline_hint__ __device__ void load_cache_3d_c2_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.y + 2 * R,
        blockDim.z + 2 * R
    );

    int curr_cell_x;
    int curr_cell_y;
    int curr_cell_z;

    #pragma unroll
    for (int z = threadIdx.z; z < cache_dims.z; z += blockDim.z) {
        curr_cell_z = blockIdx.z * blockDim.z - R + z;
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.y) {
            curr_cell_y = blockIdx.y * blockDim.y - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
                curr_cell_x = blockIdx.x * blockDim.x - R + x;
                curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

                cache[z * cache_dims.x * cache_dims.y + y * cache_dims.x + x] = world[
                    world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
                ];
            }
        }
    }
}

/**
 * Loads shared memory with contents of a block+halo from global memory.
 * 3D
 * Multi-for-loop, col major
 *
 * @param cache shared memory cache
 * @param world 1d repr of 3d world
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 */
__inline_hint__ __device__ void load_cache_3d_c2_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.y + 2 * R,
        blockDim.z + 2 * R
    );

    int curr_cell_x;
    int curr_cell_y;
    int curr_cell_z;

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.y) {
            curr_cell_y = blockIdx.y * blockDim.y - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            #pragma unroll
            for (int z = threadIdx.z; z < cache_dims.z; z += blockDim.z) {
                curr_cell_z = blockIdx.z * blockDim.z - R + z;
                curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

                cache[cache_dims.z * cache_dims.y * x + y * cache_dims.z + z] = world[
                    world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
                ];
            }
        }
    }
}

/**
 * Move cache up by 1 subdimension (row)
 * 2D
 *
 * @param cache shared memory cache
 * @param world_dims world dims
 * @param R neighbourhood radius
 * @param i row of sub-tile we are in
 */
__inline_hint__ __device__ void shift_subdim_up_2d(float* cache, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);
    int curr_index_1D = threadIdx.x + cache_dims.x;

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y) {
        // Move cell value 1 row up
        cache[curr_index_1D - cache_dims.x] = cache[curr_index_1D];

        curr_index_1D += blockDim.x;
    }
}

/**
 * Row major
 */
__inline_hint__ __device__ void shift_subdim_up_2d_multi_loop_row_major(float* cache, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    #pragma unroll
    for (int y = 1; y < cache_dims.y; y++) {
        #pragma unroll
        for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
            cache[y * cache_dims.x + x - cache_dims.x] = cache[y * cache_dims.x + x];
        }
    }
}

/**
 * Col major
 */
__inline_hint__ __device__ void shift_subdim_up_2d_multi_loop_col_major(float* cache, dim3 world_dims, int R) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    #pragma unroll
    // for (int x = 1; x < cache_dims.x; x++) {
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        #pragma unroll
        // for (int y = threadIdx.x; y < cache_dims.y; y += blockDim.x) {
        for (int y = 1; y < cache_dims.y; y++) {
            // cache[x * cache_dims.y + y - cache_dims.y] = cache[x * cache_dims.y + y];
            cache[x * cache_dims.y + (y - 1)] = cache[x * cache_dims.y + y];
        }
    }
}

/**
 * Move cache up by 1 subdimension (plane)
 * 3D
 *
 * @param cache shared memory cache
 * @param world_dims world dims
 * @param R neighbourhood radius
 * @param i row of sub-tile we are in
 */
__inline_hint__ __device__ void shift_subdim_up_3d(float* cache, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );
    int curr_index_1D = (blockDim.x * threadIdx.y + threadIdx.x) + cache_dims.x * cache_dims.y;

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y * cache_dims.z) {
        // Move cell value 1 plane up
        cache[curr_index_1D - cache_dims.x * cache_dims.y] = cache[curr_index_1D];

        curr_index_1D += blockDim.x * blockDim.x;
    }
}

/**
 * Row major
 */
__inline_hint__ __device__ void shift_subdim_up_3d_multi_loop_row_major(float* cache, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );

    #pragma unroll
    for (int z = 1; z < cache_dims.z; z++) {
        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.y) {
            #pragma unroll
            for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {

                cache[cache_dims.x * cache_dims.y * (z - 1) + y * cache_dims.x + x] = cache[
                    cache_dims.x * cache_dims.y * z + y * cache_dims.x + x
                ];
            }
        }
        // __syncthreads();
    }
}

/**
 * Col major
 */
__inline_hint__ __device__ void shift_subdim_up_3d_multi_loop_col_major(float* cache, dim3 world_dims, int R) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.y) {
            #pragma unroll
            for (int z = 1; z < cache_dims.z; z++) {

                cache[cache_dims.z * cache_dims.y * x + y * cache_dims.z + (z - 1)] = cache[
                    cache_dims.z * cache_dims.y * x + y * cache_dims.z + z
                ];
            }
        }
        // __syncthreads();
    }
}

/**
 * Load new last subdim (row) in the cache
 * 2D
 *
 * @param cache shared memory cache
 * @param world 2D world in 1D array
 * @param world_dims world dims
 * @param R neighbourhood radius
 * @param i row of sub-tile we are in
 */
__inline_hint__ __device__ void load_new_last_subdim_2d(float* cache, float* world, dim3 world_dims, int R, int i) {
    int curr_cell_x;
    int curr_cell_y;
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);
    int curr_index_1D = threadIdx.x + (cache_dims.x * cache_dims.y - cache_dims.x);

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y) {
        curr_cell_y = (blockIdx.y * blockDim.x - R + i) + curr_index_1D / cache_dims.x;
        curr_cell_x = (blockIdx.x * blockDim.x - R) + curr_index_1D % cache_dims.x;

        // wrap-around
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[world_dims.x * curr_cell_y + curr_cell_x];

        // jump to the next cell
        curr_index_1D += blockDim.x;
    }
}

/**
 * Row major
 */
__inline_hint__ __device__ void load_new_last_subdim_2d_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R, int i) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    int y = cache_dims.y - 1;
    curr_cell_y = blockIdx.y * blockDim.x - R + i + y;
    curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        cache[y * cache_dims.x + x] = world[world_dims.x * curr_cell_y + curr_cell_x];
    }
}

/**
 * Col major
 */
__inline_hint__ __device__ void load_new_last_subdim_2d_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R, int i) {
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);

    int curr_cell_y;
    int curr_cell_x;

    int y = cache_dims.y - 1;
    curr_cell_y = blockIdx.y * blockDim.x - R + i + y;
    curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[x * cache_dims.y + y] = world[world_dims.x * curr_cell_y + curr_cell_x];
    }
}

/**
 * Load new last subdim (plane) in the cache
 * 3D
 *
 * @param cache shared memory cache
 * @param world 3D world in 1D array
 * @param world_dims world dims
 * @param R neighbourhood radius
 * @param i row of sub-tile we are in
 */
__inline_hint__ __device__ void load_new_last_subdim_3d(float* cache, float* world, dim3 world_dims, int R, int i) {
    int curr_cell_x;
    int curr_cell_y;
    int curr_cell_z;
    int remainder;
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R, 1 + 2 * R);
    int curr_index_1D = (blockDim.x * threadIdx.y + threadIdx.x) + (cache_dims.x * cache_dims.y * cache_dims.z - cache_dims.x * cache_dims.y);  // move to the end of the cache and then 1 subplane up

    #pragma unroll
    while (curr_index_1D < cache_dims.x * cache_dims.y * cache_dims.z) {
        remainder = curr_index_1D % (cache_dims.x * cache_dims.y);

        curr_cell_z = (blockIdx.z * blockDim.x - R + i) + curr_index_1D / (cache_dims.x * cache_dims.y);
        curr_cell_y = (blockIdx.y * blockDim.x - R) + remainder / cache_dims.x;
        curr_cell_x = (blockIdx.x * blockDim.x - R) + remainder % cache_dims.x;

        // wrap around
        curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);

        cache[curr_index_1D] = world[
            world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
        ];

        // jump to the next cell
        curr_index_1D += blockDim.x * blockDim.x;
    }
}

/**
 * Row major
 */
__inline_hint__ __device__ void load_new_last_subdim_3d_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R, int i) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );

    int curr_cell_z;
    int curr_cell_y;
    int curr_cell_x;

    int z = cache_dims.z - 1;
    curr_cell_z = blockIdx.z * blockDim.x - R + z + i;
    curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

    #pragma unroll
    for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.x) {
        curr_cell_y = blockIdx.y * blockDim.x - R + y;
        curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

        #pragma unroll
        for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
            curr_cell_x = blockIdx.x * blockDim.x - R + x;
            curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

            cache[cache_dims.x * cache_dims.y * z + y * cache_dims.x + x] = world[
                world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
            ];
        }
    }
}

/**
 * Col major
 */
__inline_hint__ __device__ void load_new_last_subdim_3d_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R, int i) {
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.x + 2 * R,
        1 + 2 * R
    );

    int curr_cell_z;
    int curr_cell_y;
    int curr_cell_x;

    int z = cache_dims.z - 1;
    curr_cell_z = blockIdx.z * blockDim.x - R + z + i;
    curr_cell_z = (curr_cell_z + world_dims.z) & (world_dims.z - 1);    // wrap-around

    #pragma unroll
    for (int x = threadIdx.x; x < cache_dims.x; x += blockDim.x) {
        curr_cell_x = blockIdx.x * blockDim.x - R + x;
        curr_cell_x = (curr_cell_x + world_dims.x) & (world_dims.x - 1);    // wrap-around

        #pragma unroll
        for (int y = threadIdx.y; y < cache_dims.y; y += blockDim.x) {
            curr_cell_y = blockIdx.y * blockDim.x - R + y;
            curr_cell_y = (curr_cell_y + world_dims.y) & (world_dims.y - 1);    // wrap-around

            cache[cache_dims.z * cache_dims.y * x + y * cache_dims.z + z] = world[
                world_dims.x * world_dims.y * curr_cell_z + world_dims.x * curr_cell_y + curr_cell_x
            ];
        }
    }
}

/**
 * Exponential growth mapping of the potential
 * For GPU
 *
 * Note: multiplying with inverse value is faster than dividing.
 */
__inline_hint__ __device__ float exponential_growth_mapping(float potential) {
    // float mu = 0.15f;
    // float sigma = 0.015f;
    float mu = 0.3f;
    float sigma = 0.2f;
    float retval = __expf(-((potential - mu) * (potential - mu)) * __frcp_rn(2.0f * sigma * sigma));
    return 2.0f * retval - 1.0f;
}

/**
 * Exponential growth mapping of the potential
 * For CPU.
 * Divided functions, otherwise it is not compilable.
 * __expf and __frcp_rn are only available for GPUs.
 *
 * Note: multiplying with inverse value is faster than dividing.
 */
float exponential_growth_mapping_cpu(float potential) {
    // float mu = 0.15f;
    // float sigma = 0.015f;
    // float mu = 0.2f;
    // float sigma = 0.03f;
    // float mu = 0.2f;
    // float sigma = 0.05f;
    float mu = 0.3f;
    float sigma = 0.2f;
    float retval = expf(-((potential - mu) * (potential - mu)) / (2.0f * sigma * sigma));
    return 2.0f * retval - 1.0f;
}

/**
 * Clip the value to [0, 1]
 * GPU only
 *
 * @param val value to clip
 * @return clipped value
*/
__inline_hint__ __device__ float clip(float val) {
    return fmaxf(fminf(val, 1.0f), 0.0f);
}

/**
 * Clip the value to [0, 1]
 * CPU only
 *
 * @param val value to clip
 * @return clipped value
*/
float clip_cpu(float val) {
    return fmaxf(fminf(val, 1.0f), 0.0f);
}

/// WORLD INITIALIZATION

/**
 * Randomize world with floats in internal [0, 1].
 * Discretization is done by 32bit float precision.
 *
 * Roughly 60% of the cells will be alive.
 *
 * @param world 1D array representing the world
 * @param n number of cells in the world
 */
void randomize_world_discrete(float* world, int n) {
    for (int i = 0; i < n; i++) {
        float value = ((int) rand() / (float) RAND_MAX);
        // leave 80%
        world[i] = value;
        // world[i] = value < 0.90f ? value : 0;
    }
    puts("World initialized.");
}

/// Precomputed convolution kernels for Lenia

/**
 * Exponential 2D kernel
 *
 * @param D kernel diameter in number of automaton cells
 * @param r radius of the kernel
 * @return pointer to 1D array of floats representing the kernel
 */
float* exponential_kernel_2d(int D, int r) {
    float sum = 0.0f;
    float value;
    float col_sqr;
    float row_sqr;
    float l_2_dist;
    int area = D * D;

    float* kernel = (float*) calloc(area, sizeof(float));
    assert(kernel != NULL);

    for (int col = 0; col < D; col++) {
        col_sqr = (col - r) * (col - r);

        for (int row = 0; row < D; row++) {
            row_sqr = (row - r) * (row - r);
            l_2_dist = sqrt(col_sqr + row_sqr) / r;
            value = exp(4 - 1.0f / (l_2_dist * (1 - l_2_dist)));
            // Cap at 1
            if (value <= 1.0f) {
                kernel[col * D + row] = value;
                sum += value;
            }
        }
    }

    // normalize kernel: calc sum of kernel elements and divide each element by the sum
    for (int i = 0; i < area; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

/**
 * Exponential 3D kernel
 *
 * @param D kernel diameter in number of automaton cells
 * @param r radius of the kernel
 * @return pointer to 1D array of floats representing the kernel
 */
float* exponential_kernel_3d(int D, int r) {
    float sum = 0.0f;
    float value;
    float layer_sqr;
    float col_sqr;
    float row_sqr;
    float l_2_dist;
    int layer_area = D * D;
    int volume = D * layer_area;

    float* kernel = (float*) calloc(volume, sizeof(float));
    assert(kernel != NULL);

    for (int layer = 0; layer < D; layer++) {
        layer_sqr = (layer - r) * (layer - r);

        for (int col = 0; col < D; col++) {
            col_sqr = (col - r) * (col - r);

            for (int row = 0; row < D; row++) {
                row_sqr = (row - r) * (row - r);
                l_2_dist = sqrt(layer_sqr + col_sqr + row_sqr) / r;
                value = exp(4 - 1.0f / (l_2_dist * (1 - l_2_dist)));
                // Cap at 1
                if (value <= 1.0f) {
                    kernel[layer * layer_area + col * D + row] = value;
                    sum += value;
                }
            }
        }
    }

    // normalize kernel: calc sum of kernel elements and divide each element by the sum
    for (int i = 0; i < volume; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

/**
 * Rectangular 2D kernel
 *
 * @param D kernel diameter in number of automaton cells
 * @param r radius of the kernel
 * @return pointer to 1D array of floats representing the kernel
 */
float* rectangular_kernel_2d(int D, int r) {
    float* kernel = (float*) calloc(D * D, sizeof(float));
    assert(kernel != NULL);

    float sum = 0.0;
    float value;
    float col_sqr;
    float row_sqr;
    float l_2_dist;

    for (int col = 0; col < D; col++) {
        col_sqr = (col - r) * (col - r);
        for (int row = 0; row < D; row++) {
            row_sqr = (row - r) * (row - r);
            l_2_dist = sqrt(col_sqr + row_sqr) / r;
            value = (l_2_dist >= 0.25 && l_2_dist <= 0.75) ? 1.0 : 0.0;
            // Cap at 1
            if (value <= 1.0) {
                kernel[col * D + row] = value;
                sum += value;
            }
        }
    }

    // normalize kernel: calc sum of kernel elements and divide each element by the sum
    for (int i = 0; i < D * D; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

/**
 * Rectangular 3D kernel
 *
 * @param D kernel diameter in number of automaton cells
 * @param r radius of the kernel
 * @return pointer to 1D array of floats representing the kernel
 */
float* rectangular_kernel_3d(int D, int r) {
    float* kernel = (float*) calloc(D * D * D, sizeof(float));
    assert(kernel != NULL);

    float sum = 0.0;
    float value;
    float layer_sqr;
    float col_sqr;
    float row_sqr;
    float l_2_dist;
    int layer_area = D * D;

    for (int layer = 0; layer < D; layer++) {
        layer_sqr = (layer - r) * (layer - r);
        for (int col = 0; col < D; col++) {
            col_sqr = (col - r) * (col - r);
            for (int row = 0; row < D; row++) {
                row_sqr = (row - r) * (row - r);
                l_2_dist = sqrt(layer_sqr + col_sqr + row_sqr) / r;
                value = (l_2_dist >= 0.25 && l_2_dist <= 0.75) ? 1.0 : 0.0;
                // Cap at 1
                if (value <= 1.0) {
                    kernel[layer_area * layer + col * D + row] = value;
                    sum += value;
                }
            }
        }
    }

    // normalize kernel: calc sum of kernel elements and divide each element by the sum
    for (int i = 0; i < D * D * D; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}
