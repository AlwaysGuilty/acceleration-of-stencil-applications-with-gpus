// General includes

#include <stdio.h>

// CUDA related includes

#include <cuda.h>
#include <cuda_runtime.h>

// Custom

#include <libutil.cuh>
#include <kernels_3d_i0.cuh>


/**
 * Kernel for Lenia
 * 3D, NO CACHING, evolve on CPU
 *
 * @param d_world_dest world grid, destination
 * @param d_world_src world grid, source
 * @param d_conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 * @param n_iters number of iterations to evolve the world
*/
__global__ void lenia_3d_i0_c0(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // global thread indices
    int3 g_idxs = make_int3(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y,
        blockDim.z * blockIdx.z + threadIdx.z
    );

    // if (g_idxs.y >= world_dims.y || g_idxs.x >= world_dims.x || g_idxs.z >= world_dims.z) return;

    // global 1D thread index, assume dims are the same
    int g_index_1D = world_dims.x * world_dims.y * g_idxs.z + world_dims.x * g_idxs.y + g_idxs.x;

    float potential = convolve_3d(d_world_src, d_conv_kernel, world_dims, g_idxs, R);

    // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
    float growth = exponential_growth_mapping(potential);
    d_world_dest[g_index_1D] = clip(d_world_src[g_index_1D] + growth * __frcp_rn(n_iters));
}

/**
 * Kernel for Lenia
 * 3D, With Block+Halo Caching, evolve on CPU
 *
 * Method: Caching with shared memory containing a block plus R tile wide halo around it.
 *
 * We assume the world is square shaped
 *
 * @param d_world_dest world grid, destination
 * @param d_world_src world grid, source
 * @param d_conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 * @param n_iters number of iterations to evolve the world
*/
__global__ void lenia_3d_i0_c2(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Local (inside block) 3D thread indices
    int l_x = threadIdx.x;
    int l_y = threadIdx.y;
    int l_z = threadIdx.z;

    // global thread indices
    int g_x = blockDim.x * blockIdx.x + l_x;
    int g_y = blockDim.y * blockIdx.y + l_y;
    int g_z = blockDim.z * blockIdx.z + l_z;

    // if (g_y >= world_dims.y || g_x >= world_dims.x || g_z >= world_dims.z) return;

    // global 1D thread index, assume dims are the same
    int g_index_1D = world_dims.x * world_dims.y * g_z + world_dims.x * g_y + g_x;

    // Cache size in cells
    dim3 cache_dims(
        blockDim.x + 2 * R,
        blockDim.y + 2 * R,
        blockDim.z + 2 * R
    );
    extern __shared__ float cache[];

    // Current thread's index inside the cache
    int3 c_idxs = make_int3(
        l_x + R,
        l_y + R,
        l_z + R
    );

    // int c_index_1D = cache_dims.x * cache_dims.y * c_idxs.z + cache_dims.x * c_idxs.y + c_idxs.x;   // row major
    int c_index_1D = cache_dims.y * cache_dims.z * c_idxs.x + cache_dims.z * c_idxs.y + c_idxs.z;   // col major

    // load_cache_3d_c2(cache, d_world_src, world_dims, R);
    // load_cache_3d_c2_multi_loop_row_major(cache, d_world_src, world_dims, R);
    load_cache_3d_c2_multi_loop_col_major(cache, d_world_src, world_dims, R);

    __syncthreads();

    // float potential = convolve_3d_cache(cache, d_conv_kernel, cache_dims, c_idxs, R);
    float potential = convolve_3d_cache_col_major(cache, d_conv_kernel, cache_dims, c_idxs, R);

    float growth = exponential_growth_mapping(potential);
    d_world_dest[g_index_1D] = clip(cache[c_index_1D] + growth * __frcp_rn(n_iters));
}

/**
 * Kernel for Lenia
 * Sub-block implementation
 * 3D, NO CACHING, evolve on CPU
 *
 * We assume the world is square shaped
 *
 * @param d_world_dest world grid, destination
 * @param d_world_src world grid, source
 * @param d_conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 * @param n_iters number of iterations to evolve the world
*/
__global__ void lenia_3d_i0_c0_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Start (top-left) coords (global) of the current tile
    int g_t_start_x = blockIdx.x * blockDim.x;
    int g_t_start_y = blockIdx.y * blockDim.y;
    int g_t_start_z = blockIdx.z * blockDim.x;        // Rescale to the next sub-block

    // global 3D thread indices
    int3 g_idxs = make_int3(
        g_t_start_x + threadIdx.x,
        g_t_start_y + threadIdx.y,
        g_t_start_z
    );

    // if (g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y || g_idxs.z >= world_dims.z) return;

    // Global 1D thread index of the current cell we are updating
    int g_index_1D_i;
    float potential;
    float growth;

    #pragma unroll
    for (int i = 0; i < blockDim.x; i++) {
        g_idxs.z = g_t_start_z + i;

        // global 1D thread index
        g_index_1D_i = world_dims.x * world_dims.y * g_idxs.z + world_dims.x * g_idxs.y + g_idxs.x;

        // get the potential of the cell
        potential = convolve_3d(d_world_src, d_conv_kernel, world_dims, g_idxs, R);

        // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
        growth = exponential_growth_mapping(potential);
        d_world_dest[g_index_1D_i] = clip(d_world_src[g_index_1D_i] + growth * __frcp_rn(n_iters));
    }
}

/**
 * Kernel for Lenia
 * Sub-block implementation
 * 3D, BLOCK+HALO CACHING, evolve on CPU
 *
 * We assume the world is square shaped
 *
 * @param d_world_dest world grid, destination
 * @param d_world_src world grid, source
 * @param d_conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 * @param n_iters number of iterations to evolve the world
*/
__global__ void lenia_3d_i0_c2_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    int g_t_start_x = blockIdx.x * blockDim.x;
    int g_t_start_y = blockIdx.y * blockDim.x;  // assume the tile should be a cube
    int g_t_start_z = blockIdx.z * blockDim.x;

    // global 3D thread indices
    int3 g_idxs = make_int3(
        g_t_start_x + threadIdx.x,
        g_t_start_y + threadIdx.y,
        g_t_start_z
    );

    // if (g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y || g_idxs.z >= world_dims.z) return;

    // Cache size in cells
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R, blockDim.x + 2 * R);
    extern __shared__ float cache[];

    // Current thread's index inside the cache
    int3 c_idxs = make_int3(
        threadIdx.x + R,
        threadIdx.y + R,
        R
    );
    int c_index_1D_i;

    // load_cache_3d_c2_p1(cache, d_world_src, world_dims, R);
    // load_cache_3d_c2_p1_multi_loop_row_major(cache, d_world_src, world_dims, R);
    load_cache_3d_c2_p1_multi_loop_col_major(cache, d_world_src, world_dims, R);

    __syncthreads();

    // Global 1D thread index of the current cell we are updating
    int g_index_1D_i;
    float potential;
    float growth;

    #pragma unroll
    for (int i = 0; i < blockDim.x; i++) {
        g_idxs.z = g_t_start_z + i;

        // global 1D thread index
        g_index_1D_i = world_dims.x * world_dims.y * g_idxs.z + world_dims.x * g_idxs.y + g_idxs.x;

        // Caclulate the cache coords of the current cell we are updating
        c_idxs.z = R + i;
        // c_index_1D_i = cache_dims.x * cache_dims.y * c_idxs.z + cache_dims.x * c_idxs.y + c_idxs.x; // row major
        c_index_1D_i = cache_dims.y * cache_dims.z * c_idxs.x + cache_dims.z * c_idxs.y + c_idxs.z; // col major

        // get the potential of the cell
        // potential = convolve_3d_cache(cache, d_conv_kernel, cache_dims, c_idxs, R);
        potential = convolve_3d_cache_col_major(cache, d_conv_kernel, cache_dims, c_idxs, R);

        // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
        growth = exponential_growth_mapping(potential);
        d_world_dest[g_index_1D_i] = clip(cache[c_index_1D_i] + growth * __frcp_rn(n_iters));
    }
}

/**
 * Kernel for Lenia
 * Sub-block implementation
 * 3D, SUBPLANE+HALO CACHING, evolve on CPU
 *
 * We assume the world is square shaped
 *
 * @param d_world_dest world grid, destination
 * @param d_world_src world grid, source
 * @param d_conv_kernel convolution kernel
 * @param world_dims dim3 vector of world dim sizes
 * @param R neighbourhood radius
 * @param n_iters number of iterations to evolve the world
*/
__global__ void lenia_3d_i0_c3_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Start (top-left) coords of the current subblock
    int g_t_start_x = blockIdx.x * blockDim.x;
    int g_t_start_y = blockIdx.y * blockDim.x;
    int g_t_start_z = blockIdx.z * blockDim.x;

    // global 2D thread indices
    int3 g_idxs = make_int3(
        g_t_start_x + threadIdx.x,
        g_t_start_y + threadIdx.y,
        g_t_start_z
    );

    // if (g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y || g_idxs.z >= world_dims.z) return;

    // Cache size in cells
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R, 1 + 2 * R);
    extern __shared__ float cache[];

    // Current thread's index inside the cache
    int3 c_idxs = make_int3(
        threadIdx.x + R,
        threadIdx.y + R,
        R
    );
    int c_index_1D_i = cache_dims.x * cache_dims.y * c_idxs.z + cache_dims.x * c_idxs.y + c_idxs.x; // row major
    // int c_index_1D_i = cache_dims.z * cache_dims.y * c_idxs.x + cache_dims.z * c_idxs.y + c_idxs.z; // col major

    preload_cache_3d_c3_p1(cache, d_world_src, world_dims, R);
    // preload_cache_3d_c3_p1_multi_loop_row_major(cache, d_world_src, world_dims, R);
    // preload_cache_3d_c3_p1_multi_loop_col_major(cache, d_world_src, world_dims, R);
    __syncthreads();

    // Global 1D thread index of the current cell we are updating
    int g_index_1D_i;
    float potential;
    float growth;

    #pragma unroll
    for (int i = 0; i < blockDim.x; i++) {
        // if (i) {
        // Move rows in cache 1 up
        shift_subdim_up_3d(cache, world_dims, R);
        // shift_subdim_up_3d_multi_loop_row_major(cache, world_dims, R);
        // shift_subdim_up_3d_multi_loop_col_major(cache, world_dims, R);

        // __syncthreads();

        // Load new last row of cache
        load_new_last_subdim_3d(cache, d_world_src, world_dims, R, i);
        // load_new_last_subdim_3d_multi_loop_row_major(cache, d_world_src, world_dims, R, i);
        // load_new_last_subdim_3d_multi_loop_col_major(cache, d_world_src, world_dims, R, i);

        __syncthreads();
        // }

        g_idxs.z = g_t_start_z + i;

        // global 1D thread index we are updating
        g_index_1D_i = world_dims.x * world_dims.y * g_idxs.z + world_dims.x * g_idxs.y + g_idxs.x;

        // get the potential of the cell
        potential = convolve_3d_cache(cache, d_conv_kernel, cache_dims, c_idxs, R);
        // potential = convolve_3d_cache_col_major(cache, d_conv_kernel, cache_dims, c_idxs, R);

        // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
        growth = exponential_growth_mapping(potential);
        d_world_dest[g_index_1D_i] = clip(cache[c_index_1D_i] + growth * __frcp_rn(n_iters));

        // incorrect if no syncthreads here
        __syncthreads();
    }
}
