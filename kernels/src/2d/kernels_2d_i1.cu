// General includes

#include <stdio.h>

// CUDA related includes

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Custom

#include <libutil.cuh>
#include <kernels_2d_i1.cuh>


using namespace cooperative_groups;

/**
 * Kernel for Lenia
 * 2D, NO CACHING, evolve on GPU
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
__global__ void lenia_2d_i1_c0(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // global 2D thread indices
    int2 g_idxs = make_int2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y
    );

    // If using coop groups grid sync, then pack this check into a variable and check the variable each iter -> calc potential only if check passes, but points need to always execute
    // int is_outside = g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y;

    grid_group grid = this_grid();

    // global 1D thread index, assume world height and width are the same and that world hape in square, so we can use here either height or width
    int g_index_1D = world_dims.x * g_idxs.y + g_idxs.x;

    float potential;
    float growth;
    float* temp;

    #pragma unroll
    for (int i = 0; i < n_iters; i++) {
        // if (!is_outside) {
        potential = convolve_2d(d_world_src, d_conv_kernel, world_dims, g_idxs, R);
        // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
        growth = exponential_growth_mapping(potential);
        d_world_dest[g_index_1D] = clip(d_world_src[g_index_1D] + growth * __frcp_rn(n_iters));
        // }

        // __syncthreads();

        // swap world pointers
        temp = d_world_dest;
        d_world_dest = d_world_src;
        d_world_src = temp;

        __syncthreads();
        grid.sync();
    }
}

/**
 * Kernel for Lenia
 * 2D, With Block+Halo Caching, evolve on GPU
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
__global__ void lenia_2d_i1_c2(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Local (inside block) 2D thread indices
    int l_x = threadIdx.x;
    int l_y = threadIdx.y;

    // Global (in the world) 2D thread indices
    int2 g_idxs = make_int2(
        blockDim.x * blockIdx.x + l_x,
        blockDim.y * blockIdx.y + l_y
    );

    // int is_outside = g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y;

    grid_group grid = this_grid();

    // Global 1D thread index
    int g_index_1D = world_dims.x * g_idxs.y + g_idxs.x;

    // Single cache size in cells
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.y + 2 * R);
    // int cache_size = cache_dims.x * cache_dims.y;
    extern __shared__ float cache[];

    // Current thread's index inside the cache
    int2 c_idxs = make_int2(
        l_x + R,
        l_y + R
    );

    // int c_index_1D = cache_dims.x * c_idxs.y + c_idxs.x;        // row major
    int c_index_1D = cache_dims.y * c_idxs.x + c_idxs.y;        // col major

    float* temp;
    float potential;
    float clipped_value;

    // evolve the world
    #pragma unroll
    for (int i = 0; i < n_iters; i++) {
        // if (!is_outside) {
        // load_cache_2d_c2(cache, d_world_src, world_dims, R);
        // load_cache_2d_c2_multi_loop_row_major(cache, d_world_src, world_dims, R);
        load_cache_2d_c2_multi_loop_col_major(cache, d_world_src, world_dims, R);
        // }

        __syncthreads();

        // if (!is_outside) {
        // potential = convolve_2d_cache(cache, d_conv_kernel, cache_dims, c_idxs, R);
        potential = convolve_2d_cache_col_major(cache, d_conv_kernel, cache_dims, c_idxs, R);
        clipped_value = clip(
            cache[c_index_1D] + exponential_growth_mapping(potential) * __frcp_rn(n_iters)
        );
        // }

        __syncthreads();

        // if (!is_outside) {
            // cache[c_index_1D] = clipped_value;
        // Write back to global mem for the next iter to fetch halo
        d_world_dest[g_index_1D] = clipped_value;
        // }

        // swap dest and src world ptr
        temp = d_world_dest;
        d_world_dest = d_world_src;
        d_world_src = temp;

        __syncthreads();
        grid.sync();
    }
}

/**
 * Kernel for Lenia
 * Plane-by-plane implementation
 * 2D, NO CACHING, evolve on GPU
 *
 * Blocks in 2D: Nx1x1.
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
__global__ void lenia_2d_i1_c0_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Start (top-left) coords (global) of the current tile
    int g_t_start_x = blockIdx.x * blockDim.x;
    int g_t_start_y = blockIdx.y * blockDim.x;        // Rescale to the next tile in y dim

    // global 2D thread indices
    int2 g_idxs = make_int2(
        g_t_start_x + threadIdx.x,
        g_t_start_y
    );

    // int is_outside = g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y;

    grid_group grid = this_grid();

    // Global 1D thread index of the current cell we are updating
    int g_index_1D_i;
    float potential;
    float growth;
    float* temp;

    #pragma unroll
    for (int j = 0; j < n_iters; j++) {
        // if (!is_outside) {

        #pragma unroll
        for (int i = 0; i < blockDim.x; i++) {
            g_idxs.y = g_t_start_y + i;

            // global 1D thread index
            g_index_1D_i = world_dims.x * g_idxs.y + g_idxs.x;

            // get the potential of the cell
            potential = convolve_2d(d_world_src, d_conv_kernel, world_dims, g_idxs, R);

            // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
            growth = exponential_growth_mapping(potential);
            d_world_dest[g_index_1D_i] = clip(d_world_src[g_index_1D_i] + growth * __frcp_rn(n_iters));
        }
        // }

        // __syncthreads();

        // swap world pointers
        temp = d_world_dest;
        d_world_dest = d_world_src;
        d_world_src = temp;

        __syncthreads();
        grid.sync();
    }
}

/**
 * Kernel for Lenia
 * Plane-by-plane implementation
 * 2D, BLOCK+HALO CACHING, evolve on GPU
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
__global__ void lenia_2d_i1_c2_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Start (top-left) coords of the current subblock
    int g_t_start_x = blockIdx.x * blockDim.x;
    int g_t_start_y = blockIdx.y * blockDim.x;

    // global 2D thread indices
    int2 g_idxs = make_int2(
        g_t_start_x + threadIdx.x,
        g_t_start_y
    );

    // int is_outside = g_idxs.x >= world_dims.x || g_idxs.y >= world_dims.y;

    grid_group grid = this_grid();

    // Cache size in cells
    dim3 cache_dims(blockDim.x + 2 * R, blockDim.x + 2 * R);
    extern __shared__ float cache[];

    // Current thread's index inside the cache
    int2 c_idxs = make_int2(
        threadIdx.x + R,
        R
    );
    // Current thread's index inside the cache
    int c_index_1D_i;
    // Global 1D thread index of the current cell we are updating
    int g_index_1D_i;
    float potential;
    float growth;
    float* temp;

    // evolve the world
    #pragma unroll
    for (int j = 0; j < n_iters; j++) {
        // if (!is_outside) {
        // load_cache_2d_c2_p1(cache, d_world_src, world_dims, R);
        // load_cache_2d_c2_p1_multi_loop_row_major(cache, d_world_src, world_dims, R);
        load_cache_2d_c2_p1_multi_loop_col_major(cache, d_world_src, world_dims, R);
        // }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < blockDim.x; i++) {
            g_idxs.y = g_t_start_y + i;

            // global 1D thread index we are updating
            g_index_1D_i = world_dims.x * g_idxs.y + g_idxs.x;

            // Calculate the cache coords of the current cell we are updating
            c_idxs.y = R + i;
            // c_index_1D_i = cache_dims.x * c_idxs.y + c_idxs.x;  // row major
            c_index_1D_i = cache_dims.y * c_idxs.x + c_idxs.y;  // col major

            // get the potential of the cell
            // potential = convolve_2d_cache(cache, d_conv_kernel, cache_dims, c_idxs, R);
            potential = convolve_2d_cache_col_major(cache, d_conv_kernel, cache_dims, c_idxs, R);

            // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
            growth = exponential_growth_mapping(potential);
            d_world_dest[g_index_1D_i] = clip(cache[c_index_1D_i] + growth * __frcp_rn(n_iters));
        }

        // swap dest and src world ptr
        temp = d_world_dest;
        d_world_dest = d_world_src;
        d_world_src = temp;

        __syncthreads();
        grid.sync();
    }
}

/**
 * Kernel for Lenia
 * Plane-by-plane implementation
 * 2D, Cache sub-plane and R-wide halo around, evolve on GPU
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
__global__ void lenia_2d_i1_c3_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters) {
    // Start (top-left) coords of the current subblock
    int g_t_start_x = blockIdx.x * blockDim.x;
    int g_t_start_y = blockIdx.y * blockDim.x;

    // global 2D thread indices
    int2 g_idxs = make_int2(
        g_t_start_x + threadIdx.x,
        g_t_start_y
    );

    grid_group grid = this_grid();

    // Cache size in cells
    dim3 cache_dims(blockDim.x + 2 * R, 1 + 2 * R);
    extern __shared__ float cache[];

    // Current thread's index inside the cache
    int2 c_idxs = make_int2(
        threadIdx.x + R,
        R
    );
    // int c_index_1D_i = cache_dims.x * c_idxs.y + c_idxs.x;  // row major
    int c_index_1D_i = cache_dims.y * c_idxs.x + c_idxs.y;  // col major

    // Global 1D thread index of the current cell we are updating
    int g_index_1D_i;
    float potential;
    float growth;
    float* temp;

    // evolve the world
    #pragma unroll
    for (int j = 0; j < n_iters; j++) {
        // preload_cache_2d_c3_p1(cache, d_world_src, world_dims, R);
        // preload_cache_2d_c3_p1_multi_loop_row_major(cache, d_world_src, world_dims, R);
        preload_cache_2d_c3_p1_multi_loop_col_major(cache, d_world_src, world_dims, R);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < blockDim.x; i++) {
            // if (i) {
            // shift_subdim_up_2d(cache, world_dims, R);
            // shift_subdim_up_2d_multi_loop_row_major(cache, world_dims, R);
            shift_subdim_up_2d_multi_loop_col_major(cache, world_dims, R);

            // __syncthreads();

            // load_new_last_subdim_2d(cache, d_world_src, world_dims, R, i);
            // load_new_last_subdim_2d_multi_loop_row_major(cache, d_world_src, world_dims, R, i);
            load_new_last_subdim_2d_multi_loop_col_major(cache, d_world_src, world_dims, R, i);

            __syncthreads();
            // }

            g_idxs.y = g_t_start_y + i;

            // global 1D thread index we are updating
            g_index_1D_i = world_dims.x * g_idxs.y + g_idxs.x;

            // get the potential of the cell
            // potential = convolve_2d_cache(cache, d_conv_kernel, cache_dims, c_idxs, R);
            potential = convolve_2d_cache_col_major(cache, d_conv_kernel, cache_dims, c_idxs, R);

            // Apply growth mapping, add proportion of growth to the cell and clip it to [0, 1]
            growth = exponential_growth_mapping(potential);
            d_world_dest[g_index_1D_i] = clip(cache[c_index_1D_i] + growth * __frcp_rn(n_iters));

            // __syncthreads();
        }

        // swap dest and src world ptr
        temp = d_world_dest;
        d_world_dest = d_world_src;
        d_world_src = temp;

        __syncthreads();
        grid.sync();
    }
}
