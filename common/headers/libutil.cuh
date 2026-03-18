#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Enums

enum SynchronizationMode {
    SYNC_ON_CPU,
    SYNC_ON_GPU
};

enum CachingMode {
    NO_CACHING,
    BLOCK_AND_HALO,
    SUBPLANE_AND_HALO,
};

// Device on which the simulation will be running
enum SimulationDevice {
    CPU,
    GPU
};

// Input sanitization

int sanitize_int(char* arg);

// Error handling

void error_exit(int err_code, const char* msg);

// Memory handling

void free_null(void* ptr);
void cu_free_null(void* ptr);

// Debugging

void print_2d(float* arr, int h, int w);
void print_3d(float* arr, int h, int w, int d);

// Device code

__device__ void cu_print_2d(float* arr, int h, int w);
__device__ void cu_print_3d(float* arr, int h, int w, int d);

// Convolve from global mem (c0)

__inline_hint__ __host__ __device__ float convolve_2d(float* world, float* conv_kernel, dim3 world_dims, int2 c_idxs, int R);
__inline_hint__ __host__ __device__ float convolve_3d(float* world, float* conv_kernel, dim3 world_dims, int3 c_idxs, int R);

// Convolve from cache (c2 and p1)

__inline_hint__ __host__ __device__ float convolve_2d_cache(float* world, float* conv_kernel, dim3 world_dims, int2 c_idxs, int R);
__inline_hint__ __host__ __device__ float convolve_2d_cache_col_major(float* world, float* conv_kernel, dim3 world_dims, int2 c_idxs, int R);
__inline_hint__ __host__ __device__ float convolve_3d_cache(float* world, float* conv_kernel, dim3 world_dims, int3 c_idxs, int R);
__inline_hint__ __host__ __device__ float convolve_3d_cache_col_major(float* world, float* conv_kernel, dim3 world_dims, int3 c_idxs, int R);

// 2d p0 cache loaders

__inline_hint__ __device__ void load_cache_2d_c2(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_2d_c2_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_2d_c2_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R);

// 3d p0 cache loaders

__inline_hint__ __device__ void load_cache_3d_c2(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_3d_c2_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_3d_c2_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R);

// 2D P1 C2

__inline_hint__ __device__ void load_cache_2d_c2_p1(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_2d_c2_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_2d_c2_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R);

// 3D P1 C2

__inline_hint__ __device__ void load_cache_3d_c2_p1(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_3d_c2_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void load_cache_3d_c2_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R);

// 2D P1 C3

__inline_hint__ __device__ void preload_cache_2d_c3_p1(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void preload_cache_2d_c3_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void preload_cache_2d_c3_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void shift_subdim_up_2d(float* cache, dim3 world_dims, int R);
__inline_hint__ __device__ void shift_subdim_up_2d_multi_loop_row_major(float* cache, dim3 world_dims, int R);
__inline_hint__ __device__ void shift_subdim_up_2d_multi_loop_col_major(float* cache, dim3 world_dims, int R);
__inline_hint__ __device__ void load_new_last_subdim_2d(float* cache, float* world, dim3 world_dims, int R, int i);
__inline_hint__ __device__ void load_new_last_subdim_2d_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R, int i);
__inline_hint__ __device__ void load_new_last_subdim_2d_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R, int i);

// 3D P1 C3

__inline_hint__ __device__ void preload_cache_3d_c3_p1(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void preload_cache_3d_c3_p1_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void preload_cache_3d_c3_p1_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R);
__inline_hint__ __device__ void shift_subdim_up_3d(float* cache, dim3 world_dims, int R);
__inline_hint__ __device__ void shift_subdim_up_3d_multi_loop_row_major(float* cache, dim3 world_dims, int R);
__inline_hint__ __device__ void shift_subdim_up_3d_multi_loop_col_major(float* cache, dim3 world_dims, int R);
__inline_hint__ __device__ void load_new_last_subdim_3d(float* cache, float* world, dim3 world_dims, int R, int i);
__inline_hint__ __device__ void load_new_last_subdim_3d_multi_loop_row_major(float* cache, float* world, dim3 world_dims, int R, int i);
__inline_hint__ __device__ void load_new_last_subdim_3d_multi_loop_col_major(float* cache, float* world, dim3 world_dims, int R, int i);

__inline_hint__ __device__ float exponential_growth_mapping(float potential);
float exponential_growth_mapping_cpu(float potential);
__inline_hint__ __host__ __device__ float clip(float val);
float clip_cpu(float val);

// World initialization

void randomize_world_discrete(float* world, int n);

// Convolution kernel initialization

float* exponential_kernel_2d(int D, int r);
float* exponential_kernel_3d(int D, int r);
float* rectangular_kernel_2d(int D, int r);
float* rectangular_kernel_3d(int D, int r);
