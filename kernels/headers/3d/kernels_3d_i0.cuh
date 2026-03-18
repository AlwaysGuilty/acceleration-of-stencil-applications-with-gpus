#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// p0

__global__ void lenia_3d_i0_c0(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
__global__ void lenia_3d_i0_c2(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);

// p1

__global__ void lenia_3d_i0_c0_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
__global__ void lenia_3d_i0_c2_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
__global__ void lenia_3d_i0_c3_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
