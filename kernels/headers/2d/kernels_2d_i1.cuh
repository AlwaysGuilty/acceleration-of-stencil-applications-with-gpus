#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// P0

__global__ void lenia_2d_i1_c0(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
__global__ void lenia_2d_i1_c2(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);

// P1

__global__ void lenia_2d_i1_c0_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
__global__ void lenia_2d_i1_c2_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
__global__ void lenia_2d_i1_c3_p1(float* d_world_dest, float* d_world_src, float* d_conv_kernel, dim3 world_dims, int R, int n_iters);
