#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernels_2d_i0.cuh>
#include <kernels_2d_i1.cuh>
#include <kernels_3d_i0.cuh>
#include <kernels_3d_i1.cuh>
#include <libutil.cuh>
#include <LeniaSimulation.cuh>


LeniaSimulation::LeniaSimulation() {
    this->sim_device = SimulationDevice::CPU;
    this->cache_n_grid_points = 0;
    this->cache_mem_size = 0;
    this->use_subplaning = false;

    // Should we store ref results in cpu sim and compare against that in gpu sim?
    this->test_mode = false;
    // signifies we are in a debugger and therefore should adjust the ref results path
    this->debug_mode = false;

    if (this->test_mode) {
        this->set_base_results_dir_path();
        this->set_ref_results_dir_path();
        this->set_curr_results_dir_path();
    }
};

/**
 * Seperate from destructor to avoid calling cuda functions after the lib has been unloaded
 */
void LeniaSimulation::free_mem() {
    if (this->sim_device == SimulationDevice::GPU) {
        free_null(this->h_world_src);
        free_null(this->h_world_dest);
        free_null(this->h_conv_kernel);
        cu_free_null(this->d_world_src);
        cu_free_null(this->d_world_dest);
        this->d_last_dest = NULL;
    }
    else {
        free_null(this->world_src);
        free_null(this->world_dest);
    }
};

/**
 * Compute block size
 *
 * In subplaning, blocks should be 1 dim lower than the world.
 */
void LeniaSimulation::set_block_size_params() {
    this->block_dim_sizes.x = this->block_side_size;
    if (this->use_subplaning) {
        this->block_dim_sizes.y = this->dims == 2 ? 1 : this->block_side_size;
        this->block_dim_sizes.z = 1;
    }
    else {
        this->block_dim_sizes.y = this->block_side_size;
        this->block_dim_sizes.z = this->dims == 2 ? 1 : this->block_side_size;
    }
    this->block_n_grid_points = (size_t) (
        this->block_dim_sizes.x * this->block_dim_sizes.y * this->block_dim_sizes.z
    );
    this->block_mem_size = (size_t) (this->block_n_grid_points * sizeof(float));
};

void LeniaSimulation::set_world_size_params() {
    this->world_dim_sizes.x = this->world_side_size;
    this->world_dim_sizes.y = this->world_side_size;
    this->world_dim_sizes.z = this->dims == 2 ? 1 : this->world_side_size;
    this->world_n_grid_points = (size_t) (
        this->world_dim_sizes.x * this->world_dim_sizes.y * this->world_dim_sizes.z
    );
    this->world_mem_size = (size_t) (this->world_n_grid_points * sizeof(float));
};

void LeniaSimulation::set_time_params(int n_iters) {
    this->n_iters = n_iters;
    this->delta_t = 1.0 / this->n_iters;
};

void LeniaSimulation::set_stencil_params() {
    this->conv_kernel_n_grid_points = (size_t) pow(this->stencil_diameter, this->dims);
    this->conv_kernel_mem_size = (size_t) (this->conv_kernel_n_grid_points * sizeof(float));
    this->delta_R = 1.0 / this->R;
};

void LeniaSimulation::precompute_conv_kernel() {
    printf("Precomputing the convolution kernel of size %d ^ %d\n", this->stencil_diameter, this->dims);
    switch (this->dims) {
        case 2:
            this->h_conv_kernel = exponential_kernel_2d(
                this->stencil_diameter,
                this->dims
            );
            break;
        case 3:
            this->h_conv_kernel = exponential_kernel_3d(
                this->stencil_diameter,
                this->dims
            );
            break;
        default:
            error_exit(1, "Conv kernel precompute: Invalid number of dimensions.");
    }
};

void LeniaSimulation::transfer_d2h() {
    cudaError_t retval;

    // Copy data back to host
    retval = cudaMemcpy(
        this->h_world_dest,
        this->d_world_src,
        this->world_mem_size,
        cudaMemcpyDeviceToHost
    );
    assert(retval == cudaSuccess);
};

void LeniaSimulation::transfer_h2d() {
    cudaError_t retval;

    puts("Copying data to GPU mem...");
    retval = cudaMemcpy(
        this->d_world_src,
        this->h_world_src,
        this->world_mem_size,
        cudaMemcpyHostToDevice
    );
    assert(retval == cudaSuccess);
    retval = cudaMemcpy(
        this->d_conv_kernel,
        this->h_conv_kernel,
        this->conv_kernel_mem_size,
        cudaMemcpyHostToDevice
    );
    assert(retval == cudaSuccess);
};

void LeniaSimulation::init_host_world_mem() {
    if (this->sim_device == SimulationDevice::GPU) {
        // Alloc mem on host
        this->h_world_src = (float*) calloc(this->world_n_grid_points, sizeof(float));
        assert(this->h_world_src != NULL);

        this->h_world_dest = (float*) calloc(this->world_n_grid_points, sizeof(float));
        assert(this->h_world_dest != NULL);

        // Init src world
        randomize_world_discrete(this->h_world_src, this->world_n_grid_points);

        // Dump to file
        if (this->test_mode) {
            this->dump_world(this->h_world_src, this->curr_src_world_dump_path);
        }
    }
    else {
        this->world_src = (float*) calloc(this->world_n_grid_points, sizeof(float));
        assert(world_src != NULL);
        this->world_dest = (float*) calloc(this->world_n_grid_points, sizeof(float));
        assert(world_dest != NULL);

        // Init src world
        randomize_world_discrete(this->world_src, this->world_n_grid_points);

        // Dump to file
        if (this->test_mode) {
            this->dump_world(this->world_src, this->ref_src_world_dump_path);
        }
    }
};

/**
 * Init GPU mem and transfer data to GPU
 */
void LeniaSimulation::init_gpu_mem() {
    cudaError_t retval;

    // Alloc mem on GPU
    printf("Allocating memory on GPU: %lu bytes\n", 2 * this->world_mem_size + this->conv_kernel_mem_size);
    retval = cudaMalloc(&this->d_world_src, this->world_mem_size);
    assert(retval == cudaSuccess);
    retval = cudaMalloc(&this->d_world_dest, this->world_mem_size);
    assert(retval == cudaSuccess);
    retval = cudaMalloc(&this->d_conv_kernel, this->conv_kernel_mem_size);
    assert(retval == cudaSuccess);

    // Copy data to GPU
    this->transfer_h2d();
};

/**
 * Compute grid size
 *
 * For subplaning the grid needs to be smaller
 */
void LeniaSimulation::set_grid_size() {
    this->grid_dim_sizes.x = this->world_dim_sizes.x / this->block_dim_sizes.x;
    // if (this->use_subplaning) {
    //     this->grid_dim_sizes.y = this->dims == 2 ? 1 : (this->world_dim_sizes.y / this->block_dim_sizes.y);
    //     this->grid_dim_sizes.z = 1;
    // }
    // else {
    //     this->grid_dim_sizes.y = this->world_dim_sizes.y / this->block_dim_sizes.y;
    //     this->grid_dim_sizes.z = this->dims == 2 ? 1 : (this->world_dim_sizes.z / this->block_dim_sizes.z);
    // }

    // Divide by .x because we want to be grid unchanged in subplaning mode
    this->grid_dim_sizes.y = this->world_dim_sizes.y / this->block_dim_sizes.x;
    this->grid_dim_sizes.z = this->dims == 2 ? 1 : (this->world_dim_sizes.z / this->block_dim_sizes.x);
};

void LeniaSimulation::set_cache_params() {
    if (!this->caching_mode) return;

    // Cache dims
    if (this->use_subplaning) {
        if (this->caching_mode == CachingMode::BLOCK_AND_HALO) {
            this->cache_dim_sizes.x = this->block_dim_sizes.x + 2 * this->R;
            this->cache_dim_sizes.y = this->block_dim_sizes.x + 2 * this->R;
            this->cache_dim_sizes.z = this->dims == 2 ? 1 : this->block_dim_sizes.x + 2 * this->R;
        }
        else if (this->caching_mode == CachingMode::SUBPLANE_AND_HALO) {
            this->cache_dim_sizes.x = this->block_dim_sizes.x + 2 * this->R;
            if (this->dims == 2) {
                this->cache_dim_sizes.y = 1 + 2 * this->R;
                this->cache_dim_sizes.z = 1;
            }
            else {
                this->cache_dim_sizes.y = this->block_dim_sizes.y + 2 * this->R;
                this->cache_dim_sizes.z = 1 + 2 * this->R;
            }
        }
    }
    else {
        this->cache_dim_sizes.x = this->block_dim_sizes.x;
        this->cache_dim_sizes.y = this->block_dim_sizes.y;
        this->cache_dim_sizes.z = this->block_dim_sizes.z;
        if (this->caching_mode == CachingMode::BLOCK_AND_HALO) {
            // Add halo around
            this->cache_dim_sizes.x += (2 * this->R);
            this->cache_dim_sizes.y += (2 * this->R);
            if (this->dims == 3) {
                this->cache_dim_sizes.z += (2 * this->R);
            }
        }
    }

    this->cache_n_grid_points = (size_t) (
        this->cache_dim_sizes.x * this->cache_dim_sizes.y * this->cache_dim_sizes.z
    );
    this->cache_mem_size = (size_t) (this->cache_n_grid_points * sizeof(float));
    printf("Shared memory used per block: %lu bytes\n", this->cache_mem_size);
    // 48kB is the max shared mem size on RTX2060 Mobile, V100s and H100
    assert(this->cache_mem_size <= 49152);
};

/**
 * Select CUDA kernel depending on dims, sync mode, caching mode and thread domain
 *
 * TODO: selector needs to be sort of a C++ hashmap, mapping tuples of ints to func pointers?.
 * Combination of params always needs to return the same kernel.
 */
void LeniaSimulation::set_cuda_kernel() {
    switch (this->dims) {
        case 2:
            switch (this->sync_mode) {
                case SynchronizationMode::SYNC_ON_CPU:
                    if (this->use_subplaning) {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING: {
                                this->kernel = lenia_2d_i0_c0_p1;
                                break;
                            }
                            case CachingMode::BLOCK_AND_HALO: {
                                this->kernel = lenia_2d_i0_c2_p1;
                                break;
                            }
                            case CachingMode::SUBPLANE_AND_HALO: {
                                this->kernel = lenia_2d_i0_c3_p1;
                                break;
                            }
                            default: {
                                error_exit(1, "Not Implemented.");
                                break;
                            }
                        }
                    }
                    else {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING:
                                this->kernel = lenia_2d_i0_c0;
                                break;
                            case CachingMode::BLOCK_AND_HALO:
                                this->kernel = lenia_2d_i0_c2;
                                break;
                            default:
                                error_exit(1, "Invalid caching mode.");
                                break;
                        }
                    }
                    break;
                case SynchronizationMode::SYNC_ON_GPU:
                    if (this->use_subplaning) {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING: {
                                this->kernel = lenia_2d_i1_c0_p1;
                                break;
                            }
                            case CachingMode::BLOCK_AND_HALO: {
                                this->kernel = lenia_2d_i1_c2_p1;
                                break;
                            }
                            case CachingMode::SUBPLANE_AND_HALO: {
                                this->kernel = lenia_2d_i1_c3_p1;
                                break;
                            }
                            default: {
                                error_exit(1, "Not Implemented.");
                                break;
                            }
                        }
                    }
                    else {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING:
                                this->kernel = lenia_2d_i1_c0;
                                break;
                            case CachingMode::BLOCK_AND_HALO:
                                this->kernel = lenia_2d_i1_c2;
                                break;
                            default:
                                error_exit(1, "Invalid caching mode.");
                                break;
                        }
                    }
                    break;
                default:
                    error_exit(1, "Invalid sync mode.");
                    break;
            }
            break;
        case 3:
            switch (this->sync_mode) {
                case SynchronizationMode::SYNC_ON_CPU:
                    if (this->use_subplaning) {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING: {
                                this->kernel = lenia_3d_i0_c0_p1;
                                break;
                            }
                            case CachingMode::BLOCK_AND_HALO: {
                                this->kernel = lenia_3d_i0_c2_p1;
                                break;
                            }
                            case CachingMode::SUBPLANE_AND_HALO: {
                                this->kernel = lenia_3d_i0_c3_p1;
                                break;
                            }
                            default: {
                                error_exit(1, "Not Implemented.");
                                break;
                            }
                        }
                    }
                    else {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING:
                                this->kernel = lenia_3d_i0_c0;
                                break;
                            case CachingMode::BLOCK_AND_HALO:
                                this->kernel = lenia_3d_i0_c2;
                                break;
                            default:
                                error_exit(1, "Invalid caching mode.");
                                break;
                        }
                    }
                    break;
                case SynchronizationMode::SYNC_ON_GPU:
                    if (this->use_subplaning) {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING: {
                                this->kernel = lenia_3d_i1_c0_p1;
                                break;
                            }
                            case CachingMode::BLOCK_AND_HALO: {
                                this->kernel = lenia_3d_i1_c2_p1;
                                break;
                            }
                            case CachingMode::SUBPLANE_AND_HALO: {
                                this->kernel = lenia_3d_i1_c3_p1;
                                break;
                            }
                            default: {
                                error_exit(1, "Not Implemented.");
                                break;
                            }
                        }
                    }
                    else {
                        switch (this->caching_mode) {
                            case CachingMode::NO_CACHING:
                                this->kernel = lenia_3d_i1_c0;
                                break;
                            case CachingMode::BLOCK_AND_HALO:
                                this->kernel = lenia_3d_i1_c2;
                                break;
                            default:
                                error_exit(1, "Invalid caching mode.");
                                break;
                        }
                    }
                    break;
                default:
                    error_exit(1, "Invalid sync mode.");
                    break;
            }
            break;
        default:
            error_exit(1, "CUDA kernel selector: Invalid number of dimensions.");
    }
};

void LeniaSimulation::start_timer() {
    cudaError_t retval;

    if (this->sim_device == SimulationDevice::GPU) {
        retval = cudaEventCreate(&this->start);
        assert(retval == cudaSuccess);
        retval = cudaEventCreate(&this->stop);
        assert(retval == cudaSuccess);
        retval = cudaEventRecord(this->start);
        assert(retval == cudaSuccess);
    }
    else {
        this->start_time = omp_get_wtime();
    }
};

void LeniaSimulation::stop_timer() {
    cudaError_t retval;

    if (this->sim_device == SimulationDevice::GPU) {
        // retval = cudaStreamSynchronize(0);
        // assert(retval == cudaSuccess);
        retval = cudaDeviceSynchronize();
        assert(retval == cudaSuccess && "Could not device sync.");
        retval = cudaEventRecord(this->stop);
        assert(retval == cudaSuccess && "Could not record event.");
        retval = cudaEventSynchronize(this->stop);
        assert(retval == cudaSuccess && "Cound not event sync.");
        retval = cudaEventElapsedTime(&this->gpu_exec_time, this->start, this->stop);
        assert(retval == cudaSuccess && "Could not calc elapsed time.");
        // Convert to s
        this->exec_time = this->gpu_exec_time / 1000.0;
    }
    else {
        this->stop_time = omp_get_wtime();
        this->exec_time = this->stop_time - this->start_time;
    }
    printf("Sim exec time: %0.3f s\n", this->exec_time);
    puts("--------------------------------------------");
};

/**
 * Runtime checks
 *
 * Used as a place where runtime checks that should prevent runtime cuda errors/segfaults should be put.
 * Should be called just before running the simulation.
 */
void LeniaSimulation::gpu_runtime_checks() {
    // Fail if world is not at least as big as a block
    assert(this->world_side_size >= this->block_side_size);

    assert(this->block_side_size >= 4 && "Invalid block size.");

    // Fail if c2 and block
    if (this->caching_mode == CachingMode::BLOCK_AND_HALO) {
        if (this->dims == 2) {
            assert(this->block_side_size <= 128);
        }
    }

    if (this->use_subplaning) {
        // assert(this->block_side_size >= 8 && "Invalid block size for P=1.");
        // if (this->dims == 2) {
        //     assert(this->block_side_size >= 32 && "Invalid block size for P=1, D=2");
        // }
        // else if (this->dims == 3) {
        //     assert(this->block_side_size >= 8 && this->block_side_size <= 16 && "Invalid block size for P=1, D=3");
        // }
    }
    else {
        // Fail if c3 not in subplaning mode
        assert(this->caching_mode != CachingMode::SUBPLANE_AND_HALO);

        if (this->dims == 3) {
            assert(this->block_side_size >= 4 && this->block_side_size <= 8 && "Invalid block size for P=0, D=3");
        }
        else if (this->dims == 2) {
            assert(this->block_side_size >= 8 && this->block_side_size <= 16 && "Invalid block size for P=0, D=2");
        }
    }
}

/**
 * Runs simulation on GPU
 */
void LeniaSimulation::run_gpu_sim() {
    cudaError_t retval;

    switch (this->sync_mode) {
        case SynchronizationMode::SYNC_ON_CPU: {
            // Run kernel each iteration
            for (int i = 0; i < this->n_iters; i++) {
                this->kernel<<<this->grid_dim_sizes, this->block_dim_sizes, this->cache_mem_size>>>(
                    this->d_world_dest,
                    this->d_world_src,
                    this->d_conv_kernel,
                    this->world_dim_sizes,
                    this->R,
                    this->n_iters
                );
                retval = cudaGetLastError();
                assert((retval == cudaSuccess) && "Kernel execution failed\n");

                // Swap pointers
                this->d_last_dest = this->d_world_dest;
                this->d_world_dest = this->d_world_src;
                this->d_world_src = this->d_last_dest;
            }
            // After sim we need to transfer d2h from d_world_src in any way
            break;
        }
        case SynchronizationMode::SYNC_ON_GPU: {
            // Iterate inside the kernel

            int n_devices = 0;
            int dev = 0;
            int supports_coop_launch = 0;

            retval = cudaGetDeviceCount(&n_devices);
            assert(retval == cudaSuccess);
            assert(n_devices == 1);

            retval = cudaDeviceGetAttribute(
                &supports_coop_launch,
                cudaDevAttrCooperativeLaunch,
                dev
            );
            assert(retval == cudaSuccess);
            assert(supports_coop_launch);

            void* launch_args[] = {
                (void*) &this->d_world_dest,
                (void*) &this->d_world_src,
                (void*) &this->d_conv_kernel,
                (void*) &this->world_dim_sizes,
                (void*) &this->R,
                (void*) &this->n_iters
            };

            cudaDeviceProp device_prop;
            retval = cudaGetDeviceProperties(&device_prop, dev);
            assert(retval == cudaSuccess);
            int n_threads = this->block_dim_sizes.x * this->block_dim_sizes.y * this->block_dim_sizes.z;
            int n_blocks_per_sm = 0;
            retval = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &n_blocks_per_sm,
                this->kernel,
                n_threads,
                this->cache_mem_size
            );
            assert(retval == cudaSuccess);
            printf("Max active blocks per SM: %d\n", n_blocks_per_sm);
            printf("Number of multiprocessors: %d\n", device_prop.multiProcessorCount);
            assert(
                device_prop.multiProcessorCount * n_blocks_per_sm >= this->grid_dim_sizes.x * this->grid_dim_sizes.y * this->grid_dim_sizes.z
            );

            retval = cudaLaunchCooperativeKernel(
                (void*) this->kernel,
                this->grid_dim_sizes,
                this->block_dim_sizes,
                launch_args,
                this->cache_mem_size,
                NULL
            );
            assert((retval == cudaSuccess) && "Kernel execution failed\n");

            // this->kernel<<<this->grid_dim_sizes, this->block_dim_sizes, this->cache_mem_size>>>(
            //     this->d_world_dest,
            //     this->d_world_src,
            //     this->d_conv_kernel,
            //     this->world_dim_sizes,
            //     this->R,
            //     this->n_iters
            // );
            // retval = cudaGetLastError();
            // assert((retval == cudaSuccess) && "Kernel execution failed\n");

            // this->d_last_dest = this->d_world_dest;
            break;
        }
        default: {
            error_exit(1, "Invalid sync mode.");
        }
    }
};

/**
 * Runs simulation on CPU
 */
void LeniaSimulation::run_cpu_sim() {
    float* temp;

    // Iterate over time steps
    for (int t = 0; t < this->n_iters; t++) {
        // Iterate over the 1D world
        for (int i = 0; i < this->world_n_grid_points; i++) {
            // Convolve the world with the kernel
            switch (this->dims) {
                case 2: {
                    int2 cell_idxs = make_int2(
                        i % this->world_side_size,
                        i / this->world_side_size
                    );
                    this->world_dest[i] = convolve_2d(
                        this->world_src,
                        this->h_conv_kernel,
                        this->world_dim_sizes,
                        cell_idxs,
                        this->R
                    );
                    break;
                }
                case 3: {
                    int3 cell_idxs = make_int3(
                        i % this->world_side_size,
                        (i / this->world_side_size) % this->world_side_size,
                        i / (this->world_side_size * this->world_side_size)
                    );
                    this->world_dest[i] = convolve_3d(
                        this->world_src,
                        this->h_conv_kernel,
                        this->world_dim_sizes,
                        cell_idxs,
                        this->R
                    );
                    break;
                }
                default: {
                    error_exit(1, "CPU sim: Invalid number of dimensions.");
                }
            }

            // Growth mapping
            this->world_dest[i] = exponential_growth_mapping_cpu(this->world_dest[i]);
            // Clip
            this->world_dest[i] = clip_cpu(
                this->world_src[i] + this->delta_t * this->world_dest[i]
            );

        }
        // Switch src and dest world pointers after each timestep
        temp = this->world_src;
        this->world_src = this->world_dest;
        this->world_dest = temp;
    }
    temp = NULL;
};

void LeniaSimulation::set_base_results_dir_path() {
    int retval;

    this->base_results_dir = (char*) calloc(200, sizeof(char));
    assert(this->base_results_dir != NULL);

    if (this->debug_mode) {
        retval = snprintf(
            this->base_results_dir,
            200,
            "%s/diplomska/results",
            getcwd(NULL, 0)
        );
    }
    else {
        retval = snprintf(
            this->base_results_dir,
            200,
            "%s/results",
            getcwd(NULL, 0)
        );
    }
    assert(retval > 0 && retval <= 200);

    printf("Base results dir path: %s\n", this->base_results_dir);
}

/**
 * Compute path to output file for current CPU sim params.
 */
void LeniaSimulation::set_ref_output_file_path() {
    int retval;

    this->ref_output_file_path = (char*) calloc(200, sizeof(char));
    assert(this->ref_output_file_path != NULL);

    assert(this->world_side_size > 0);
    assert(this->dims);

    retval = snprintf(
        this->ref_output_file_path,
        200,
        "%s/ref_W%dx%dx%d_T%d_R%d.txt",
        this->ref_results_dir,
        this->world_side_size,
        this->world_side_size,
        this->dims == 2 ? 1 : this->world_side_size,
        this->n_iters,
        this->R
    );
    assert(retval > 0 && retval <= 200);
};

/**
 * Compute path to reference src world (the one from CPU sim) file for current sim params.
 */
void LeniaSimulation::set_ref_src_world_dump_path() {
    int retval;

    this->ref_src_world_dump_path = (char*) calloc(200, sizeof(char));
    assert(this->ref_src_world_dump_path != NULL);

    assert(this->world_side_size > 0);
    assert(this->dims);

    retval = snprintf(
        this->ref_src_world_dump_path,
        200,
        "%s/ref_src_%dx%dx%d.txt",
        this->ref_results_dir,
        this->world_side_size,
        this->world_side_size,
        this->dims == 2 ? 1 : this->world_side_size
    );
    assert(retval > 0 && retval <= 200);
};

/**
 * Compute path to reference (CPU) results dir.
 */
void LeniaSimulation::set_ref_results_dir_path() {
    int retval;

    this->ref_results_dir = (char*) calloc(200, sizeof(char));
    assert(this->ref_results_dir != NULL);

    retval = snprintf(
        this->ref_results_dir,
        200,
        "%s/reference",
        this->base_results_dir
    );
    assert(retval > 0 && retval <= 200);

    printf("Ref results dir path: %s\n", this->ref_results_dir);
};

/**
 * Compute path to results dir for current GPU sim params.
 */
void LeniaSimulation::set_curr_results_dir_path() {
    int retval;

    this->curr_results_dir = (char*) calloc(200, sizeof(char));
    assert(this->curr_results_dir != NULL);

    retval = snprintf(
        this->curr_results_dir,
        200,
        "%s/curr",
        this->base_results_dir
    );
    assert(retval > 0 && retval <= 200);

    printf("Curr results dir path: %s\n", this->curr_results_dir);
};

/**
 * Compute path to output file for current GPU sim params.
 */
void LeniaSimulation::set_curr_output_file_path() {
    int retval;

    this->curr_output_file_path = (char*) calloc(200, sizeof(char));
    assert(this->curr_output_file_path != NULL);

    assert(this->world_side_size > 0);
    assert(this->dims);

    retval = snprintf(
        this->curr_output_file_path,
        200,
        "%s/curr_W%dx%dx%d_I%d_B%d_T%d_R%d_C%d_P%d.txt",
        this->curr_results_dir,
        this->world_side_size,
        this->world_side_size,
        this->dims == 2 ? 1 : this->world_side_size,
        this->sync_mode,
        this->block_side_size,
        this->n_iters,
        this->R,
        this->caching_mode,
        int(this->use_subplaning)
    );
    assert(retval > 0 && retval <= 200);
};

/**
 * Compute path to current GPU sim's source world dump file.
 */
void LeniaSimulation::set_curr_src_world_dump_path() {
    int retval;

    this->curr_src_world_dump_path = (char*) calloc(200, sizeof(char));
    assert(this->curr_src_world_dump_path != NULL);

    assert(this->world_side_size > 0);
    assert(this->dims);

    retval = snprintf(
        this->curr_src_world_dump_path,
        200,
        "%s/curr_src_%dx%dx%d.txt",
        this->curr_results_dir,
        this->world_side_size,
        this->world_side_size,
        this->dims == 2 ? 1 : this->world_side_size
    );
    assert(retval > 0 && retval <= 200);
}

/**
 * Ensure dir for path exists
 */
void LeniaSimulation::ensure_dir_path(char* path) {
    int retval;

    DIR* dir = opendir(path);
    if (dir) return;

    retval = mkdir(path, 0700);
    assert(!retval);
};

/**
 * Store world to a file
 */
void LeniaSimulation::dump_world(float* world, char* filename) {
    int retval;

    // Dump contents of this->world_dest to a text file
    // File format:
    // First line: <dims> <dimx> <dimy> <dimz>
    // Z sections of Y rows of X values
    // Cell values separated by tabs, rows separated by newlines, sections by row of dashes

    FILE* file = fopen(filename, "w");
    assert(file != NULL);

    // first line
    retval = fprintf(
        file,
        "%d %d %d %d\n",
        this->dims,
        this->world_dim_sizes.x,
        this->world_dim_sizes.y,
        this->world_dim_sizes.z
    );
    assert(retval > 0);

    // world data
    for (int i = 0; i < this->world_dim_sizes.z; i++) {
        for (int j = 0; j < this->world_dim_sizes.y; j++) {
            for (int k = 0; k < this->world_dim_sizes.x; k++) {
                retval = fprintf(
                    file,
                    "%.3f\t",
                    world[
                        i * this->world_dim_sizes.y * this->world_dim_sizes.x +
                        j * this->world_dim_sizes.x +
                        k
                    ]
                );
                assert(retval > 0);
            }
            retval = fprintf(file, "\n");
            assert(retval == 1);
        }
        retval = fprintf(file, "-------------------------------\n");
        assert(retval > 0);
    }

    retval = fclose(file);
    assert(retval == 0);
};

/**
 * Test results of the simulation against reference CPU results
 */
void LeniaSimulation::test_results() {
    int retval;

    // Open output file, read from it to a one-dim array of floats and compare to this->h_world_src contents
    FILE* ref_world_file = fopen(this->ref_output_file_path, "r");
    assert(ref_world_file != NULL);

    float* ref_world_data = (float*) calloc(this->world_n_grid_points, sizeof(float));
    assert(ref_world_data != NULL);

    // Read first line
    int dims, dim_x, dim_y, dim_z;
    retval = fscanf(ref_world_file, "%d %d %d %d\n", &dims, &dim_x, &dim_y, &dim_z);
    assert(retval == 4);
    assert(dims == this->dims);
    assert(dim_x == this->world_dim_sizes.x);
    assert(dim_y == this->world_dim_sizes.y);
    assert(dim_z == this->world_dim_sizes.z);

    // Read world data into ref_contents
    for (int i = 0; i < this->world_dim_sizes.z; i++) {
        for (int j = 0; j < this->world_dim_sizes.y; j++) {
            for (int k = 0; k < this->world_dim_sizes.x; k++) {
                retval = fscanf(ref_world_file, "%f\t", &ref_world_data[
                    i * this->world_dim_sizes.y * this->world_dim_sizes.x +
                    j * this->world_dim_sizes.x +
                    k
                ]);
                assert(retval == 1);
            }
            retval = fscanf(ref_world_file, "\n");
            assert(!retval);
        }
        retval = fscanf(ref_world_file, "-------------------------------\n");
        assert(!retval);
    }

    // Compare results
    for (size_t i = 0; i < this->world_n_grid_points; i++) {
        if (fabs(this->h_world_dest[i] - ref_world_data[i]) > 0.01) {
            fprintf(
                stderr,
                "Results do not match at index %lu: %f != %f\n",
                i,
                this->h_world_dest[i],
                ref_world_data[i]
            );
            break;
        }
    }

    free_null(ref_world_data);
    retval = fclose(ref_world_file);
    assert(!retval);
}
