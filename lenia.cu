/// General includes

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

/// C++ includes

#include <cmath>

/// CUDA related includes

#include <cuda.h>
#include <cuda_runtime.h>

/// Custom libs

#include <libutil.cuh>
#include <LeniaSimulation.cuh>


/// Defines

#define WORLD_SEED 0x1337c0d3


/// Globals

// Simulation singleton
LeniaSimulation sim;


void gpu() {
    sim.set_block_size_params();
    sim.set_stencil_params();
    sim.init_gpu_mem();
    sim.set_grid_size();
    sim.set_cache_params();
    sim.set_cuda_kernel();

    printf(
        "Grid size: %d x %d x %d blocks\n",
        sim.grid_dim_sizes.x,
        sim.grid_dim_sizes.y,
        sim.grid_dim_sizes.z
    );
    printf(
        "Block size: %d x %d x %d threads\n",
        sim.block_dim_sizes.x,
        sim.block_dim_sizes.y,
        sim.block_dim_sizes.z
    );
    puts("Running simulation on GPU...");

    sim.start_timer();
    sim.run_gpu_sim();
    sim.stop_timer();

    sim.transfer_d2h();

    if (sim.test_mode) {
        // Dump worlds of GPU sim
        sim.dump_world(sim.h_world_dest, sim.curr_output_file_path);
        sim.test_results();
    }
}

/**
 * CPU implementation of the Lenia, reusing some CUDA code, compiled for the host
 */
void cpu() {
    puts("Running simulation on CPU...");

    sim.start_timer();
    sim.run_cpu_sim();
    sim.stop_timer();

    if (sim.test_mode) {
        // Inverted ptrs, because after each iter ptrs are always swapped
        sim.dump_world(sim.world_src, sim.ref_output_file_path);
    }
}

/**
 * Execution mode selector
 *
 * Seed random, precalc conv kernel, choose exec mode (CPU or GPU)
 * and invoke the simulation
 */
void prepare_execution() {
    srand(WORLD_SEED);

    if (sim.test_mode) {
        sim.ensure_dir_path(sim.base_results_dir);

        // Needs to be set in any case
        sim.set_ref_output_file_path();
        sim.ensure_dir_path(sim.ref_results_dir);

        // GPU
        if (sim.sim_device == SimulationDevice::GPU) {
            sim.set_curr_output_file_path();
            sim.set_curr_src_world_dump_path();
            sim.ensure_dir_path(sim.curr_results_dir);
        }
        // CPU
        else {
            sim.set_ref_src_world_dump_path();
        }
    }

    sim.set_world_size_params();

    sim.precompute_conv_kernel();
    sim.init_host_world_mem();

    if (sim.sim_device == SimulationDevice::GPU) {
        sim.gpu_runtime_checks();
        gpu();
    }
    else {
        cpu();
    }
    sim.free_mem();
}

void process_args(int argc, char* argv[]) {
    int option;
    int arg;
    while ((option = getopt(argc, argv, "gd:i:b:w:t:r:c:p:")) != -1) {
        if (optarg) {
            arg = sanitize_int(optarg);
        }
        switch (option) {
            case 'g':
                sim.sim_device = SimulationDevice::GPU;
                break;
            case 'p':
                assert((arg == 0 || arg == 1) && "Invalid P mode. Should be 0 or 1.");
                if (arg) {
                    sim.use_subplaning = true;
                } else {
                    sim.use_subplaning = false;
                }
                break;
            case 'd':
                assert((arg == 2 || arg == 3) && "Invalid number of dimensions. Should be 2 or 3.");
                sim.dims = arg;
                break;
            case 'i':
                assert((arg == 0 || arg == 1) && "Invalid sync mode. Should be 0 or 1.");
                sim.sync_mode = (SynchronizationMode) arg;
                break;
            case 'b':
                sim.block_side_size = arg;
                break;
            case 'w':
                assert((arg >= 4) && "Invalid world size.");
                sim.world_side_size = arg;
                break;
            case 't':
                assert((arg && !(arg % 2)) && "Invalid number of time steps. It has to be positive and divisible by 2.");
                sim.set_time_params(arg);
                break;
            case 'r':
                assert((arg >= 1) && "Invalid kernel radius.");
                sim.R = arg;
                sim.stencil_diameter = 2 * sim.R + 1;
                break;
            case 'c':
                assert((arg >= 0 && arg <= 3) && "Invalid caching mode. Should be in {0, 1, 2, 3}.");
                sim.caching_mode = (CachingMode) arg;
                break;
            case '?':
                error_exit(1, "Invalid or missing args.");
                break;
            default:
                error_exit(1, "Invalid args.");
                break;
        }
    }
}

void print_options() {
    puts("Running simulation with these parameters:");
    printf("Grid synchronization mode: %d\n", sim.sync_mode);
    printf("Number of dims: %d\n", sim.dims);
    printf("World size: %d ^ %d\n", sim.world_side_size, sim.dims);
    printf("Max time: %d\n", sim.n_iters);
    printf("Neighbourhood radius/convolution kernel radius: %d\n", sim.R);
    printf("Caching mode: %d\n", sim.caching_mode);
    printf("Thread domain: %d\n", sim.use_subplaning);
}

int main(int argc, char* argv[]) {
    process_args(argc, argv);
    print_options();
    prepare_execution();
    return 0;
}
