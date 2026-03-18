#pragma once

class LeniaSimulation {
    public:
        int dims;
        int n_iters;
        float delta_t;

        int world_side_size;
        int world_n_grid_points;
        size_t world_mem_size;
        dim3 world_dim_sizes;

        int block_side_size;
        int block_n_grid_points;
        size_t block_mem_size;
        dim3 block_dim_sizes;

        int R;
        int stencil_diameter;
        float delta_R;
        // Number of grid points in the entire conv kernel
        size_t conv_kernel_n_grid_points;
        size_t conv_kernel_mem_size;

        // for sim on CPU
        float* world_src;
        float* world_dest;

        // for sim on GPU
        float* h_world_src;
        float* h_world_dest;    // Final result shoult be copied here
        float* d_world_src;
        float* d_world_dest;
        float* d_last_dest;
        float* d_conv_kernel;

        float* h_conv_kernel;

        dim3 grid_dim_sizes;

        // Number of grid points in the entire cache

        SynchronizationMode sync_mode;

        CachingMode caching_mode;
        size_t cache_n_grid_points;
        size_t cache_mem_size;
        dim3 cache_dim_sizes;
        SimulationDevice sim_device;
        bool use_subplaning;

        // CPU execution time vars
        double start_time;
        double stop_time;

        // GPU kernel execution time vars
        float gpu_exec_time;
        cudaEvent_t start;
        cudaEvent_t stop;
        double exec_time;

        // Base result dir path (before the curr/ref/device name)
        char* base_results_dir;


        // Reference output file path
        char* ref_output_file_path;
        // Reference results directory path of the CPU sim in test mode
        char* ref_results_dir;
        // Reference source world dump file path
        char* ref_src_world_dump_path;

        // Results dir path for the current GPU sim
        char* curr_output_file_path;
        // Results file path of the current GPU sim
        char* curr_results_dir;
        // Current GPU sim's source world dump file path
        char* curr_src_world_dump_path;


        bool test_mode;
        bool debug_mode;

        LeniaSimulation();
        // ~LeniaSimulation();

        void set_block_size_params();
        void set_world_size_params();
        void set_time_params(int n_iters);
        void set_stencil_params();
        void set_grid_size();
        void set_cache_params();
        void set_cuda_kernel();
        void precompute_conv_kernel();
        void init_host_world_mem();
        void init_gpu_mem();
        void start_timer();
        void stop_timer();
        void gpu_runtime_checks();
        void run_gpu_sim();
        void run_cpu_sim();
        void set_base_results_dir_path();
        void set_ref_output_file_path();
        void set_ref_src_world_dump_path();
        void set_ref_results_dir_path();
        void set_curr_output_file_path();
        void set_curr_src_world_dump_path();
        void set_curr_results_dir_path();
        // void ensure_ref_results_dir_path();
        // void ensure_curr_results_dir_path();
        void ensure_dir_path(char* path);
        void dump_world(float* world, char* filename);
        // void ensure_output_path();
        void test_results();
        void transfer_h2d();
        void transfer_d2h();
        void free_mem();

        void (*kernel)(float*, float*, float*, dim3, int, int);
};
