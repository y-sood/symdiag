#pragma once
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef abs
#undef abs
#endif

struct JacobiConfig {
    dim_t order;
    dim_t n;
    dim_t block_size;
    int n_iterations;
    bool debug;
    bool print_initial_tensor;
    bool print_final_tensor;
    bool print_factor_matrix;
    bool print_partition;
    bool print_matlab_summary;
    int seed;
};

JacobiConfig default_jacobi_config();
void parse_args(int argc, char** argv, JacobiConfig* config);
void print_run_header(const JacobiConfig& config);

