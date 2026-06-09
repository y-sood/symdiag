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

enum class StopMode {
    Delta,
    Ratio,
    Both,
    Either
};

struct JacobiConfig {
    //Tensor generation parameters
    dim_t order;
    dim_t n = 8;
    int seed = 42;
    //BCCS parameter
    dim_t block_size = 4;
    //Jacobi parameter
    int n_iterations = 10;
    //Debugging
    bool debug = false;
    bool print_initial_tensor = false;
    bool print_final_tensor = false;
    bool print_factor_matrix = false;
    bool print_partition = false;
    bool print_matlab_summary = false;
    //Stopping criteria
    bool enable_stopping = true;
    double tol_delta = 1e-10; 
    double tol_ratio = 1e-8;
    int min_iterations = 2;
    bool require_both = false;
    StopMode stop_mode = StopMode::Either;
};

JacobiConfig default_jacobi_config();
void parse_args(int argc, char** argv, JacobiConfig* config);
void print_run_header(const JacobiConfig& config);

