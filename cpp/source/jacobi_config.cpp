#include "jacobi_config.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

//Read test arguments from CLI
void parse_args(int argc, char** argv, JacobiConfig* config) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            config->n = (dim_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            config->block_size = (dim_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            config->n_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) {
            config->seed = atoi(argv[++i]);  
        } else if (strcmp(argv[i], "--debug") == 0) {
            config->debug = true;
        } else if (strcmp(argv[i], "--printi-initial-tensor") == 0) {
            config->print_initial_tensor = true;
        } else if (strcmp(argv[i], "--print-final-tensor") == 0) {
            config->print_final_tensor = true;
        } else if (strcmp(argv[i], "--print-factor-matrix") == 0) {
            config->print_factor_matrix = true;
        } else if (strcmp(argv[i], "--print-partition") == 0) {
            config->print_partition = true;
        } else if (strcmp(argv[i], "--no-matlab-summary") == 0) {
            config->print_matlab_summary = false;
        }        
        else {
            printf("Unknown argument: %s\\n", argv[i]);
            exit(1);
        }
    }
}

