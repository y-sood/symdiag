#include "jacobi_config.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static StopMode parse_stop_mode(const char* s) {
    if (strcmp(s, "delta") == 0) {
        return StopMode::Delta;
    } else if (strcmp(s, "ratio") == 0) {
        return StopMode::Ratio;
    } else if (strcmp(s, "both") == 0) {
        return StopMode::Both;
    } else if (strcmp(s, "either") == 0) {
        return StopMode::Either;
    } else if (strcmp(s, "fixed") == 0) {
        return StopMode::Fixed;
    } else {
        printf("Invalid value for --stop-mode: %s\n", s);
        printf("Expected one of: fixed, delta, ratio, both, either\n");
        exit(1);
    }
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --input <file>         External symmetric tensor: *.npy (dense) or text/CSV 'i,j,k,...,value'\n");
    printf("                         (order and n are taken from the file; see source/tensor_io.hpp)\n");
    printf("  --index-base <0|1>     Index base of a text/CSV input (default 0)\n");
    printf("  --sym-tol <double>     Relative symmetry tolerance for external input (default 1e-12)\n");
    printf("  --output-prefix <path> Write <path>_diag.csv (final diagonal) and <path>_F.csv (factor matrix)\n");
    printf("  --order <int>          Tensor order (default 3; 3..9)\n");
    printf("  --generic              Random symmetric input (not exactly diagonalisable)\n");
    printf("  --no-check             Skip validation: no reference copy of the input is kept\n");
    printf("  --check-full           Also reconstruct from all entries of T_final (dense scratch)\n");
    printf("  --n <int>\n");
    printf("  --block-size <int>\n");
    printf("  --max-iters <int>\n");
    printf("  --seed <int>\n");
    printf("  --debug\n");
    printf("  --print-initial-tensor\n");
    printf("  --print-final-tensor\n");
    printf("  --print-factor-matrix\n");
    printf("  --print-partition\n");
    printf("  --no-matlab-summary\n");
    printf("\nStopping options:\n");
    printf("  --tol-delta <double>\n");
    printf("  --tol-ratio <double>\n");
    printf("  --min-iters <int>\n");
    printf("  --eps-pivot <double>   Skip a pivot pair when all its pivot off-diagonal entries are below\n");
    printf("                  eps_pivot * ||T||_F (scale-aware; default 1e-15)\n");
    printf("  --eps-sine <double>    A rotation is applied only if |sin(theta)| > this (default 1e-15)\n");
    printf("  --tol-off <double>     Also stop when ||offdiag||_F/||T||_F < this (default 0 = off)\n");
    printf("  --stop-mode <fixed|delta|ratio|both|either>  (default fixed: stop when a sweep applies no rotation)\n");
    printf("  --require-both\n");
    printf("  --disable-stopping\n");
    printf("\nPerformance options:\n");
    printf("  --fast-unsafe   Disable libflame's internal FLA_Check_* validation (~2.6x faster,\n");
    printf("                  but also silences libflame's own error messages)\n");
}

// Read test arguments from CLI
void parse_args(int argc, char** argv, JacobiConfig* config) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            config->input_path = argv[++i];

        } else if (strcmp(argv[i], "--index-base") == 0 && i + 1 < argc) {
            config->index_base = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--sym-tol") == 0 && i + 1 < argc) {
            config->sym_tol = atof(argv[++i]);

        } else if (strcmp(argv[i], "--output-prefix") == 0 && i + 1 < argc) {
            config->output_prefix = argv[++i];

        } else if (strcmp(argv[i], "--order") == 0 && i + 1 < argc) {
            config->order = (dim_t)atoi(argv[++i]);
            config->order_set = true;

        } else if (strcmp(argv[i], "--check-full") == 0) {
            config->check_full = true;

        } else if (strcmp(argv[i], "--no-check") == 0) {
            config->check = false;

        } else if (strcmp(argv[i], "--generic") == 0) {
            config->generic = true;

        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            config->n = (dim_t)atoi(argv[++i]);
            config->n_set = true;

        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            config->block_size = (dim_t)atoi(argv[++i]);
            config->block_size_set = true;

        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            config->n_iterations = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config->seed = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--tol-delta") == 0 && i + 1 < argc) {
            config->tol_delta = atof(argv[++i]);

        } else if (strcmp(argv[i], "--tol-ratio") == 0 && i + 1 < argc) {
            config->tol_ratio = atof(argv[++i]);

        } else if (strcmp(argv[i], "--min-iters") == 0 && i + 1 < argc) {
            config->min_iterations = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--eps-pivot") == 0 && i + 1 < argc) {
            config->eps_pivot = atof(argv[++i]);

        } else if (strcmp(argv[i], "--eps-sine") == 0 && i + 1 < argc) {
            config->eps_sine = atof(argv[++i]);

        } else if (strcmp(argv[i], "--tol-off") == 0 && i + 1 < argc) {
            config->tol_off = atof(argv[++i]);

        } else if (strcmp(argv[i], "--stop-mode") == 0 && i + 1 < argc) {
            config->stop_mode = parse_stop_mode(argv[++i]);

        } else if (strcmp(argv[i], "--require-both") == 0) {
            config->require_both = true;

        } else if (strcmp(argv[i], "--disable-stopping") == 0) {
            config->enable_stopping = false;

        } else if (strcmp(argv[i], "--fast-unsafe") == 0) {
            config->fast_unsafe = true;

        } else if (strcmp(argv[i], "--debug") == 0) {
            config->debug = true;

        } else if (strcmp(argv[i], "--print-initial-tensor") == 0) {
            config->print_initial_tensor = true;

        } else if (strcmp(argv[i], "--print-final-tensor") == 0) {
            config->print_final_tensor = true;

        } else if (strcmp(argv[i], "--print-factor-matrix") == 0) {
            config->print_factor_matrix = true;

        } else if (strcmp(argv[i], "--print-partition") == 0) {
            config->print_partition = true;

        } else if (strcmp(argv[i], "--no-matlab-summary") == 0) {
            config->print_matlab_summary = false;

        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);

        } else {
            printf("Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
}

