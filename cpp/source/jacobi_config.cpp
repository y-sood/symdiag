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
    } else {
        printf("Invalid value for --stop-mode: %s\n", s);
        printf("Expected one of: delta, ratio, both, either\n");
        exit(1);
    }
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
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
    printf("  --stop-mode <delta|ratio|both|either>\n");
    printf("  --require-both\n");
    printf("  --disable-stopping\n");
}

// Read test arguments from CLI
void parse_args(int argc, char** argv, JacobiConfig* config) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            config->n = (dim_t)atoi(argv[++i]);

        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            config->block_size = (dim_t)atoi(argv[++i]);

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

        } else if (strcmp(argv[i], "--stop-mode") == 0 && i + 1 < argc) {
            config->stop_mode = parse_stop_mode(argv[++i]);

        } else if (strcmp(argv[i], "--require-both") == 0) {
            config->require_both = true;

        } else if (strcmp(argv[i], "--disable-stopping") == 0) {
            config->enable_stopping = false;

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

