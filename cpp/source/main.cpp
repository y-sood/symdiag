//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

//Default tensor order; override with --order
#define T_ORDER 3

//Linked files
#include "jacobi_config.hpp"
#include "jacobi.hpp"
#include "reporting.hpp"
#include "checking.hpp"
#include "partition.hpp"
#include "tensor_utils.hpp"
#include "tensor_io.hpp"

//C++ header files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <memory>
#include <string>


int main(int argc, char* argv[]){
    // Test configuration - Uses default
    JacobiConfig config;
    config.order = T_ORDER;
    int input_scale_exp = 0;   // external input rescaled by 2^-input_scale_exp (see tensor_io.cpp)

    //Arguments from CLI - Overwrite config
    parse_args(argc, argv, &config);

    //External input: order and n come from the file
    if (!config.input_path.empty()) {
        std::string err;
        auto L = std::make_shared<LoadedTensor>();
        if (!load_tensor_file(config.input_path, config.index_base, config.sym_tol, L.get(), &err)) {
            printf("ERROR reading '%s': %s\n", config.input_path.c_str(), err.c_str());
            return 1;
        }
        if (config.order_set && (size_t)config.order != (size_t)L->order) {
            printf("ERROR: --order %ld does not match the input file (order %d)\n", (long)config.order, L->order);
            return 1;
        }
        if (config.n_set && (size_t)config.n != L->n) {
            printf("ERROR: --n %ld does not match the input file (n = %zu)\n", (long)config.n, L->n);
            return 1;
        }
        config.order = (dim_t)L->order;
        config.n = (dim_t)L->n;
        if (config.generic) { printf("ERROR: --generic and --input cannot be combined\n"); return 1; }
        if (!config.block_size_set) {
            config.block_size = (dim_t)largest_divisor_up_to(L->n, 16);
            printf("note: --block-size not given; using %ld (largest divisor of n=%zu that is <= 16)\n", (long)config.block_size, L->n);
            if (config.block_size < 4) printf("warning: small block size %ld is memory-hungry (dense block-pointer array); consider an n with a larger divisor\n", (long)config.block_size);
        }
        printf("input = %s (%s, %zu non-zero canonical entries, max |entry| = %.6e)\n", config.input_path.c_str(), L->format.c_str(), L->entries(), L->max_abs);
        if (L->scale_exp != 0)
            printf("note: input scale is extreme; values were multiplied by 2^%d (exact). Printed tensor quantities are in scaled units; the --output-prefix diagonal is written in the ORIGINAL units.\n", -L->scale_exp);
        input_scale_exp = L->scale_exp;
        config.loaded = L;
    }

    //Self-describe
    printf("LibFLAME based Jacobi Tensor Diagonalization\\n");
    printf("order = %ld\\n", (long)config.order);
    printf("n = %ld\\n", (long)config.n);
    printf("block_size = %ld\\n", (long)config.block_size);
    printf("n_iterations = %d\\n", config.n_iterations);
    printf("debug = %d\\n", config.debug ? 1 : 0);
    printf("fast_unsafe = %d\\n", config.fast_unsafe ? 1 : 0);
    
    if (config.order < 3 || config.order > 9) {
        printf("ERROR: --order must be between 3 and 9\n");
        return 1;
    }
    if (config.n % config.block_size != 0) {
        printf("ERROR: Mode length n=%ld must be divisible by the block size %ld. Divisors of n:", (long)config.n, (long)config.block_size);
        for (dim_t d = 1; d <= config.n; ++d) if (config.n % d == 0) printf(" %ld", (long)d);
        printf("\n");
        return 1;
    }
    
    FLA_Init();
    if (config.fast_unsafe) {
        FLA_Check_error_level_set(FLA_NO_ERROR_CHECKING);
        printf("fast_unsafe = 1 (libflame FLA_Check_* validation disabled)\n");
    }
    printf("Initialized libflame\n\n");
    
    // Setup: T (A) and the factor matrix (B) are the only objects alive during the sweeps.
    FLA_Obj A, B;
    FLA_Obj A_initial;
    JacobiPartition vpartition;
    dim_t tSize[FLA_MAX_ORDER];

    setup_jacobi(&A, &B, config, &vpartition, tSize);

    // Reference copy of the input for validation. An arbitrary input tensor has no seed
    // to regenerate it from, so this is a real extra tensor for the whole run (--no-check
    // skips it). Only canonical positions are read from it.
    if (config.check) {
        initSymmTensor(config.order, tSize, config.block_size, &A_initial);
        FLA_Set_zero_tensor(A_initial);
        copy_tensor_values(A, A_initial);
    }

    //Print initial state
    if(config.debug) print_initial_state(&config, A, B, &vpartition, config.n, config.order);

    // Timing
    auto jacobi_start = std::chrono::high_resolution_clock::now();
    
    // Launch diagonalisation loop
    jacobi_diagonalization(&A, &B, config, &vpartition, tSize);
    
    // Timing
    auto jacobi_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> jacobi_duration = jacobi_end - jacobi_start;
    
    // Output to the same log file that shell script parses
    printf("jacobi_elapsed_s = %.10f\n", jacobi_duration.count());

    //Verify state of diagonalisation
    if (config.check) {
        check_diagonalization_with_reconstruction(A, B, A_initial, config.n, config.order, 1e-6);
        if (config.check_full) full_reconstruction_report(A, B, A_initial, config.n, config.order);
    } else {
        printf("\ncheck skipped (--no-check): no reference copy of the input was kept\n");
    }

    //Result files for downstream use
    if (!config.output_prefix.empty()) {
        std::vector<double> dg(config.n), Fm((size_t)config.n * config.n);
        dim_t di[FLA_MAX_ORDER];
        for (dim_t i = 0; i < config.n; ++i) {
            for (dim_t m = 0; m < config.order; ++m) di[m] = i;
            dg[i] = std::ldexp(get_tensor_element_bccs_alt(A, di, config.order), input_scale_exp);
            for (dim_t j = 0; j < config.n; ++j) Fm[(size_t)i * config.n + j] = get_dense_matrix_element(B, i, j);
        }
        std::string err;
        if (write_result_files(config.output_prefix, dg, Fm, config.n, &err))
            printf("wrote %s_diag.csv and %s_F.csv (T_init = sum_r diag_r * F[:,r]^(x)order, F orthogonal)\n", config.output_prefix.c_str(), config.output_prefix.c_str());
        else printf("WARNING: %s\n", err.c_str());
    }

    //Print final state 
    if(config.debug) print_final_state(&config, A, B, config.n, config.order);
    
    //All done!
    cleanup_jacobi(&A, &B);
    if (config.check) cleanup_tensor(&A_initial);
    cleanup_partition(&vpartition);
    
    FLA_Finalize();
    printf("\nProgram completed successfully!\n");
    
    return 0;
}
