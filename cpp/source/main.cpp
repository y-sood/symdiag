//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

//Hardcoded to 3rd order tensor for now
#define T_ORDER 3

//Linked files
#include "jacobi_config.hpp"
#include "jacobi.hpp"
#include "reporting.hpp"
#include "checking.hpp"
#include "partition.hpp"
#include "tensor_utils.hpp"

//C++ header files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]){
    // Test configuration - Default
    JacobiConfig config;
    config.order = T_ORDER;
    config.n = 8;
    config.block_size = 4;
    config.n_iterations = 10;
    config.debug = false;
    config.print_initial_tensor = false;
    config.print_final_tensor = false;
    config.print_factor_matrix = false;
    config.print_partition = false;
    config.print_matlab_summary = true; 
    config.seed = 42;

    //Arguments from CLI - Overwrite config
    parse_args(argc, argv, &config);

    //Self-describe
    printf("LibFLAME based Jacobi Tensor Diagonalization\\n");
    printf("n = %ld\\n", (long)config.n);
    printf("block_size = %ld\\n", (long)config.block_size);
    printf("n_iterations = %d\\n", config.n_iterations);
    printf("debug = %d\\n", config.debug ? 1 : 0);
    
    if (config.n % config.block_size != 0) {
        printf("ERROR: Mode length must be divisible by block size\n");
        return 1;
    }
    
    FLA_Init();
    printf("Initialized libflame\n\n");
    
    // Setup
    FLA_Obj A, B, B_temp, C;
    FLA_Obj A_initial, D_diag, T_reconstructed;
    JacobiPartition vpartition;
    dim_t tSize[FLA_MAX_ORDER];

    // Setup (allocates A, B, B_temp, C)
    setup_jacobi(&A, &B, &B_temp, &C, config, &vpartition, tSize, config.seed);
    
    //Setup tensor and matrix objects for checking
    initSymmTensor(config.order, tSize, config.block_size, &A_initial);
    FLA_Set_zero_tensor(A_initial);
    copy_tensor_values(A, A_initial);
    fill_intra_block_symmetry(A_initial, config.order, config.block_size);

    // Workspace tensors for reconstruction-aware checker
    initSymmTensor(config.order, tSize, config.block_size, &D_diag);
    FLA_Set_zero_tensor(D_diag);

    initSymmTensor(config.order, tSize, config.block_size, &T_reconstructed);
    FLA_Set_zero_tensor(T_reconstructed);

    //Print initial state
    if(config.debug) print_initial_state(&config, A, B, &vpartition, config.n, config.order);

    // Launch diagonalisation loop (uses B_temp as workspace)
    jacobi_diagonalization(&A, &B, &B_temp, &C, config, &vpartition, tSize);
    //Verify state of diagonalisation
    check_diagonalization_with_reconstruction(A, B, A_initial, D_diag, T_reconstructed, config.n, config.order, 1e-6);
    
    //Print final state 
    if(config.debug) print_final_state(&config, A, B, config.n, config.order);
    
    //All done!
    cleanup_jacobi(&A, &B, &B_temp, &C);
    cleanup_tensor(&A_initial);
    cleanup_tensor(&D_diag);
    cleanup_tensor(&T_reconstructed);
    cleanup_partition(&vpartition);
    
    FLA_Finalize();
    printf("\nProgram completed successfully!\n");
    
    return 0;
}
