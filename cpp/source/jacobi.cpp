#include "jacobi.hpp"
#include "tensor_utils.hpp"
#include "angle.hpp"
#include "reporting.hpp"
#include "checking.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

static inline bool should_stop(double delta, double ratio, const JacobiConfig& config){
    bool cond_delta = delta < config.tol_delta;
    bool cond_ratio = ratio < config.tol_ratio;

    bool stop = false;
    switch (config.stop_mode) {
        case StopMode::Delta:  stop = cond_delta; break;
        case StopMode::Ratio:  stop = cond_ratio; break;
        case StopMode::Both:   stop = cond_delta && cond_ratio; break;
        case StopMode::Either: stop = cond_delta || cond_ratio; break;
        default:               stop = false; break;
    }

    if (config.require_both) {
        stop = cond_delta && cond_ratio;
    }

    return stop;
}

void setup_jacobi(FLA_Obj* T, FLA_Obj* F, FLA_Obj* F_temp, FLA_Obj* O, const JacobiConfig& config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER], int seed){   
    //Setup
    dim_t i;
    int mSize = config.n;
    
    // Set all dimension to n - Symmetric
    for(i = 0; i < config.order; i++)
        tSize[i] = config.n;
    
    //Initialise
    
    //INPUT TENSOR - Diagonalisable by construction
    initDiagonalizableTensor(config.order, tSize, config.block_size, T, mSize, seed);
    // Fill intra-block non-canonical positions (set_tensor_element_bccs only writes
    // sorted-index positions; STTSM reads all positions within each block).
    fill_intra_block_symmetry(*T, config.order, config.block_size);
    
    //Partition
    //Partition to get disjoint PQ pairs
    partition_stats(config.n, &vpartition->nr_groups, &vpartition->group_size);
    vpartition->pairs =(PQPair**)malloc(vpartition->nr_groups * vpartition->group_size * sizeof(PQPair*));
    for (int i = 0; i < vpartition->nr_groups * vpartition->group_size; i++) { 
        vpartition->pairs[i] = (PQPair*)malloc(sizeof(PQPair));}
    partition(vpartition->pairs, config.n, &vpartition->nr_groups, &vpartition->group_size);

    //Output Tensor (Should always be zeroed for correct STTSM)
    initSymmTensor(config.order, tSize, config.block_size, O);
    FLA_Set_zero_tensor(*O);

    //Factor matrices
    initIdentityDenseMatrix(config.n, F);
    initZeroDenseMatrix(config.n, F_temp); 

    // Print tensor after this group
    if(config.debug == true) FLA_Obj_print_matlab("Initial T", *T);
}

//Main diagonalisation loop - With debug prints
void jacobi_diagonalization(FLA_Obj* T, FLA_Obj* F, FLA_Obj* F_temp, FLA_Obj* O, const JacobiConfig& config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER]){
    dim_t n = config.n;
    dim_t order = config.order;
    dim_t mSize = config.n;

    printf("\n%%%% ============================================\n");
    printf("%%%% JACOBI DIAGONALIZATION ALGORITHM\n");
    printf("%%%% ============================================\n");

    //Constants for STTSM
    FLA_Obj alpha = FLA_ONE;
    FLA_Obj beta = FLA_ONE;

    //Givens rotations allocation
    FLA_Obj G_sttsm, G_fac;
    initIdentityMatrix(config.n, config.block_size, config.block_size, &G_sttsm); //For STTSM operation
    initIdentityDenseMatrix(config.n, &G_fac); //For matrix multiplication

    double prev_rel_off = std::numeric_limits<double>::infinity();
    bool converged = false;
    int performed_iters = 0;

    //Perform iterations of Jacobi diagonalisation
    for (int iter = 0; iter < config.n_iterations; iter++) {
        //Loop over disjoint groups of PQ pairs
        for (int group = 0; group < vpartition->nr_groups; group++) {
            if(config.debug) printf("    Processing group %d/%d\n", group + 1, vpartition->nr_groups);

            //Reset O
            FLA_Set_zero_tensor(*O);
            if(config.debug) printf("Output tensor zeroed successfully\n");
            
            //Reset
            setIdentityMatrix(n, &G_sttsm);
            setIdentityDenseMatrix(n, &G_fac);

            if(config.debug){ 
                FLA_Obj_print_matlab("Initialised Givens Rotation Matrix G", G_sttsm);
                printf("Identity matrix initiated successfully\n");
            }

            //Calculate and embed rotations for group into G
            //Currently sequential - Must convert to kernel
            for(int pair_idx = 0; pair_idx < vpartition->group_size; pair_idx++)
                {
                    //Get p and q from partition
                    int idx = group * vpartition->group_size + pair_idx;
                    int p = vpartition->pairs[idx]->p;
                    int q = vpartition->pairs[idx]->q;

                    //Fill indice arrays
                    dim_t ppp[3] = {(dim_t)p, (dim_t)p, (dim_t)p};
                    dim_t qqq[3] = {(dim_t)q, (dim_t)q, (dim_t)q};
                    dim_t qpp[3] = {(dim_t)q, (dim_t)p, (dim_t)p};
                    dim_t pqq[3] = {(dim_t)p, (dim_t)q, (dim_t)q}; 

                    //Store coefficients to pass (4 values: ppp, qqq, ppq, pqq)
                    double A_arr[4];
                    //Retrieve elements stored at above indices
                    A_arr[0] = get_tensor_element_bccs(*T, p, p, p);
                    A_arr[1] = get_tensor_element_bccs(*T, q, q, q);
                    double Appq = get_tensor_element_bccs(*T, p, p, q);
                    double Aqqp = get_tensor_element_bccs(*T, p, q, q);
                    A_arr[2] = Appq;
                    A_arr[3] = Aqqp;

                    double c,s;
                    //FILTER - FLAG
                    if(fabs(Appq) > 1e-6 || fabs(Aqqp) > 1e-6){
                        //Calculate rotation angle - Works only for third order tensor
                        calculate_rotation_angle(*T, A_arr, order, &c, &s);
                        if(config.debug){
                            printf("Rotation angle calculated successfully\n");
                            //Print detailed rotation info
                            print_rotation_details(iter, group, pair_idx, p, q, c, s, *T, order);
                        }
                    }
                    else{
                        if(config.debug) printf("Rotation skipped since elements %lf and %lf are insignificant\n", Appq, Aqqp);
                        c = 1.0; s = 0.0;
                    }

                    //FILTER
                    //Embed rotation into G only if significant
                    double abs_sine = fabs(s);
                    if(abs_sine > 1e-6){
                        embed_givens_rotation(G_sttsm, p, q, c, s);
                        embed_givens_rotation_dense(G_fac, p, q, c, s);
                    }
                    else{
                        if(config.debug) printf("Rotation insignificant sin = %f and cos = %f, SKIPPING\n", s, c);
                    }

                    if(config.debug) printf("Rotation embedded successfulyy\n");
                }

            if(config.debug){
                FLA_Obj_print_matlab("Embedded Givens Rotation Matrix G", G_sttsm);
                FLA_Obj_print_matlab("Pre-STTSM Tensor", *T);
            }

            //Rotate Tensor
            //STTSM currently sequential - Must convert to sequential
            FLA_Sttsm_with_psym_temps(alpha, *T, beta, G_sttsm, *O);
            if(config.debug){
                printf("Rotation applied successfully\n");
                FLA_Obj_print_matlab("Post-STTSM Tensor", *O);
            }
            if(config.debug) { 
                print_dense_matrix_matlab("F_before_gem", *F);
                print_dense_matrix_matlab("G_before_gem", G_fac);
                print_dense_matrix_matlab("F_temp_before_gem", *F_temp);
            }
            //Update factor matrix - Can switch out with CUDA kernel
            FLA_Set(FLA_ZERO, *F_temp);
            FLA_Gemm(FLA_NO_TRANSPOSE, FLA_TRANSPOSE, FLA_ONE, *F, G_fac, FLA_ZERO, *F_temp);
            if(config.debug) { 
                printf("Factor matrix updated successfully\n");
                print_dense_matrix_matlab("FTemp_after_gemm", *F_temp);
            }
            copy_tensor_values(*O, *T);
            // Restore intra-block symmetry: STTSM may not write non-canonical local
            // positions within canonical blocks, leaving them 0 (from FLA_Set_zero_tensor).
            fill_intra_block_symmetry(*T, order, config.block_size);
            FLA_Copy(*F_temp, *F);
            
            if(config.debug){ 
                printf("Copy of tensor successful\n");
                print_dense_matrix_matlab("F_after copy", *F);
            }

            //Cleanup G and restart next
            if(config.debug) printf("Cleanup successful, next group\n");
        }

        //Calculate norms per iteration
        double diag_sq = diag_norm_sq_tensor(*T, n, order);
        double off_sq  = offdiag_norm_sq_tensor(*T, n, order);
        double frob_sq = diag_sq + off_sq;
        double ratio   = off_sq / fmax(diag_sq, 1e-300);
        double rel_off = sqrt(fmax(off_sq, 0.0)) / fmax(sqrt(fmax(frob_sq, 1e-300)), 1e-300);
        double trace   = tensor_trace_general(*T, n, order);
        //Check stopping criteria based on norms
        double delta = (iter == 0) ? std::numeric_limits<double>::infinity() : fabs(prev_rel_off - rel_off);
        bool stop = false;
        if (config.enable_stopping && (iter + 1) >= config.min_iterations) {stop = should_stop(delta, rel_off, config);}
        //Iteration details
        printf("SWEEP iter=%d diag_norm_sq=%.15e offdiag_norm_sq=%.15e " "ratio=%.15e rel_offdiag=%.15e delta=%.15e trace=%.15e stop=%d\n", iter + 1, diag_sq, off_sq, ratio, rel_off, delta, trace, (int)stop);
        //Update parameters for next iteration
        prev_rel_off = rel_off;
        performed_iters = iter + 1;
        //Stop iterations
        if (stop) { converged = true;
        printf("Converged at iter %d: delta=%.15e rel_off=%.15e\n", iter + 1, delta, rel_off);
        break; //End loop
        }
    }
    printf("Jacobi finished after %d iteration(s), converged=%d\n", performed_iters, (int)converged);
    //Cleanup Givens
    cleanup_matrix(&G_fac);
    cleanup_tensor(&G_sttsm);
}

void cleanup_jacobi(FLA_Obj* T, FLA_Obj* F, FLA_Obj* F_temp, FLA_Obj* O){
    cleanup_tensor(T);
    cleanup_matrix(F);
    cleanup_matrix(F_temp);
    cleanup_tensor(O);
}
