#include "jacobi.hpp"
#include "tensor_utils.hpp"
#include "angle.hpp"
#include "reporting.hpp"
#include "checking.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <chrono>

static inline bool should_stop(double delta, double ratio, const JacobiConfig& config){
    bool cond_delta = delta < config.tol_delta;
    bool cond_ratio = ratio < config.tol_ratio;

    bool stop = false;
    switch (config.stop_mode) {
        case StopMode::Delta:  stop = cond_delta; break;
        case StopMode::Ratio:  stop = cond_ratio; break;
        case StopMode::Both:   stop = cond_delta && cond_ratio; break;
        case StopMode::Either: stop = cond_delta || cond_ratio; break;
        case StopMode::Fixed:  stop = false; break;   // handled by the fixed-point rule in the sweep loop
        default:               stop = false; break;
    }

    if (config.require_both) {
        stop = cond_delta && cond_ratio;
    }

    return stop;
}

void make_input_tensor(const JacobiConfig& config, dim_t tSize[FLA_MAX_ORDER], FLA_Obj* T){
    // Set all dimension to n - Symmetric
    for(dim_t i = 0; i < config.order; i++)
        tSize[i] = config.n;

    if (config.loaded) {
        // External tensor: write the parsed canonical entries into BCSS, then release the parsed copy.
        initSymmTensor(config.order, tSize, config.block_size, T);
        FLA_Set_zero_tensor(*T);
        const LoadedTensor& L = *config.loaded;
        dim_t idx[FLA_MAX_ORDER];
        for (size_t e = 0; e < L.entries(); ++e) {
            for (dim_t m = 0; m < config.order; ++m) idx[m] = (dim_t)L.idx[e * config.order + m];
            set_tensor_element_bccs(*T, idx, config.order, L.val[e]);
        }
        config.loaded.reset();
    } else if (config.generic) {
        initSymmTensor(config.order, tSize, config.block_size, T);
        srand(config.seed);
        FLA_Random_psym_tensor(*T);
    } else {
        //Diagonalisable by construction
        initDiagonalizableTensor(config.order, tSize, config.block_size, T, (int)config.n, config.seed);
    }
}

void setup_jacobi(FLA_Obj* T, FLA_Obj* F, const JacobiConfig& config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER]){
    //INPUT TENSOR
    make_input_tensor(config, tSize, T);
    // Fill intra-block non-canonical positions (set_tensor_element_bccs only writes
    // sorted-index positions; the cellwise kernel and the norms read all positions).
    fill_intra_block_symmetry(*T, config.order, config.block_size);

    //Partition
    //Partition to get disjoint PQ pairs
    partition_stats(config.n, &vpartition->nr_groups, &vpartition->group_size);
    vpartition->pairs =(PQPair**)malloc(vpartition->nr_groups * vpartition->group_size * sizeof(PQPair*));
    for (int i = 0; i < vpartition->nr_groups * vpartition->group_size; i++) {
        vpartition->pairs[i] = (PQPair*)malloc(sizeof(PQPair));}
    partition(vpartition->pairs, config.n, &vpartition->nr_groups, &vpartition->group_size);

    //Factor matrix
    initIdentityDenseMatrix(config.n, F);

    // Print tensor after this group
    if(config.debug == true) FLA_Obj_print_matlab("Initial T", *T);
}

//Main diagonalisation loop - With debug prints
// Rotations are applied in place to T and F (no output tensor / temp matrix needed).
// Test hook: -DFORCE_GENERAL_PATH routes order 3 through the order-generic angle solve
// and cellwise kernel too, to cross-check them against the order-3 fast paths.
#ifdef FORCE_GENERAL_PATH
static const bool force_general = true;
#else
static const bool force_general = false;
#endif

void jacobi_diagonalization(FLA_Obj* T, FLA_Obj* F, const JacobiConfig& config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER]){
    dim_t n = config.n;
    dim_t order = config.order;
    dim_t mSize = config.n;

    printf("\n%%%% ============================================\n");
    printf("%%%% JACOBI DIAGONALIZATION ALGORITHM\n");
    printf("%%%% ============================================\n");

    // Per-pair rotation buffers and rotation-slot buffer for the cellwise tensor
    // update (replaces STTSM in the sweep; see CLAUDE.md "Planned replacement kernel").
    std::vector<int> pair_p(vpartition->group_size), pair_q(vpartition->group_size);
    std::vector<double> pair_c(vpartition->group_size), pair_s(vpartition->group_size);
    std::vector<RotSlot> slots(n);

    // ||T||_F is invariant under the rotations: it sets the scale for the pivot-skip threshold.
    const double tensor_scale = sqrt(diag_norm_sq_tensor(*T, n, order) + offdiag_norm_sq_tensor(*T, n, order));
    const double pivot_threshold = config.eps_pivot * tensor_scale;
    printf("tensor_frobenius_norm = %.15e   pivot_skip_threshold = %.6e (eps_pivot %.1e x ||T||_F)   eps_sine = %.1e\n",
           tensor_scale, pivot_threshold, config.eps_pivot, config.eps_sine);

    // Time spent in the per-sweep norm/trace computation (inside jacobi_elapsed_s): rotation time = jacobi_elapsed_s - norm_block_seconds.
    double norm_block_seconds = 0.0;
    double prev_rel_off = std::numeric_limits<double>::infinity();
    bool converged = false;
    int performed_iters = 0;

    //Perform iterations of Jacobi diagonalisation
    for (int iter = 0; iter < config.n_iterations; iter++) {
        long sweep_rotations = 0;   // rotations actually applied in this sweep (0 => exact fixed point)
        //Loop over disjoint groups of PQ pairs
        for (int group = 0; group < vpartition->nr_groups; group++) {
            if(config.debug) printf("    Processing group %d/%d\n", group + 1, vpartition->nr_groups);

            //Calculate and embed rotations for group into G
            //Currently sequential - Must convert to kernel
            bool any_active = false;
            for(int pair_idx = 0; pair_idx < vpartition->group_size; pair_idx++)
                {
                    //Get p and q from partition
                    int idx = group * vpartition->group_size + pair_idx;
                    int p = vpartition->pairs[idx]->p;
                    int q = vpartition->pairs[idx]->q;

                    //Pivot sub-tensor entries A_0..A_d: A_k has k copies of q, order-k of p.
                    //(order 3: A_0=Appp, A_1=Appq, A_2=Apqq, A_3=Aqqq)
                    double A_full[FLA_MAX_ORDER + 1];
                    for (dim_t k = 0; k <= order; k++){
                        dim_t idx[FLA_MAX_ORDER];
                        for (dim_t m = 0; m < order; m++) idx[m] = (m < k) ? (dim_t)q : (dim_t)p;
                        A_full[k] = get_tensor_element_bccs_alt(*T, idx, order);
                    }
                    bool significant = false;
                    for (dim_t k = 1; k < order; k++)
                        if (fabs(A_full[k]) > pivot_threshold) significant = true;

                    double c,s;
                    //FILTER - FLAG
                    if(significant){
                        if(order == 3 && !force_general){
                            //Order 3: exact closed form, order [ppp, qqq, ppq, pqq]
                            double A_arr[4] = {A_full[0], A_full[3], A_full[1], A_full[2]};
                            calculate_rotation_angle(*T, A_arr, order, &c, &s);
                        } else {
                            calculate_rotation_angle_general(A_full, order, &c, &s);
                        }
                        if(config.debug){
                            printf("Rotation angle calculated successfully\n");
                            //Print detailed rotation info
                            print_rotation_details(iter, group, pair_idx, p, q, c, s, *T, order);
                        }
                    }
                    else{
                        if(config.debug) printf("Rotation skipped since pivot off-diagonal elements are insignificant\n");
                        c = 1.0; s = 0.0;
                    }

                    //FILTER
                    //Record the pair's rotation for the cellwise tensor update below,
                    //and apply it directly to F's columns, only if significant
                    pair_p[pair_idx] = p; pair_q[pair_idx] = q;
                    pair_c[pair_idx] = c; pair_s[pair_idx] = s;
                    double abs_sine = fabs(s);
                    if(abs_sine > config.eps_sine){
                        sweep_rotations++;
                        apply_givens_rotation_to_columns(*F, p, q, c, s);
                        any_active = true;
                    }
                    else{
                        if(config.debug) printf("Rotation insignificant sin = %f and cos = %f, SKIPPING\n", s, c);
                    }

                    if(config.debug) printf("Rotation embedded successfulyy\n");
                }

            // No pair in this group cleared the significance threshold, so
            // build_group_rotation_slots would produce only identity slots:
            // the tensor rotation and symmetry refill are both no-ops. Skip them.
            if (!any_active) {
                if(config.debug) printf("    Group %d/%d has no active pairs, skipping tensor rotation\n", group + 1, vpartition->nr_groups);
                continue;
            }

            if(config.debug){
                FLA_Obj_print_matlab("Pre-rotation Tensor", *T);
            }

            //Rotate Tensor in place, cell by cell (replaces STTSM; see CLAUDE.md
            //"Planned replacement kernel" - each cell reads its old values before
            //writing any of them, so this is safe without a separate output tensor).
            int num_slots = build_group_rotation_slots(n, pair_p.data(), pair_q.data(),
                                                         pair_c.data(), pair_s.data(),
                                                         vpartition->group_size, slots.data(),
                                                         config.eps_sine);
            if (order == 3 && !force_general) apply_group_rotation_cellwise(*T, slots.data(), num_slots);
            else            apply_group_rotation_cellwise_general(*T, order, slots.data(), num_slots);
            if(config.debug){
                printf("Rotation applied successfully\n");
                FLA_Obj_print_matlab("Post-rotation Tensor", *T);
            }
            if(config.debug) {
                print_dense_matrix_matlab("F_before_gem", *F);
            }
            // No intra-block symmetry refill here: the cellwise kernel writes only canonical
            // (sorted-index) positions, and everything that reads T (the kernel itself, the norms,
            // the checks) goes through sorting accessors, so the stale non-canonical positions inside
            // diagonal blocks are never read. The refill used to cost ~half the runtime at n=128.
            // (Raw buffer dumps via FLA_Obj_print_matlab in --debug will show those positions stale.)

            if(config.debug){
                printf("Rotation write-back successful\n");
                print_dense_matrix_matlab("F_after copy", *F);
            }

            //Cleanup G and restart next
            if(config.debug) printf("Cleanup successful, next group\n");
        }

        const auto norm_t0 = std::chrono::high_resolution_clock::now();
        //Calculate norms per iteration
        double diag_sq = diag_norm_sq_tensor(*T, n, order);
        double off_sq  = offdiag_norm_sq_tensor(*T, n, order);
        double frob_sq = diag_sq + off_sq;
        double ratio   = off_sq / fmax(diag_sq, 1e-300);
        double rel_off = sqrt(fmax(off_sq, 0.0)) / fmax(sqrt(fmax(frob_sq, 1e-300)), 1e-300);
        double trace   = tensor_trace_general(*T, n, order);
        norm_block_seconds += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - norm_t0).count();
        //Check stopping criteria based on norms
        double delta = (iter == 0) ? std::numeric_limits<double>::infinity() : fabs(prev_rel_off - rel_off);
        bool stop = false;
        if (config.enable_stopping && (iter + 1) >= config.min_iterations) {stop = should_stop(delta, rel_off, config);}
        // Exact fixed point: no rotation was applied anywhere in this sweep, so the next sweep would see the
        // identical tensor and change nothing. Rounding-proof and scale-free; independent of min_iterations.
        bool fixed_point = false;
        if (config.enable_stopping && sweep_rotations == 0) { stop = true; fixed_point = true; }
        if (config.enable_stopping && config.tol_off > 0.0 && rel_off < config.tol_off) stop = true;
        //Iteration details
        printf("SWEEP iter=%d diag_norm_sq=%.15e offdiag_norm_sq=%.15e " "ratio=%.15e rel_offdiag=%.15e delta=%.15e trace=%.15e stop=%d\n", iter + 1, diag_sq, off_sq, ratio, rel_off, delta, trace, (int)stop);
        //Update parameters for next iteration
        prev_rel_off = rel_off;
        performed_iters = iter + 1;
        //Stop iterations
        if (stop) { converged = true;
        printf("Converged at iter %d: delta=%.15e rel_off=%.15e%s\n", iter + 1, delta, rel_off,
               fixed_point ? " (fixed point: no rotation applied in this sweep)" : "");
        break; //End loop
        }
    }
    printf("norm_block_seconds = %.10f\n", norm_block_seconds);
    printf("Jacobi finished after %d iteration(s), converged=%d\n", performed_iters, (int)converged);
}

void cleanup_jacobi(FLA_Obj* T, FLA_Obj* F){
    cleanup_tensor(T);
    cleanup_matrix(F);
}
