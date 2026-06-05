#include "reporting.hpp"
#include "tensor_utils.hpp"
#include "checking.hpp"
#include <cstdio>

void print_dense_matrix_matlab(const char* label, FLA_Obj A){
    dim_t m  = FLA_Obj_length(A);
    dim_t n  = FLA_Obj_width(A);
    double* buf = (double*) FLA_Obj_buffer_at_view(A);
    dim_t rs = FLA_Obj_row_stride(A);
    dim_t cs = FLA_Obj_col_stride(A);

    printf("%s = reshape([", label);
    for (dim_t j = 0; j < n; ++j) {
        for (dim_t i = 0; i < m; ++i) {
            printf("%.6f ", buf[i * rs + j * cs]);
        }
    }
    printf("],[%ld %ld ]);\n\n", (long)m, (long)n);
}

// Dump all canonical elements (i<=j<=k) via get_tensor_element_bccs
void dump_canonical_elements(const char* label, FLA_Obj T, dim_t n, dim_t order){
    printf("%%%% DUMP %s\n", label);
    for (dim_t i = 0; i < n; i++)
        for (dim_t j = i; j < n; j++)
            for (dim_t k = j; k < n; k++) {
                dim_t idx[3] = {i, j, k};
                double val = get_tensor_element_bccs(T, idx, order);
                printf("T_%s(%ld,%ld,%ld) = %.15e;\n", label, (long)(i+1), (long)(j+1), (long)(k+1), val);
            }
    printf("%%%% END DUMP %s\n\n", label);
}

//Initial state update
void print_initial_state(const JacobiConfig* config, FLA_Obj T, FLA_Obj F, JacobiPartition* vpartition, dim_t n, dim_t order){
    if (config->print_partition) {
        printf("%%%% Partition Info:\\n");
        printf("%%%% nr_groups = %d\\n", vpartition->nr_groups);
        printf("%%%% group_size = %d\\n", vpartition->group_size);
        for (int g = 0; g < vpartition->nr_groups; g++) {
            printf("%%%% Group %d: ", g + 1);
            for (int p = 0; p < vpartition->group_size; p++) {
                int idx = g * vpartition->group_size + p;
                printf("(%d,%d) ", vpartition->pairs[idx]->p, vpartition->pairs[idx]->q);
            }
            printf("\\n");
        }
        printf("\\n");
    }

    if (config->print_initial_tensor) {
        dump_canonical_elements("initial", T, n, order);
    }

    if (config->print_factor_matrix) {
        print_dense_matrix_matlab("InitialF", F);
    }
}

//Print final state
void print_final_state(const JacobiConfig* config, FLA_Obj T, FLA_Obj F, dim_t n, dim_t order){
    if (config->print_final_tensor) {
        dump_canonical_elements("final", T, n, order);
    }

    if (config->print_factor_matrix) {
        print_dense_matrix_matlab("FinalF", F);
    }

    if (config->print_matlab_summary) {
        double diag_norm, offdiag_norm;
        compute_tensor_norms(T, n, order, &diag_norm, &offdiag_norm);
        double ratio = (diag_norm > 0.0) ? offdiag_norm / diag_norm : 0.0;

        printf("final_diag_norm = %.15e;\\n", diag_norm);
        printf("final_offdiag_norm = %.15e;\\n", offdiag_norm);
        printf("final_ratio = %.15e;\\n", ratio);
    } 
}

void print_rotation_details(int iter, int group, int pair_idx, int p, int q, double c, double s, FLA_Obj T, dim_t order){
    printf("\n%%%% === Iteration %d, Group %d, Pair %d: (p=%d, q=%d) ===\n",
           iter + 1, group + 1, pair_idx + 1, p, q);
    printf("iter=%d; group=%d; pair=%d; p=%d; q=%d;\n", iter + 1, group + 1, pair_idx + 1, p, q);
    printf("c_%d_%d_%d = %.15e;\n", iter + 1, group + 1, pair_idx + 1, c);
    printf("s_%d_%d_%d = %.15e;\n", iter + 1, group + 1, pair_idx + 1, s);

    if (order != 3) {
        printf("print_rotation_details currently expects order=3\n");
        return;
    }

    dim_t p_dim = (dim_t)p;
    dim_t q_dim = (dim_t)q;

    dim_t idx_ppp[3] = {p_dim, p_dim, p_dim};
    dim_t idx_qqq[3] = {q_dim, q_dim, q_dim};
    dim_t idx_ppq[3] = {p_dim, p_dim, q_dim};
    dim_t idx_pqq[3] = {p_dim, q_dim, q_dim};

    double val_ppp = get_tensor_element_bccs(T, idx_ppp, order);
    double val_qqq = get_tensor_element_bccs(T, idx_qqq, order);
    double val_ppq = get_tensor_element_bccs(T, idx_ppq, order);
    double val_pqq = get_tensor_element_bccs(T, idx_pqq, order);

    printf("T_ppp_%d_%d_%d = %.15e; %% T(%d,%d,%d)\n",
           iter + 1, group + 1, pair_idx + 1, val_ppp, p + 1, p + 1, p + 1);
    printf("T_qqq_%d_%d_%d = %.15e; %% T(%d,%d,%d)\n",
           iter + 1, group + 1, pair_idx + 1, val_qqq, q + 1, q + 1, q + 1);
    printf("T_ppq_%d_%d_%d = %.15e; %% T(%d,%d,%d)\n",
           iter + 1, group + 1, pair_idx + 1, val_ppq, p + 1, p + 1, q + 1);
    printf("T_pqq_%d_%d_%d = %.15e; %% T(%d,%d,%d)\n",
           iter + 1, group + 1, pair_idx + 1, val_pqq, p + 1, q + 1, q + 1);
    printf("\n");
}