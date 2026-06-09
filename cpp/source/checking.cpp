#include "checking.hpp"
#include "reporting.hpp"
#include "tensor_utils.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

// Increment an order-dimensional index in row-major style: idx[0] fastest.
// Returns 0 when wrapped past the last element (done), 1 otherwise.
static int next_index(dim_t* idx, dim_t order, dim_t n){
    for (dim_t d = 0; d < order; ++d) {
        idx[d]++;
        if (idx[d] < n) return 1;
        idx[d] = 0;
    }
    return 0;
}

double compute_tensor_difference(FLA_Obj A, FLA_Obj B){
    double max_diff = 0.0;
    dim_t n_blocks = FLA_Obj_num_elem_alloc(A);
    FLA_Obj* A_buf = (FLA_Obj*)FLA_Obj_base_buffer(A);
    FLA_Obj* B_buf = (FLA_Obj*)FLA_Obj_base_buffer(B);

    for (dim_t i = 0; i < n_blocks; i++) {
        if (A_buf[i].isStored) {
            dim_t n_elem = FLA_Obj_num_elem_alloc(A_buf[i]);
            double* A_data = (double*)FLA_Obj_base_buffer(A_buf[i]);
            double* B_data = (double*)FLA_Obj_base_buffer(B_buf[i]);

            for (dim_t j = 0; j < n_elem; j++) {
                double diff = fabs(A_data[j] - B_data[j]);
                if (diff > max_diff) max_diff = diff;
            }
        }
    }

    return max_diff;
}

double orthogonality_error_matrix(FLA_Obj U, dim_t n){
    double sum_sq = 0.0;

    for (dim_t i = 0; i < n; ++i) {
        for (dim_t j = 0; j < n; ++j) {
            double dot = 0.0;
            for (dim_t k = 0; k < n; ++k) {
                dot += get_matrix_element(U, k, i) * get_matrix_element(U, k, j);
            }
            double target = (i == j) ? 1.0 : 0.0;
            double diff = dot - target;
            sum_sq += diff * diff;
        }
    }

    return sqrt(sum_sq);
}

template <typename F>
void for_all_indices(dim_t* idx, dim_t order, dim_t n, F&& f) {
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;
    while (true) {
        f(idx);
        if (!next_index(idx, order, n)) break;
    }
}

double diag_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order) {
    double sum_diag_sq = 0.0;
    dim_t idx[FLA_MAX_ORDER]; // or order if you prefer VLAs.
    for_all_indices(idx, order, n, [&](dim_t* idx){
        if (is_superdiagonal(idx, order)) {
            double val = get_tensor_element_bccs_alt(T, idx, order);
            sum_diag_sq += val * val;
        }
    });
    return sum_diag_sq;
}

double frob_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order){
    double sum_sq = 0.0;
    dim_t idx[order];
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;

    while (1) {
        double val = get_tensor_element_bccs_alt(T, idx, order);
        sum_sq += val * val;
        if (!next_index(idx, order, n)) break;
    }
    return sum_sq;
}

double offdiag_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order){
    return frob_norm_sq_tensor(T, n, order) - diag_norm_sq_tensor(T, n, order);
}

double tensor_trace_general(FLA_Obj T, dim_t n, dim_t order) {
    double tr = 0.0;
    dim_t idx[order];

    for (dim_t i = 0; i < n; ++i) {
        for (dim_t d = 0; d < order; ++d) idx[d] = i;
        tr += get_tensor_element_bccs_alt(T, idx, order);
    }

    return tr;
}

void compute_tensor_norms(FLA_Obj T, dim_t n, dim_t order, double* diag_norm, double* offdiag_norm){
    double diag_sq = diag_norm_sq_tensor(T, n, order);
    double off_sq = offdiag_norm_sq_tensor(T, n, order);
    *diag_norm = sqrt(diag_sq);
    *offdiag_norm = sqrt(off_sq);
}

void check_diagonalization(FLA_Obj T, dim_t n, dim_t order, double tolerance){
    double sum_offdiag_abs = 0.0;
    double sum_diag_abs = 0.0;
    double max_offdiag_abs = 0.0;
    unsigned long long num_offdiag = 0;
    unsigned long long num_diag = 0;

    dim_t idx[order];
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;

    while (1) {
        double val = get_tensor_element_bccs_alt(T, idx, order);
        double abs_val = fabs(val);

        if (is_superdiagonal(idx, order)) {
            sum_diag_abs += abs_val;
            num_diag++;
        } else {
            sum_offdiag_abs += abs_val;
            num_offdiag++;
            if (abs_val > max_offdiag_abs) max_offdiag_abs = abs_val;
        }

        if (!next_index(idx, order, n)) break;
    }

    double avg_offdiag = (num_offdiag > 0) ? sum_offdiag_abs / (double)num_offdiag : 0.0;
    double avg_diag = (num_diag > 0) ? sum_diag_abs / (double)num_diag : 0.0;
    double offdiag_ratio_abs = (avg_diag > 0.0) ? avg_offdiag / avg_diag : 0.0;
    double diag_norm_sq = diag_norm_sq_tensor(T, n, order);
    double offdiag_norm_sq = offdiag_norm_sq_tensor(T, n, order);
    double frob_norm_sq = diag_norm_sq + offdiag_norm_sq;
    double ratio = offdiag_norm_sq / fmax(diag_norm_sq, 1e-300);
    double rel_offdiag = sqrt(fmax(offdiag_norm_sq, 0.0)) / fmax(sqrt(fmax(frob_norm_sq, 1e-300)), 1e-300);
    double tr = tensor_trace_general(T, n, order);

    printf("\n=== Checking Diagonalization ===\n");
    printf("diag_norm_sq = %.15e\n", diag_norm_sq);
    printf("offdiag_norm_sq = %.15e\n", offdiag_norm_sq);
    printf("frob_norm_sq = %.15e\n", frob_norm_sq);
    printf("ratio = %.15e\n", ratio);
    printf("rel_offdiag = %.15e\n", rel_offdiag);
    printf("trace = %.15e\n", tr);
    printf("avg_diag_abs = %.15e\n", avg_diag);
    printf("avg_offdiag_abs = %.15e\n", avg_offdiag);
    printf("avg_offdiag_over_avg_diag = %.15e\n", offdiag_ratio_abs);
    printf("max_offdiag_abs = %.15e\n", max_offdiag_abs);

    if (max_offdiag_abs < tolerance)
        printf("diagonalization_check = PASS (max_offdiag_abs < %.2e)\n", tolerance);
    else
        printf("diagonalization_check = FAIL (max_offdiag_abs >= %.2e)\n", tolerance);

    printf("\nSample diagonal elements:\n");
    dim_t sample_size = (n < 5) ? n : 5;
    for (dim_t i = 0; i < sample_size; ++i) {
        for (dim_t d = 0; d < order; ++d) idx[d] = i;
        double v = get_tensor_element_bccs_alt(T, idx, order);
        printf(" T[%ld", (long)i);
        for (dim_t d = 1; d < order; ++d) printf(",%ld", (long)i);
        printf("] = %.15e\n", v);
    }

    printf("================================\n\n");
}

double max_abs_tensor(FLA_Obj T, dim_t n, dim_t order){
    double max_abs = 0.0;
    dim_t idx[order];
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;

    while (1) {
        double val = fabs(get_tensor_element_bccs_alt(T, idx, order));
        if (val > max_abs) max_abs = val;
        if (!next_index(idx, order, n)) break;
    }

    return max_abs;
}

void extract_diagonal_tensor(FLA_Obj T, FLA_Obj D, dim_t n, dim_t order){
    FLA_Set_zero_tensor(D);

    dim_t idx[order];
    for (dim_t i = 0; i < n; ++i) {
        for (dim_t d = 0; d < order; ++d) idx[d] = i;
        double val = get_tensor_element_bccs_alt(T, idx, order);
        set_tensor_element_bccs(D, idx, order, val);
    }
}

void tensor_diff_metrics( FLA_Obj A, FLA_Obj B, dim_t n, dim_t order, double* err_abs, double* err_rel, double* max_abs, double* max_rel){
    double sum_sq = 0.0;
    double max_diff = 0.0;
    double max_ref = 0.0;

    dim_t idx[order];
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;

    while (1) {
        double a = get_tensor_element_bccs_alt(A, idx, order);
        double b = get_tensor_element_bccs_alt(B, idx, order);
        double diff = fabs(a - b);

        sum_sq += diff * diff;
        if (diff > max_diff) max_diff = diff;

        double abs_ref = fabs(b);
        if (abs_ref > max_ref) max_ref = abs_ref;

        if (!next_index(idx, order, n)) break;
    }

    *err_abs = sqrt(sum_sq);
    *err_rel = (*err_abs) / fmax(sqrt(frob_norm_sq_tensor(B, n, order)), 1e-300);
    *max_abs = max_diff;
    *max_rel = max_diff / fmax(max_ref, 1e-300);
}

void reconstruct_m( FLA_Obj Ddiag, FLA_Obj Udense, FLA_Obj Trec, dim_t n){
    FLA_Set_zero_tensor(Trec);

    dim_t idx[3], didx[3];
    for (dim_t i = 0; i < n; ++i) {
        for (dim_t j = i; j < n; ++j) {
            for (dim_t k = j; k < n; ++k) {
                double s = 0.0;
                for (dim_t a = 0; a < n; ++a) {
                    didx[0] = a; didx[1] = a; didx[2] = a;
                    double d  = get_tensor_element_bccs_alt(Ddiag, didx, 3);
                    double ui = get_dense_matrix_element(Udense, i, a);
                    double uj = get_dense_matrix_element(Udense, j, a);
                    double uk = get_dense_matrix_element(Udense, k, a);
                    s += d * ui * uj * uk;
                }
                idx[0] = i; idx[1] = j; idx[2] = k;
                set_tensor_element_bccs(Trec, idx, 3, s);
            }
        }
    }
    //dump_canonical_elements("reconstructed", Trec, n, 3);
}

void check_diagonalization_with_reconstruction(FLA_Obj T_final, FLA_Obj U_final, FLA_Obj T_initial, FLA_Obj D_diag, FLA_Obj T_reconstructed, dim_t n, dim_t order, double tolerance) {
    if (order != 3) {
        printf("check_diagonalization_with_reconstruction currently expects order=3\n");
        return;
    }

    DiagonalizationReport r;
    double sum_offdiag_abs = 0.0;
    double sum_diag_abs = 0.0;
    double max_offdiag_abs = 0.0;
    unsigned long long num_offdiag = 0;
    unsigned long long num_diag = 0;

    dim_t idx[order];
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;

    while (1) {
        double val = get_tensor_element_bccs_alt(T_final, idx, order);
        double abs_val = fabs(val);

        if (is_superdiagonal(idx, order)) {
            sum_diag_abs += abs_val;
            num_diag++;
        } else {
            sum_offdiag_abs += abs_val;
            num_offdiag++;
            if (abs_val > max_offdiag_abs) max_offdiag_abs = abs_val;
        }

        if (!next_index(idx, order, n)) break;
    }

    r.diag_norm_sq = diag_norm_sq_tensor(T_final, n, order);
    r.offdiag_norm_sq = offdiag_norm_sq_tensor(T_final, n, order);
    r.frob_norm_sq = r.diag_norm_sq + r.offdiag_norm_sq;
    r.ratio = r.offdiag_norm_sq / fmax(r.diag_norm_sq, 1e-300);
    r.rel_offdiag = sqrt(fmax(r.offdiag_norm_sq, 0.0)) / fmax(sqrt(fmax(r.frob_norm_sq, 1e-300)), 1e-300);
    r.trace = tensor_trace_general(T_final, n, order);
    r.max_offdiag_abs = max_offdiag_abs;
    r.avg_diag_abs = (num_diag > 0) ? sum_diag_abs / (double)num_diag : 0.0;
    r.avg_offdiag_abs = (num_offdiag > 0) ? sum_offdiag_abs / (double)num_offdiag : 0.0;
    r.orthogonality_error = orthogonality_error_matrix(U_final, n);

    extract_diagonal_tensor(T_final, D_diag, n, order);
    reconstruct_m(D_diag, U_final, T_reconstructed, n);

    tensor_diff_metrics(
        T_reconstructed,
        T_initial,
        n,
        order,
        &r.reconstruction_error_abs,
        &r.reconstruction_error_rel,
        &r.reconstruction_max_abs,
        &r.reconstruction_max_rel
    );

    printf("\n=== Final diagonalization + reconstruction report ===\n");
    printf("diag_norm_sq = %.15e\n", r.diag_norm_sq);
    printf("offdiag_norm_sq = %.15e\n", r.offdiag_norm_sq);
    printf("frob_norm_sq = %.15e\n", r.frob_norm_sq);
    printf("ratio = %.15e\n", r.ratio);
    printf("rel_offdiag = %.15e\n", r.rel_offdiag);
    printf("trace = %.15e\n", r.trace);
    printf("avg_diag_abs = %.15e\n", r.avg_diag_abs);
    printf("avg_offdiag_abs = %.15e\n", r.avg_offdiag_abs);
    printf("max_offdiag_abs = %.15e\n", r.max_offdiag_abs);
    printf("orthogonality_error_U_final = %.15e\n", r.orthogonality_error);
    printf("reconstruction_error_abs = %.15e\n", r.reconstruction_error_abs);
    printf("reconstruction_error_rel = %.15e\n", r.reconstruction_error_rel);
    printf("reconstruction_max_abs = %.15e\n", r.reconstruction_max_abs);
    printf("reconstruction_max_rel = %.15e\n", r.reconstruction_max_rel);

    if (r.max_offdiag_abs < tolerance)
        printf("diagonalization_check = PASS (max_offdiag_abs < %.2e)\n", tolerance);
    else
        printf("diagonalization_check = FAIL (max_offdiag_abs >= %.2e)\n", tolerance);

    printf("================================\n\n");
}