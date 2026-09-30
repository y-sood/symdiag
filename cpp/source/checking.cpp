#include "checking.hpp"
#include "reporting.hpp"
#include "tensor_utils.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

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
                dot += get_dense_matrix_element(U, k, i) * get_dense_matrix_element(U, k, j);
            }
            double target = (i == j) ? 1.0 : 0.0;
            double diff = dot - target;
            sum_sq += diff * diff;
        }
    }

    return sqrt(sum_sq);
}

// Calls f(idx, weight) once per canonical (non-decreasing) index tuple. weight is the number
// of distinct permutations of idx = order! / prod(run length!), so sum_canonical weight * g(idx)
// equals the sum of g over all n^order entries whenever g is permutation-symmetric (as for a
// symmetric tensor). Visits C(n+order-1, order) tuples instead of n^order.
template <typename F>
static void for_each_canonical(dim_t n, dim_t order, F&& f) {   // f(dim_t* idx, double weight)
    double fact_order = 1.0;
    for (dim_t m = 2; m <= order; ++m) fact_order *= (double)m;

    dim_t idx[FLA_MAX_ORDER];
    for (dim_t m = 0; m < order; ++m) idx[m] = 0;
    while (true) {
        double w = fact_order;
        for (dim_t m = 0; m < order; ) {
            dim_t run = 1;
            while (m + run < order && idx[m + run] == idx[m]) ++run;
            for (dim_t k = 2; k <= run; ++k) w /= (double)k;
            m += run;
        }
        f(idx, w);

        dim_t pos = order;
        while (pos > 0 && idx[pos - 1] == n - 1) pos--;
        if (pos == 0) break;
        idx[pos - 1]++;
        for (dim_t m = pos; m < order; ++m) idx[m] = idx[pos - 1];
    }
}

// Superdiagonal only: n entries, no need to enumerate the rest.
double diag_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order) {
    double sum_diag_sq = 0.0;
    dim_t idx[FLA_MAX_ORDER];
    for (dim_t i = 0; i < n; ++i) {
        for (dim_t d = 0; d < order; ++d) idx[d] = i;
        double val = get_tensor_element_bccs_alt(T, idx, order);
        sum_diag_sq += val * val;
    }
    return sum_diag_sq;
}

double frob_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order){
    double sum_sq = 0.0;
    for_each_canonical(n, order, [&](dim_t* idx, double w){
        double val = get_tensor_element_bccs_alt(T, idx, order);
        sum_sq += w * val * val;
    });
    return sum_sq;
}

double offdiag_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order){
    // Not frob_norm_sq_tensor(T,...) - diag_norm_sq_tensor(T,...): once the
    // off-diagonal mass is small relative to the diagonal, that subtraction
    // of two nearly-equal O(1) quantities catastrophically cancels to
    // exactly 0.0 - even while real off-diagonal entries remain (e.g.
    // max_offdiag_abs ~1e-9). Since this feeds `ratio` in the stopping
    // check, a spurious exact 0.0 satisfies any tol_ratio and stops the
    // sweep early. Sum off-diagonal squares directly instead.
    double sum_offdiag_sq = 0.0;
    for_each_canonical(n, order, [&](dim_t* idx, double w){
        if (!is_superdiagonal(idx, order)) {
            double val = get_tensor_element_bccs_alt(T, idx, order);
            sum_offdiag_sq += w * val * val;
        }
    });
    return sum_offdiag_sq;
}

double tensor_trace_general(FLA_Obj T, dim_t n, dim_t order) {
    double tr = 0.0;
    dim_t idx[FLA_MAX_ORDER];

    for (dim_t i = 0; i < n; ++i) {
        for (dim_t d = 0; d < order; ++d) idx[d] = i;
        tr += get_tensor_element_bccs_alt(T, idx, order);
    }

    return tr;
}

struct EntryStats {
    double sum_diag_abs = 0.0, sum_offdiag_abs = 0.0, max_offdiag_abs = 0.0;
    unsigned long long num_diag = 0, num_offdiag = 0;
};

// Superdiagonal / off-diagonal |entry| statistics over ALL n^order entries, computed from
// canonical entries only (each weighted by its permutation count).
static EntryStats entry_stats(FLA_Obj T, dim_t n, dim_t order) {
    EntryStats st;
    double total = 1.0;
    for (dim_t m = 0; m < order; ++m) total *= (double)n;
    st.num_diag = (unsigned long long)n;
    st.num_offdiag = (unsigned long long)total - (unsigned long long)n;
    for_each_canonical(n, order, [&](dim_t* idx, double w){
        double abs_val = fabs(get_tensor_element_bccs_alt(T, idx, order));
        if (is_superdiagonal(idx, order)) {
            st.sum_diag_abs += w * abs_val;
        } else {
            st.sum_offdiag_abs += w * abs_val;
            if (abs_val > st.max_offdiag_abs) st.max_offdiag_abs = abs_val;
        }
    });
    return st;
}

void compute_tensor_norms(FLA_Obj T, dim_t n, dim_t order, double* diag_norm, double* offdiag_norm){
    double diag_sq = diag_norm_sq_tensor(T, n, order);
    double off_sq = offdiag_norm_sq_tensor(T, n, order);
    *diag_norm = sqrt(diag_sq);
    *offdiag_norm = sqrt(off_sq);
}

void check_diagonalization(FLA_Obj T, dim_t n, dim_t order, double tolerance){
    EntryStats st = entry_stats(T, n, order);
    double sum_offdiag_abs = st.sum_offdiag_abs, sum_diag_abs = st.sum_diag_abs;
    double max_offdiag_abs = st.max_offdiag_abs;
    unsigned long long num_offdiag = st.num_offdiag, num_diag = st.num_diag;
    dim_t idx[FLA_MAX_ORDER];

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

    if (max_offdiag_abs < tolerance * sqrt(frob_norm_sq))
        printf("diagonalization_check = PASS (max_offdiag_abs < %.2e * ||T||_F)\n", tolerance);
    else
        printf("diagonalization_check = FAIL (max_offdiag_abs >= %.2e * ||T||_F)\n", tolerance);

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
    for_each_canonical(n, order, [&](dim_t* idx, double){
        double val = fabs(get_tensor_element_bccs_alt(T, idx, order));
        if (val > max_abs) max_abs = val;
    });
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

    for_each_canonical(n, order, [&](dim_t* idx, double w){
        double a = get_tensor_element_bccs_alt(A, idx, order);
        double b = get_tensor_element_bccs_alt(B, idx, order);
        double diff = fabs(a - b);

        sum_sq += w * diff * diff;
        if (diff > max_diff) max_diff = diff;

        double abs_ref = fabs(b);
        if (abs_ref > max_ref) max_ref = abs_ref;
    });

    *err_abs = sqrt(sum_sq);
    *err_rel = (*err_abs) / fmax(sqrt(frob_norm_sq_tensor(B, n, order)), 1e-300);
    *max_abs = max_diff;
    *max_rel = max_diff / fmax(max_ref, 1e-300);
}

void reconstruct_m( FLA_Obj Ddiag, FLA_Obj Udense, FLA_Obj Trec, dim_t n, dim_t order){
    FLA_Set_zero_tensor(Trec);

    // One entry per canonical (non-decreasing) index tuple; set_tensor_element_bccs
    // canonicalizes anyway. For order 3 this is the old i<=j<=k triple loop.
    dim_t idx[FLA_MAX_ORDER], didx[FLA_MAX_ORDER];
    for (dim_t m = 0; m < order; ++m) idx[m] = 0;
    while (true) {
        double s = 0.0;
        for (dim_t a = 0; a < n; ++a) {
            for (dim_t m = 0; m < order; ++m) didx[m] = a;
            double term = get_tensor_element_bccs_alt(Ddiag, didx, order);
            for (dim_t m = 0; m < order; ++m) term *= get_dense_matrix_element(Udense, idx[m], a);
            s += term;
        }
        set_tensor_element_bccs(Trec, idx, order, s);

        dim_t pos = order;
        while (pos > 0 && idx[pos - 1] == n - 1) pos--;
        if (pos == 0) break;
        idx[pos - 1]++;
        for (dim_t m = pos; m < order; ++m) idx[m] = idx[pos - 1];
    }
}

void check_diagonalization_with_reconstruction(FLA_Obj T_final, FLA_Obj U_final, FLA_Obj T_initial, dim_t n, dim_t order, double tolerance) {
    DiagonalizationReport r;
    EntryStats st = entry_stats(T_final, n, order);
    double sum_offdiag_abs = st.sum_offdiag_abs, sum_diag_abs = st.sum_diag_abs;
    double max_offdiag_abs = st.max_offdiag_abs;
    unsigned long long num_offdiag = st.num_offdiag, num_diag = st.num_diag;

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

    // Reconstruction error over canonical entries (non-decreasing index tuples), each
    // weighted by its number of distinct permutations = order! / prod(run length!).
    std::vector<double> dvec(n), Ud(n * n);
    {
        dim_t didx[FLA_MAX_ORDER];
        for (dim_t a = 0; a < n; ++a) {
            for (dim_t m = 0; m < order; ++m) didx[m] = a;
            dvec[a] = get_tensor_element_bccs_alt(T_final, didx, order);
            for (dim_t i = 0; i < n; ++i) Ud[i * n + a] = get_dense_matrix_element(U_final, i, a);
        }
    }
    double sum_sq = 0.0, init_sq = 0.0, max_diff = 0.0, max_ref = 0.0;
    for_each_canonical(n, order, [&](dim_t* cidx, double w){
        double rec = 0.0;
        for (dim_t a = 0; a < n; ++a) {
            double term = dvec[a];
            for (dim_t m = 0; m < order; ++m) term *= Ud[cidx[m] * n + a];
            rec += term;
        }
        double ref = get_tensor_element_bccs_alt(T_initial, cidx, order);
        double diff = fabs(rec - ref);
        sum_sq += w * diff * diff;
        init_sq += w * ref * ref;
        if (diff > max_diff) max_diff = diff;
        if (fabs(ref) > max_ref) max_ref = fabs(ref);
    });
    r.reconstruction_error_abs = sqrt(sum_sq);
    r.reconstruction_error_rel = r.reconstruction_error_abs / fmax(sqrt(init_sq), 1e-300);
    r.reconstruction_max_abs = max_diff;
    r.reconstruction_max_rel = max_diff / fmax(max_ref, 1e-300);

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

    // tolerance is relative to ||T||_F (scale-free), so the verdict does not depend on the input scale
    const double abs_tol = tolerance * sqrt(r.frob_norm_sq);
    if (r.max_offdiag_abs < abs_tol)
        printf("diagonalization_check = PASS (max_offdiag_abs < %.2e * ||T||_F)\n", tolerance);
    else
        printf("diagonalization_check = FAIL (max_offdiag_abs >= %.2e * ||T||_F)\n", tolerance);

    printf("================================\n\n");
}

bool full_reconstruction_report(FLA_Obj T_final, FLA_Obj F, FLA_Obj T_initial, dim_t n, dim_t order){
    double total = 1.0;
    for (dim_t m = 0; m < order; ++m) total *= (double)n;
    if (total * 16.0 > 2.0e9) {
        printf("full_reconstruction: skipped (n^order scratch = %.2f GB)\n", total * 16.0 / 1e9);
        return false;
    }
    const size_t N = (size_t)total;
    std::vector<double> cur(N), nxt(N), Ud(n * n);
    for (dim_t i = 0; i < n; ++i)
        for (dim_t a = 0; a < n; ++a) Ud[i * n + a] = get_dense_matrix_element(F, i, a);

    // Expand T_final to a dense row-major array (index[0] slowest).
    dim_t idx[FLA_MAX_ORDER];
    for (dim_t m = 0; m < order; ++m) idx[m] = 0;
    for (size_t lin = 0; lin < N; ++lin) {
        cur[lin] = get_tensor_element_bccs_alt(T_final, idx, order);
        dim_t m = order;
        while (m > 0 && ++idx[m - 1] == n) idx[--m] = 0;
    }

    // Mode products with F: new[..i..] = sum_a F[i,a] * old[..a..]
    for (dim_t m = 0; m < order; ++m) {
        size_t outer = 1, inner = 1;
        for (dim_t k = 0; k < m; ++k) outer *= (size_t)n;
        for (dim_t k = m + 1; k < order; ++k) inner *= (size_t)n;
        for (size_t o = 0; o < outer; ++o)
            for (dim_t i = 0; i < n; ++i)
                for (size_t in = 0; in < inner; ++in) {
                    double acc = 0.0;
                    for (dim_t a = 0; a < n; ++a)
                        acc += Ud[i * n + a] * cur[(o * n + a) * inner + in];
                    nxt[(o * n + i) * inner + in] = acc;
                }
        cur.swap(nxt);
    }

    double sum_sq = 0.0, ref_sq = 0.0, max_diff = 0.0, max_ref = 0.0;
    for (dim_t m = 0; m < order; ++m) idx[m] = 0;
    for (size_t lin = 0; lin < N; ++lin) {
        double ref = get_tensor_element_bccs_alt(T_initial, idx, order);
        double diff = fabs(cur[lin] - ref);
        sum_sq += diff * diff;
        ref_sq += ref * ref;
        if (diff > max_diff) max_diff = diff;
        if (fabs(ref) > max_ref) max_ref = fabs(ref);
        dim_t m = order;
        while (m > 0 && ++idx[m - 1] == n) idx[--m] = 0;
    }
    printf("full_reconstruction_error_abs = %.15e\n", sqrt(sum_sq));
    printf("full_reconstruction_error_rel = %.15e\n", sqrt(sum_sq) / fmax(sqrt(ref_sq), 1e-300));
    printf("full_reconstruction_max_abs = %.15e\n", max_diff);
    printf("full_reconstruction_max_rel = %.15e\n", max_diff / fmax(max_ref, 1e-300));
    return true;
}
