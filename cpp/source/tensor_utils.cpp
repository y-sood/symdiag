#include "tensor_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

void set_dense_matrix_element(FLA_Obj A, dim_t row, dim_t col, double value){
    double* buf = (double*) FLA_Obj_buffer_at_view(A);
    dim_t rs = FLA_Obj_row_stride(A);
    dim_t cs = FLA_Obj_col_stride(A);
    buf[row * rs + col * cs] = value;
}

double get_dense_matrix_element(FLA_Obj A, dim_t i, dim_t j){
    double* buf = (double*) FLA_Obj_buffer_at_view(A);
    dim_t rs = FLA_Obj_row_stride(A);
    dim_t cs = FLA_Obj_col_stride(A);
    return buf[i * rs + j * cs];
}

void set_tensor_element_bccs(FLA_Obj T, dim_t* index, dim_t order, double value){
    // Sort to canonical form for symmetric tensor. order is always small
    // (<= FLA_MAX_ORDER) here, so these are fixed-size stack arrays rather
    // than per-call mallocs.
    dim_t canonical_index[FLA_MAX_ORDER];
    memcpy(canonical_index, index, order * sizeof(dim_t));
    std::sort(canonical_index, canonical_index + order);

    // Check if tensor is blocked
    if (FLA_Obj_elemtype(T) == FLA_TENSOR) {
        // BCCS: Navigate to correct block first
        FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);

        // Get block size
        dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
        dim_t n_blocks_per_mode = FLA_Obj_dimsize(T, 0);

        // Calculate local index within block and linear block index together
        // (row-major), same computation as before but without the temporary
        // block_index array.
        dim_t local_index[FLA_MAX_ORDER];
        dim_t linear_block_idx = 0;
        dim_t block_stride = 1;
        for (dim_t i = order; i > 0; i--) {
            dim_t d = i - 1;
            local_index[d] = canonical_index[d] % block_size;
            linear_block_idx += (canonical_index[d] / block_size) * block_stride;
            block_stride *= n_blocks_per_mode;
        }

        // Direct pointer arithmetic instead of FLA_Obj_tensor_buffer_at_view,
        // which mallocs+frees a stride array and an offset array on every
        // call (same fix as apply_group_rotation_cellwise's elem_ptr, order-
        // general here since this function serves any order).
        //
        // FLA_Obj_tensor_buffer_at_view sums over the OBJECT's own .order
        // field, not the order passed in here - for a plain (non-tensor)
        // FLA_Obj like the factor matrix F, .order is 0, so the original
        // always read buffer[0] regardless of index. Match that exactly:
        // loop to block.order, and for any position beyond what we wrote
        // into local_index (i.e. i >= order), use the block's own
        // pre-existing offset[i], since the original only overwrote
        // offset[0..order-1] via memcpy and left the rest as inherited.
        FLA_Obj block = blocks[linear_block_idx];
        dim_t buf_order = block.order;
        FLA_Base_obj* base = block.base;
        double* buf = (double*)base->buffer;
        dim_t offset = 0;
        for (dim_t i = 0; i < buf_order; i++) {
            dim_t li = (i < order) ? local_index[i] : block.offset[i];
            offset += li * base->stride[i];
        }
        buf[offset] = value;

    } else {
        // Scalar tensor (non-blocked). Same .order-vs-order-parameter
        // subtlety as above.
        dim_t buf_order = T.order;
        FLA_Base_obj* base = T.base;
        double* buf = (double*)base->buffer;
        dim_t offset = 0;
        for (dim_t i = 0; i < buf_order; i++) {
            dim_t idx_val = (i < order) ? canonical_index[i] : T.offset[i];
            offset += idx_val * base->stride[i];
        }
        buf[offset] = value;
    }
}

void set_matrix_element_bccs(FLA_Obj G, dim_t row, dim_t col, double value){
    FLA_Obj* blocks = (FLA_Obj*) FLA_Obj_base_buffer(G);
    dim_t blocksize = FLA_Obj_dimsize(blocks[0], 0);

    dim_t blockrow = row / blocksize;
    dim_t blockcol = col / blocksize;
    dim_t localrow = row % blocksize;
    dim_t localcol = col % blocksize;

    dim_t* outerstride = FLA_Obj_stride(G);
    dim_t linearblockidx = blockrow * outerstride[0] + blockcol * outerstride[1];

    FLA_Obj block = blocks[linearblockidx];
    FLA_Obj blockview = block;
    blockview.offset[0] = localrow;
    blockview.offset[1] = localcol;

    double* buffer = (double*) FLA_Obj_tensor_buffer_at_view(blockview);
    *buffer = value;
}

static inline void sort3(dim_t& a, dim_t& b, dim_t& c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
}

double get_tensor_element_bccs(FLA_Obj T, dim_t i, dim_t j, dim_t k) {
    sort3(i, j, k);

    FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);
    dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
    dim_t n_blocks_per_mode = FLA_Obj_dimsize(T, 0);

    dim_t bi = i / block_size, bj = j / block_size, bk = k / block_size;
    dim_t li = i % block_size, lj = j % block_size, lk = k % block_size;

    dim_t linear_block_idx = (bi * n_blocks_per_mode + bj) * n_blocks_per_mode + bk;

    FLA_Obj block_view = blocks[linear_block_idx];
    block_view.offset[0] = li;
    block_view.offset[1] = lj;
    block_view.offset[2] = lk;

    return *(double*)FLA_Obj_tensor_buffer_at_view(block_view);
}

double get_tensor_element_bccs_alt(FLA_Obj T, dim_t* index, dim_t order){
    // Sort to canonical form for symmetric tensor. order is always small
    // (<= FLA_MAX_ORDER) here, so a fixed-size stack array avoids a
    // malloc/free on every call (this used to be the dominant cost here -
    // ~5 mallocs/frees per call between this function's own three temporary
    // arrays and FLA_Obj_tensor_buffer_at_view's two internal ones, times
    // up to O(n^4) calls in the checking/reconstruction code - see
    // OPTIMIZATION_LOG.md).
    dim_t canonical_index[FLA_MAX_ORDER];
    memcpy(canonical_index, index, order * sizeof(dim_t));
    std::sort(canonical_index, canonical_index + order);

    //CHECK IF TENSOR IS BLOCKED
    if (FLA_Obj_elemtype(T) == FLA_TENSOR) {
        //BCCS: Navigate to correct block first
        FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);

        //Get block size
        dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
        dim_t n_blocks_per_mode = FLA_Obj_dimsize(T, 0);

        // Local index within block and linear block index (row-major),
        // computed together without a separate block_index array.
        dim_t local_index[FLA_MAX_ORDER];
        dim_t linear_block_idx = 0;
        dim_t block_stride = 1;
        for (dim_t i = order; i > 0; i--) {
            dim_t d = i - 1;
            local_index[d] = canonical_index[d] % block_size;
            linear_block_idx += (canonical_index[d] / block_size) * block_stride;
            block_stride *= n_blocks_per_mode;
        }

        // Direct pointer arithmetic instead of FLA_Obj_tensor_buffer_at_view.
        //
        // FLA_Obj_tensor_buffer_at_view sums over the OBJECT's own .order
        // field, not the order passed in here - for a plain (non-tensor)
        // FLA_Obj like the factor matrix F, .order is 0, so the original
        // always read buffer[0] regardless of index. Match that exactly:
        // loop to block.order, and for any position beyond what we wrote
        // into local_index (i.e. i >= order), use the block's own
        // pre-existing offset[i], since the original only overwrote
        // offset[0..order-1] via memcpy and left the rest as inherited.
        FLA_Obj block = blocks[linear_block_idx];
        dim_t buf_order = block.order;
        FLA_Base_obj* base = block.base;
        double* buf = (double*)base->buffer;
        dim_t offset = 0;
        for (dim_t i = 0; i < buf_order; i++) {
            dim_t li = (i < order) ? local_index[i] : block.offset[i];
            offset += li * base->stride[i];
        }
        return buf[offset];

    } else {
        //Scalar tensor (non-blocked). Same .order-vs-order-parameter
        // subtlety as above.
        dim_t buf_order = T.order;
        FLA_Base_obj* base = T.base;
        double* buf = (double*)base->buffer;
        dim_t offset = 0;
        for (dim_t i = 0; i < buf_order; i++) {
            dim_t idx_val = (i < order) ? canonical_index[i] : T.offset[i];
            offset += idx_val * base->stride[i];
        }
        return buf[offset];
    }
}

double get_matrix_element(FLA_Obj A, dim_t i, dim_t j){
    dim_t idx[2] = {i, j};
    return get_tensor_element_bccs_alt(A, idx, 2);
}

void set_matrix_element(FLA_Obj A, dim_t i, dim_t j, double value){
    dim_t idx[2] = {i, j};
    set_tensor_element_bccs(A, idx, 2, value);
}

// Return 1 if idx[0]==idx[1]==...==idx[order-1], else 0.
int is_superdiagonal(const dim_t* idx, dim_t order){
    for (dim_t d = 1; d < order; ++d)
        if (idx[d] != idx[0]) return 0;
    return 1;
}

//Reorthogonalise columns
void gram_schmidt(const std::vector<double>& M, std::vector<double>& U, int n){
    U = M;
    auto at = [n](const std::vector<double>& A, int i, int j) -> double {
        return A[i * n + j];
    };
    auto ref = [n](std::vector<double>& A, int i, int j) -> double& {
        return A[i * n + j];
    };

    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < n; ++i) dot += at(U, i, k) * at(M, i, j);
            for (int i = 0; i < n; ++i) ref(U, i, j) -= dot * at(U, i, k);
        }
        double norm = 0.0;
        for (int i = 0; i < n; ++i) norm += at(U, i, j) * at(U, i, j);
        norm = std::sqrt(norm);
        if (norm < 1e-15) {
            fprintf(stderr, "Gram-Schmidt produced a near-zero column\n");
            std::exit(EXIT_FAILURE);
        }
        for (int i = 0; i < n; ++i) ref(U, i, j) /= norm;
    }
} 

//Create a nxnxn diagonalizable tensor
void initDiagonalizableTensor(dim_t order, dim_t size[], dim_t b, FLA_Obj* obj, int n, unsigned int seed){
    dim_t i;
    dim_t blocked_stride[FLA_MAX_ORDER];
    dim_t block_size[FLA_MAX_ORDER];
    dim_t blocked_size[FLA_MAX_ORDER];
    TLA_sym sym;

    for (i = 0; i < order; i++) block_size[i] = b;
    FLA_array_elemwise_quotient(order, size, block_size, blocked_size);
    FLA_Set_tensor_stride(order, blocked_size, blocked_stride);

    sym.order = order;
    sym.nSymGroups = 1;
    sym.symGroupLens[0] = sym.order;
    for (i = 0; i < sym.order; i++) (sym.symModes)[i] = i;

    FLA_Obj_create_blocked_psym_tensor(FLA_DOUBLE, order, size, blocked_stride, block_size, sym, obj);
    FLA_Set_zero_tensor(*obj);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> lambda(n);
    for (int r = 0; r < n; ++r) lambda[r] = dist(rng);

    std::vector<double> M(n * n);
    for (int irow = 0; irow < n; ++irow)
        for (int jcol = 0; jcol < n; ++jcol)
            M[irow * n + jcol] = dist(rng);

    std::vector<double> U;
    gram_schmidt(M, U, n);

    // Fill all n^order index tuples (each canonical position is written by every
    // permutation, as before). For order 3 this is bit-identical to the old triple loop.
    dim_t idx[FLA_MAX_ORDER];
    for (dim_t d = 0; d < order; ++d) idx[d] = 0;
    while (true) {
        double val = 0.0;
        for (int r = 0; r < n; ++r) {
            double term = lambda[r];
            for (dim_t d = 0; d < order; ++d) term *= U[idx[d] * n + r];
            val += term;
        }
        set_tensor_element_bccs(*obj, idx, order, val);
        // last index varies fastest (matches the old a,b,c loop nest order)
        dim_t d = order;
        while (d > 0 && ++idx[d - 1] == (dim_t)n) idx[--d] = 0;
        if (d == 0) break;
    }
}

void initSymmTensor(dim_t order, dim_t size[], dim_t b, FLA_Obj* obj){
    dim_t i;
    dim_t blocked_stride[FLA_MAX_ORDER];
    dim_t block_size[FLA_MAX_ORDER];
    dim_t blocked_size[FLA_MAX_ORDER];
    TLA_sym sym;

    for(i = 0; i < order; i++){
        block_size[i] = b;
    }

    FLA_array_elemwise_quotient(order, size, block_size, blocked_size);
    FLA_Set_tensor_stride(order, blocked_size, blocked_stride);

    sym.order = order;
    sym.nSymGroups = 1;
    sym.symGroupLens[0] = sym.order;
    for(i = 0; i < sym.order; i++)
        (sym.symModes)[i] = i;

    FLA_Obj_create_blocked_psym_tensor(FLA_DOUBLE, order, size, blocked_stride, block_size, sym, obj);
    FLA_Random_psym_tensor(*obj);
}

void initIdentityMatrix(dim_t n, dim_t bC, dim_t bA, FLA_Obj* B){
    dim_t order = 2;
    dim_t size[2] = { n, n };
    dim_t sizeObj[2] = { n / bC, n / bA };
    dim_t strideObj[2] = { 1, sizeObj[0] };
    dim_t sizeBlk[2] = { bC, bA };

    FLA_Obj_create_blocked_tensor(FLA_DOUBLE, order, size, strideObj, sizeBlk, B);
    FLA_Set_zero_tensor(*B);

    for (dim_t i = 0; i < n; ++i) {
        set_matrix_element(*B, i, i, 1.0);
    }
}

void initZeroMatrix(dim_t n, dim_t bC, dim_t bA, FLA_Obj* B){
    dim_t order = 2;
    dim_t size[2] = { n, n };
    dim_t sizeObj[2] = { n / bC, n / bA };
    dim_t strideObj[2] = { 1, sizeObj[0] };
    dim_t sizeBlk[2] = { bC, bA };

    FLA_Obj_create_blocked_tensor(FLA_DOUBLE, order, size, strideObj, sizeBlk, B);
    FLA_Set_zero_tensor(*B);
}

void initIdentityDenseMatrix(dim_t n, FLA_Obj* B){
    FLA_Obj_create(FLA_DOUBLE, n, n, 0, 0, B);
    FLA_Set(FLA_ZERO, *B);

    for (dim_t i = 0; i < n; ++i) {
        set_dense_matrix_element(*B, i, i, 1.0);
    }
}

void initZeroDenseMatrix(dim_t n, FLA_Obj* B){
    FLA_Obj_create(FLA_DOUBLE, n, n, 0, 0, B);
    FLA_Set(FLA_ZERO, *B);
}

void setIdentityMatrix(dim_t n, FLA_Obj* G_sttsm){
    FLA_Set_zero_tensor(*G_sttsm);
    for(dim_t i=0; i<n; i++){
            set_matrix_element_bccs(*G_sttsm, i, i, 1.0);
        }
}

void fill_intra_block_symmetry(FLA_Obj T, dim_t order, dim_t block_size){
    FLA_Obj* buf = (FLA_Obj*)FLA_Obj_base_buffer(T);
    dim_t* outer_stride = FLA_Obj_stride(T);
    dim_t nb = FLA_Obj_dimsize(T, 0);

    dim_t ls[FLA_MAX_ORDER];
    ls[0] = 1;
    for (dim_t m = 1; m < order; m++) ls[m] = ls[m - 1] * block_size;

    // Stored blocks are the canonical ones: bidx[0] <= bidx[1] <= ... (non-decreasing).
    dim_t bidx[FLA_MAX_ORDER];
    for (dim_t m = 0; m < order; m++) bidx[m] = 0;
    while (true) {
        dim_t lin = 0;
        for (dim_t m = 0; m < order; m++) lin += bidx[m] * outer_stride[m];
        if (buf[lin].isStored) {
            double* data = (double*)FLA_Obj_base_buffer(buf[lin]);

            dim_t lidx[FLA_MAX_ORDER];
            for (dim_t m = 0; m < order; m++) lidx[m] = 0;
            while (true) {
                dim_t g[FLA_MAX_ORDER];
                bool canonical = true;
                for (dim_t m = 0; m < order; m++) g[m] = bidx[m] * block_size + lidx[m];
                for (dim_t m = 1; m < order; m++) if (g[m - 1] > g[m]) { canonical = false; break; }

                if (!canonical) {
                    dim_t sg[FLA_MAX_ORDER];
                    for (dim_t m = 0; m < order; m++) sg[m] = g[m];
                    std::sort(sg, sg + order);
                    dim_t dst = 0, src = 0;
                    for (dim_t m = 0; m < order; m++) {
                        dst += lidx[m] * ls[m];
                        src += (sg[m] - bidx[m] * block_size) * ls[m];
                    }
                    data[dst] = data[src];
                }

                dim_t m = order;
                while (m > 0 && ++lidx[m - 1] == block_size) lidx[--m] = 0;
                if (m == 0) break;
            }
        }

        // advance to next non-decreasing block index tuple (last index fastest)
        dim_t pos = order;
        while (pos > 0 && bidx[pos - 1] == nb - 1) pos--;
        if (pos == 0) break;
        bidx[pos - 1]++;
        for (dim_t m = pos; m < order; m++) bidx[m] = bidx[pos - 1];
    }
}

int build_group_rotation_slots(dim_t n, const int* pair_p, const int* pair_q,
                                const double* pair_c, const double* pair_s, int num_pairs,
                                RotSlot* slots_out, double eps_pivot){
    std::vector<bool> touched(n, false);
    int num_slots = 0;
    for (int i = 0; i < num_pairs; i++){
        if (fabs(pair_s[i]) > eps_pivot){
            dim_t p = (dim_t)pair_p[i], q = (dim_t)pair_q[i];
            double c = pair_c[i], s = pair_s[i];
            // M = G restricted to rows/cols {p,q} = [[c,s],[-s,c]] (T <- T x1 G x2 G x3 G)
            slots_out[num_slots++] = RotSlot{p, p, q, c, s, -s, c};
            touched[p] = true;
            touched[q] = true;
        }
    }
    for (dim_t m = 0; m < n; m++){
        if (!touched[m]){
            slots_out[num_slots++] = RotSlot{m, m, m, 1.0, 0.0, 0.0, 1.0};
        }
    }
    std::sort(slots_out, slots_out + num_slots,
              [](const RotSlot& a, const RotSlot& b){ return a.rep < b.rep; });
    return num_slots;
}

void apply_group_rotation_cellwise(FLA_Obj T, const RotSlot* slots, int num_slots){
    // Fast path for the hot get/set calls below: FLA_Obj_tensor_buffer_at_view
    // (used by get_tensor_element_bccs/set_tensor_element_bccs) mallocs+frees a
    // stride array and an offset array on every single call, on top of copying
    // whole FLA_Obj structs across a non-LTO'd libflame call boundary. Block
    // layout (block_size, n_blocks_per_mode, and each block's own base pointer
    // and stride) is invariant for this whole call, so resolve it once here and
    // index directly with pointer arithmetic instead.
    FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);
    dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
    dim_t n_blocks_per_mode = FLA_Obj_dimsize(T, 0);

    auto elem_ptr = [&](dim_t i, dim_t j, dim_t k) -> double* {
        sort3(i, j, k);
        dim_t bi = i / block_size, li = i % block_size;
        dim_t bj = j / block_size, lj = j % block_size;
        dim_t bk = k / block_size, lk = k % block_size;
        dim_t lb = (bi * n_blocks_per_mode + bj) * n_blocks_per_mode + bk;
        FLA_Base_obj* base = blocks[lb].base;
        double* buf = (double*)base->buffer;
        return buf + li * base->stride[0] + lj * base->stride[1] + lk * base->stride[2];
    };

    for (int si = 0; si < num_slots; si++){
        for (int sj = si; sj < num_slots; sj++){
            for (int sk = sj; sk < num_slots; sk++){
                const RotSlot& Si = slots[si];
                const RotSlot& Sj = slots[sj];
                const RotSlot& Sk = slots[sk];
                // All three slots identity (idx0==idx1, M=I): the cell maps onto
                // itself unchanged (8 reads of the same element written back
                // unchanged) - skip it.
                if (Si.idx0 == Si.idx1 && Sj.idx0 == Sj.idx1 && Sk.idx0 == Sk.idx1) continue;
                dim_t Ii[2] = {Si.idx0, Si.idx1};
                dim_t Ij[2] = {Sj.idx0, Sj.idx1};
                dim_t Ik[2] = {Sk.idx0, Sk.idx1};
                double Mi[2][2] = {{Si.m00, Si.m01}, {Si.m10, Si.m11}};
                double Mj[2][2] = {{Sj.m00, Sj.m01}, {Sj.m10, Sj.m11}};
                double Mk[2][2] = {{Sk.m00, Sk.m01}, {Sk.m10, Sk.m11}};

                // Read all old values for this cell before writing any of them.
                double old_local[2][2][2];
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                        for (int c2 = 0; c2 < 2; c2++)
                            old_local[a][b][c2] = *elem_ptr(Ii[a], Ij[b], Ik[c2]);

                double tmp1[2][2][2]; // mode-1
                for (int a2 = 0; a2 < 2; a2++)
                    for (int b = 0; b < 2; b++)
                        for (int c2 = 0; c2 < 2; c2++)
                            tmp1[a2][b][c2] = Mi[a2][0]*old_local[0][b][c2] + Mi[a2][1]*old_local[1][b][c2];

                double tmp2[2][2][2]; // mode-2
                for (int a2 = 0; a2 < 2; a2++)
                    for (int b2 = 0; b2 < 2; b2++)
                        for (int c2 = 0; c2 < 2; c2++)
                            tmp2[a2][b2][c2] = Mj[b2][0]*tmp1[a2][0][c2] + Mj[b2][1]*tmp1[a2][1][c2];

                double new_local[2][2][2]; // mode-3
                for (int a2 = 0; a2 < 2; a2++)
                    for (int b2 = 0; b2 < 2; b2++)
                        for (int c2b = 0; c2b < 2; c2b++)
                            new_local[a2][b2][c2b] = Mk[c2b][0]*tmp2[a2][b2][0] + Mk[c2b][1]*tmp2[a2][b2][1];

                for (int a = 0; a < 2; a++){
                    for (int b = 0; b < 2; b++){
                        for (int c2 = 0; c2 < 2; c2++){
                            *elem_ptr(Ii[a], Ij[b], Ik[c2]) = new_local[a][b][c2];
                        }
                    }
                }
            }
        }
    }
}

void apply_group_rotation_cellwise_general(FLA_Obj T, dim_t order, const RotSlot* slots, int num_slots){
    // Any-order version of apply_group_rotation_cellwise: one cell per non-decreasing
    // slot tuple (s_0 <= ... <= s_{d-1}), each holding <= 2^d entries. Bit m of the local
    // mask selects idx1 (else idx0) of slot s_m; mode m is transformed by that slot's 2x2.
    FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);
    dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
    dim_t nb = FLA_Obj_dimsize(T, 0);
    const int d = (int)order;
    const int L = 1 << d;

    auto elem_ptr = [&](const dim_t* idx_in) -> double* {
        dim_t idx[FLA_MAX_ORDER];
        for (int m = 0; m < d; m++) idx[m] = idx_in[m];
        std::sort(idx, idx + d);
        dim_t lb = 0;
        for (int m = 0; m < d; m++) lb = lb * nb + idx[m] / block_size;
        FLA_Base_obj* base = blocks[lb].base;
        double* p = (double*)base->buffer;
        for (int m = 0; m < d; m++) p += (idx[m] % block_size) * base->stride[m];
        return p;
    };

    int sl[FLA_MAX_ORDER];
    for (int m = 0; m < d; m++) sl[m] = 0;
    double vals[1 << FLA_MAX_ORDER];
    double* ptrs[1 << FLA_MAX_ORDER];

    while (true) {
        bool all_identity = true;
        for (int m = 0; m < d; m++)
            if (slots[sl[m]].idx0 != slots[sl[m]].idx1) { all_identity = false; break; }

        if (!all_identity) {
            for (int mask = 0; mask < L; mask++) {
                dim_t idx[FLA_MAX_ORDER];
                for (int m = 0; m < d; m++)
                    idx[m] = ((mask >> m) & 1) ? slots[sl[m]].idx1 : slots[sl[m]].idx0;
                ptrs[mask] = elem_ptr(idx);
                vals[mask] = *ptrs[mask];
            }
            for (int m = 0; m < d; m++) {
                const RotSlot& S = slots[sl[m]];
                for (int mask = 0; mask < L; mask++) {
                    if ((mask >> m) & 1) continue;
                    int hi = mask | (1 << m);
                    double x0 = vals[mask], x1 = vals[hi];
                    vals[mask] = S.m00 * x0 + S.m01 * x1;
                    vals[hi]   = S.m10 * x0 + S.m11 * x1;
                }
            }
            for (int mask = 0; mask < L; mask++) *ptrs[mask] = vals[mask];
        }

        int pos = d;
        while (pos > 0 && sl[pos - 1] == num_slots - 1) pos--;
        if (pos == 0) break;
        sl[pos - 1]++;
        for (int m = pos; m < d; m++) sl[m] = sl[pos - 1];
    }
}

//Frees up memory - Tensor
void cleanup_tensor(FLA_Obj* T){
    FLA_Obj_blocked_psym_tensor_free_buffer(T);
    FLA_Obj_free_without_buffer(T);
}

//Copies tensor values
void copy_tensor_values(FLA_Obj src, FLA_Obj dst){
    dim_t n_blocks = FLA_Obj_num_elem_alloc(src);
    FLA_Obj* src_buf = (FLA_Obj*)FLA_Obj_base_buffer(src);
    FLA_Obj* dst_buf = (FLA_Obj*)FLA_Obj_base_buffer(dst);
    
    for (dim_t i = 0; i < n_blocks; i++) {
        if (src_buf[i].isStored) {
            dim_t n_elem = FLA_Obj_num_elem_alloc(src_buf[i]);
            memcpy(FLA_Obj_base_buffer(dst_buf[i]), FLA_Obj_base_buffer(src_buf[i]), n_elem * sizeof(double));
        }
    }
}

void copy_matrix_values(FLA_Obj src, FLA_Obj dst){
    dim_t n_blocks = FLA_Obj_num_elem_alloc(src);
    FLA_Obj* src_buf = (FLA_Obj*)FLA_Obj_base_buffer(src);
    FLA_Obj* dst_buf = (FLA_Obj*)FLA_Obj_base_buffer(dst);

    for (dim_t i = 0; i < n_blocks; i++) {
        dim_t n_elem = FLA_Obj_num_elem_alloc(src_buf[i]);
        memcpy(FLA_Obj_base_buffer(dst_buf[i]),
               FLA_Obj_base_buffer(src_buf[i]),
               n_elem * sizeof(double));
    }
}

//Frees up memory - Matrix
void cleanup_matrix(FLA_Obj* F){
    FLA_Obj_blocked_tensor_free_buffer(F);
    FLA_Obj_free_without_buffer(F);
}