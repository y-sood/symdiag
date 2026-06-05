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
    // Sort to canonical form for symmetric tensor
    dim_t* canonical_index = (dim_t*)malloc(order * sizeof(dim_t));
    memcpy(canonical_index, index, order * sizeof(dim_t));
    std::sort(canonical_index, canonical_index + order);
    
    // Check if tensor is blocked
    if (FLA_Obj_elemtype(T) == FLA_TENSOR) {
        // BCCS: Navigate to correct block first
        FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);
        
        // Get block size
        dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
        
        // Calculate which block and local index within block
        dim_t* block_index = (dim_t*)malloc(order * sizeof(dim_t));
        dim_t* local_index = (dim_t*)malloc(order * sizeof(dim_t));
        
        for (dim_t i = 0; i < order; i++) {
            block_index[i] = canonical_index[i] / block_size;
            local_index[i] = canonical_index[i] % block_size;
        }
        
        // Calculate linear block index (row-major)
        dim_t n_blocks_per_mode = FLA_Obj_dimsize(T, 0);
        dim_t linear_block_idx = 0;
        dim_t stride = 1;
        for (dim_t i = order; i > 0; i--) {
            linear_block_idx += block_index[i-1] * stride;
            stride *= n_blocks_per_mode;
        }
        
        // Get the specific block
        FLA_Obj block = blocks[linear_block_idx];
        
        // Create view with local offset
        FLA_Obj block_view = block;
        memcpy(&(block_view.offset[0]), local_index, order * sizeof(dim_t));
        
        // Set element in block
        double* buffer = (double*)FLA_Obj_tensor_buffer_at_view(block_view);
        *buffer = value;
        
        free(canonical_index);
        free(block_index);
        free(local_index);
        
    } else {
        // Scalar tensor (non-blocked)
        FLA_Obj T_view = T;
        memcpy(&(T_view.offset[0]), canonical_index, order * sizeof(dim_t));
        
        double* buffer = (double*)FLA_Obj_tensor_buffer_at_view(T_view);
        *buffer = value;
        
        free(canonical_index);
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

double get_tensor_element_bccs(FLA_Obj T, dim_t* index, dim_t order){
    // Sort to canonical form for symmetric tensor
    dim_t* canonical_index = (dim_t*)malloc(order * sizeof(dim_t));
    memcpy(canonical_index, index, order * sizeof(dim_t));
    std::sort(canonical_index, canonical_index + order);
    
    //CHECK IF TENSOR IS BLOCKED
    if (FLA_Obj_elemtype(T) == FLA_TENSOR) {
        //BCCS: Navigate to correct block first
        FLA_Obj* blocks = (FLA_Obj*)FLA_Obj_base_buffer(T);
        
        //Get block size
        dim_t block_size = FLA_Obj_dimsize(blocks[0], 0);
        
        //Calculate which block and local index within block
        dim_t* block_index = (dim_t*)malloc(order * sizeof(dim_t));
        dim_t* local_index = (dim_t*)malloc(order * sizeof(dim_t));
        
        for (dim_t i = 0; i < order; i++) {
            block_index[i] = canonical_index[i] / block_size;
            local_index[i] = canonical_index[i] % block_size;
        }
        
        //Calculate linear block index (row-major for now)
        dim_t n_blocks_per_mode = FLA_Obj_dimsize(T, 0);
        dim_t linear_block_idx = 0;
        dim_t stride = 1;
        for (dim_t i = order; i > 0; i--) {
            linear_block_idx += block_index[i-1] * stride;
            stride *= n_blocks_per_mode;
        }
        
        //Get the specific block
        FLA_Obj block = blocks[linear_block_idx];
        
        //Create view with local offset
        FLA_Obj block_view = block;
        memcpy(&(block_view.offset[0]), local_index, order * sizeof(dim_t));
        
        //Get element from block
        double* buffer = (double*)FLA_Obj_tensor_buffer_at_view(block_view);
        double value = *buffer;
        
        free(canonical_index);
        free(block_index);
        free(local_index);
        
        return value;
        
    } else {
        //Scalar tensor (non-blocked): original code
        FLA_Obj T_view = T;
        memcpy(&(T_view.offset[0]), canonical_index, order * sizeof(dim_t));
        
        double* buffer = (double*)FLA_Obj_tensor_buffer_at_view(T_view);
        double value = *buffer;
        
        free(canonical_index);
        return value;
    }
}

double get_matrix_element(FLA_Obj A, dim_t i, dim_t j){
    dim_t idx[2] = {i, j};
    return get_tensor_element_bccs(A, idx, 2);
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

    for (int a = 0; a < n; ++a) {
        for (int bb = 0; bb < n; ++bb) {
            for (int c = 0; c < n; ++c) {
                double val = 0.0;
                for (int r = 0; r < n; ++r) {
                    val += lambda[r] * U[a * n + r] * U[bb * n + r] * U[c * n + r];
                }
                dim_t idx[3] = {(dim_t)a, (dim_t)bb, (dim_t)c};
                set_tensor_element_bccs(*obj, idx, order, val);
            }
        }
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

void fill_intra_block_symmetry(FLA_Obj T, dim_t order, dim_t block_size){
    FLA_Obj* buf = (FLA_Obj*)FLA_Obj_base_buffer(T);
    dim_t* outer_stride = FLA_Obj_stride(T);
    dim_t nb = FLA_Obj_dimsize(T, 0);

    dim_t ls[3];
    ls[0] = 1;
    ls[1] = block_size;
    ls[2] = block_size * block_size;

    for (dim_t bi = 0; bi < nb; bi++)
    for (dim_t bj = bi; bj < nb; bj++)
    for (dim_t bk = bj; bk < nb; bk++) {
        dim_t lin = bi*outer_stride[0] + bj*outer_stride[1] + bk*outer_stride[2];
        if (!buf[lin].isStored) continue;
        double* data = (double*)FLA_Obj_base_buffer(buf[lin]);

        for (dim_t r = 0; r < block_size; r++)
        for (dim_t s = 0; s < block_size; s++)
        for (dim_t t = 0; t < block_size; t++) {
            dim_t g0 = bi*block_size + r;
            dim_t g1 = bj*block_size + s;
            dim_t g2 = bk*block_size + t;
            if (g0 <= g1 && g1 <= g2) continue;

            dim_t sg[3] = {g0, g1, g2};
            if (sg[0] > sg[1]) { dim_t tmp = sg[0]; sg[0] = sg[1]; sg[1] = tmp; }
            if (sg[1] > sg[2]) { dim_t tmp = sg[1]; sg[1] = sg[2]; sg[2] = tmp; }
            if (sg[0] > sg[1]) { dim_t tmp = sg[0]; sg[0] = sg[1]; sg[1] = tmp; }

            dim_t sr = sg[0] - bi*block_size;
            dim_t ss = sg[1] - bj*block_size;
            dim_t st = sg[2] - bk*block_size;

            data[r*ls[0] + s*ls[1] + t*ls[2]] =
                data[sr*ls[0] + ss*ls[1] + st*ls[2]];
        }
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