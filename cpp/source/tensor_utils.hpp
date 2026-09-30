#pragma once
#include <vector>
//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef abs
#undef abs
#endif

void set_dense_matrix_element(FLA_Obj A, dim_t row, dim_t col, double value);
double get_dense_matrix_element(FLA_Obj A, dim_t i, dim_t j);
void set_tensor_element_bccs(FLA_Obj T, dim_t* index, dim_t order, double value);
void set_matrix_element_bccs(FLA_Obj G, dim_t row, dim_t col, double value);
double get_tensor_element_bccs(FLA_Obj T, dim_t i, dim_t j, dim_t k);
double get_tensor_element_bccs_alt(FLA_Obj T, dim_t* index, dim_t order);
double get_matrix_element(FLA_Obj A, dim_t i, dim_t j);
void gram_schmidt(const std::vector<double>& M, std::vector<double>& U, int n);
int is_superdiagonal(const dim_t* idx, dim_t order);
void initDiagonalizableTensor(dim_t order, dim_t size[], dim_t b, FLA_Obj* obj, int n, unsigned int seed);
void initIdentityMatrix(dim_t n, dim_t bC, dim_t bA, FLA_Obj* B);
void initSymmTensor(dim_t order, dim_t size[], dim_t b, FLA_Obj* obj);
void initZeroMatrix(dim_t n, dim_t bC, dim_t bA, FLA_Obj* B);
void initIdentityDenseMatrix(dim_t n, FLA_Obj* B);
void initZeroDenseMatrix(dim_t n, FLA_Obj* B);
void setIdentityMatrix(dim_t n, FLA_Obj* G_sttsm);
void fill_intra_block_symmetry(FLA_Obj T, dim_t order, dim_t block_size);

// One group's rotation, decomposed per mode index: an active pivot pair (|s| > 1e-6)
// is a 2-index slot with a 2x2 rotation; everything else (inactive pairs, and the
// leftover unpaired index when n is odd) is a 1-index identity slot (idx0 == idx1).
// m** is the mode transform M[new][old], i.e. new_local = M * old_local.
struct RotSlot {
    dim_t rep;
    dim_t idx0, idx1;
    double m00, m01, m10, m11;
};

// Builds the sorted slot list for a group of pivot-pair rotations. Returns the slot
// count (<= n); slots_out must have room for n entries. A pair becomes an active
// 2-index slot only if |pair_s[i]| > eps_pivot; otherwise it's an identity slot.
int build_group_rotation_slots(dim_t n, const int* pair_p, const int* pair_q,
                                const double* pair_c, const double* pair_s, int num_pairs,
                                RotSlot* slots_out, double eps_pivot);

// In-place replacement for FLA_Sttsm_with_psym_temps for a single group's rotation:
// T <- T x1 G x2 G x3 G, computed cell by cell (order 3 only). See CLAUDE.md's
// "Planned replacement kernel". Cells are disjoint, so this is safe in place; each
// cell reads its <= 8 old entries into a local buffer before writing any of them.
void apply_group_rotation_cellwise(FLA_Obj T, const RotSlot* slots, int num_slots);

// Any-order (>= 2) version of the above, same semantics. Slower per cell than the
// order-3 fast path (runtime loops, sorts indices per access) - a correct reference.
void apply_group_rotation_cellwise_general(FLA_Obj T, dim_t order, const RotSlot* slots, int num_slots);

void cleanup_tensor(FLA_Obj* T);
void copy_tensor_values(FLA_Obj src, FLA_Obj dst);
void copy_matrix_values(FLA_Obj src, FLA_Obj dst);
void cleanup_matrix(FLA_Obj* F);