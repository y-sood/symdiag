#pragma once
// standard headers first: FLAME.h defines min/max macros that break them
#include <memory>
#include <string>
#include "tensor_io.hpp"
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

enum class StopMode {
    Delta,
    Ratio,
    Both,
    Either,
    Fixed      // only the exact fixed-point rule (no rotation applied in a sweep); the default
};

struct JacobiConfig {
    //Tensor generation parameters
    dim_t order;
    dim_t n = 8;
    int seed = 42;
    //External input (see tensor_io.hpp). When set, order and n come from the file.
    std::string input_path;
    int index_base = 0;            // text input: 0 or 1
    double sym_tol = 1e-12;        // relative symmetry tolerance for external input
    std::string output_prefix;     // if set, write <prefix>_diag.csv and <prefix>_F.csv
    bool order_set = false, n_set = false, block_size_set = false;   // given explicitly on the command line
    mutable std::shared_ptr<LoadedTensor> loaded;   // parsed external tensor, released once copied into BCSS
    //Start from a random symmetric tensor (generally NOT exactly diagonalisable)
    //instead of the odeco test tensor
    bool generic = false;
    // Validate against a reference copy of the input. The copy is a full extra tensor held
    // through the sweeps (an arbitrary input has no seed to regenerate it from). Turn off
    // (--no-check) to run with only T and F alive, e.g. for memory measurements.
    bool check = true;
    // Also reconstruct from ALL entries of T_final (T_init = T_final x1 F x2 F ... xd F) and
    // report it next to the diagonal-only reconstruction. Needs dense n^order scratch.
    bool check_full = false;
    //BCCS parameter
    dim_t block_size = 4;
    //Jacobi parameter
    int n_iterations = 10;
    //Debugging
    bool debug = false;
    bool print_initial_tensor = false;
    bool print_final_tensor = false;
    bool print_factor_matrix = false;
    bool print_partition = false;
    bool print_matlab_summary = false;
    //Stopping criteria
    bool enable_stopping = true;
    // Absolute thresholds on delta (sweep-to-sweep change in diag_norm_sq) and
    // ratio (offdiag_norm_sq/diag_norm_sq). Tightened from 1e-10/1e-8: those
    // looser defaults let the stopping criterion trip one sweep before the
    // true fixed point on a meaningful fraction of runs, especially at larger
    // n (delta is an absolute, extensive quantity, so it scales up with n,
    // making it more likely to dip under a fixed threshold early) - verified
    // this cost 3-5 orders of magnitude of reconstruction accuracy in the
    // affected cases (e.g. 2e-7 instead of ~3e-13), with the "missing" sweep
    // reaching an exact bit-for-bit fixed point (delta=0.0) when allowed to
    // run. 1e-16 is tight enough to force that extra sweep whenever it
    // matters, while still being satisfied by an exact fixed point.
    double tol_delta = 1e-16;
    double tol_ratio = 1e-16;
    int min_iterations = 2;
    bool require_both = false;
    // Default stopping rule: stop when a full sweep applies no rotation (exact fixed point: every pivot
    // pair is below the skip threshold or has a rotation angle below eps_sine). This is discrete, so it
    // does not depend on rounding of norm differences, and it is scale-free (see eps_pivot). The
    // delta/ratio modes below are kept for compatibility with the benchmark scripts; note that the
    // "ratio" test is applied to rel_offdiag, not to off^2/diag^2 (historical, left unchanged because
    // the committed benchmark logs depend on it).
    StopMode stop_mode = StopMode::Fixed;
    // Optional extra rule: stop when the relative off-diagonal norm ||offdiag||_F / ||T||_F < tol_off
    // (0 = disabled).
    double tol_off = 0.0;
    // A pivot pair (p,q) is only solved for a rotation angle if some pivot entry A_k (0<k<order)
    // exceeds eps_pivot * ||T||_F. ||T||_F is invariant under the rotations, so this threshold is RELATIVE
    // to the tensor scale: multiplying the input by any constant gives the same sweeps. (It used to be an
    // absolute 1e-14, which left tensors with entries << 1 undiagonalised: an input scaled by 1e-10 stopped
    // at rel_offdiag 1e-4.) The default 1e-15 was chosen by experiment (12 seeds x orders 3-6, n up to 64: every run still reached the fixed point,
    // typical final off-diagonal norm 2-4x better than 1e-14; worst cases unchanged).
    double eps_pivot = 1e-15;
    // A computed rotation counts as significant (is applied to T and F, and makes a sweep 'active') only if
    // |sin(theta)| > eps_sine. Dimensionless, so scale-free.
    double eps_sine = 1e-15;
    //Performance
    // Disables libflame's internal FLA_Check_* dimension/type validation (FLA_FULL_ERROR_CHECKING
    // is the library default). Measured ~2.6x speedup at n=48/block_size=4 with bit-identical
    // results, but it also silences libflame's own error messages, so keep it opt-in while
    // precision issues in the algorithm are still being tracked down.
    bool fast_unsafe = false;
};

JacobiConfig default_jacobi_config();
void parse_args(int argc, char** argv, JacobiConfig* config);
void print_run_header(const JacobiConfig& config);

