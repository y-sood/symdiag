#pragma once
// External tensor input/output. Independent of FLAME: parses a file into the canonical entries
// (non-decreasing index tuples) of a symmetric tensor; jacobi.cpp writes them into the BCSS tensor.
//
// Supported input formats (chosen by file extension):
//   *.npy   NumPy dense array, float64 (or float32), C or Fortran order, shape (n,n,...,n), order >= 3.
//           Must be symmetric under every index permutation (checked).
//   other   Text / CSV coordinate list, one entry per line:  i,j,k,...,value
//           - separators: comma, semicolon, tab or spaces; blank lines and lines starting with '#' or '%' ignored
//           - an optional header line of letters (e.g. "i,j,k,value") is skipped
//           - an optional comment "# order=3 n=8" fixes n (otherwise n = 1 + largest index)
//           - order = number of columns - 1 (all rows must agree); indices are integers, base 0 by default
//           - list only canonical entries (i<=j<=k) or all permutations; listed permutations of the same
//             entry must agree within the symmetry tolerance; entries not listed are zero.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct LoadedTensor {
    int order = 0;
    size_t n = 0;
    std::vector<uint32_t> idx;   // canonical entries, `order` indices each, flattened
    std::vector<double> val;     // one value per entry (exact zeros are omitted)
    double max_abs = 0.0;
    std::string format;          // "npy" or "text"
    int scale_exp = 0;           // values were multiplied by 2^-scale_exp (exact) when max|entry| is outside [1e-6,1e6]
    size_t entries() const { return val.size(); }
};

// Returns true on success. On failure returns false and fills *err with a message meant for the user.
// index_base: 0 or 1 (text only). sym_tol: relative symmetry tolerance (relative to max |entry|).
bool load_tensor_file(const std::string& path, int index_base, double sym_tol, LoadedTensor* out, std::string* err);

// Largest divisor of n that is <= cap (used to pick a default block size for external inputs).
size_t largest_divisor_up_to(size_t n, size_t cap);

// Writes <prefix>_diag.csv ("index,value" rows) and <prefix>_F.csv (n x n, comma separated, %.17g).
bool write_result_files(const std::string& prefix, const std::vector<double>& diag,
                        const std::vector<double>& F_rowmajor, size_t n, std::string* err);
