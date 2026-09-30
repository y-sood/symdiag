#include "tensor_io.hpp"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

namespace {

const int kMaxOrder = 12;   // FLA_MAX_ORDER

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string g6(double x) { char b[40]; snprintf(b, sizeof b, "%.6g", x); return b; }

std::string idx_str(const uint32_t* idx, int order, int base) {
    std::string s = "(";
    for (int m = 0; m < order; ++m) { if (m) s += ","; s += std::to_string((unsigned long)idx[m] + base); }
    return s + ")";
}

// Sort the canonical entries lexicographically and merge duplicates, verifying that duplicates agree.
bool merge_entries(int order, std::vector<uint32_t>& idx, std::vector<double>& val, double max_abs,
                   double sym_tol, int index_base, std::string* err) {
    size_t N = val.size();
    std::vector<size_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        return std::lexicographical_compare(idx.begin() + a * order, idx.begin() + (a + 1) * order,
                                            idx.begin() + b * order, idx.begin() + (b + 1) * order);
    });
    std::vector<uint32_t> nidx; std::vector<double> nval;
    nidx.reserve(idx.size()); nval.reserve(N);
    double tol = sym_tol * std::max(max_abs, 1e-300);
    for (size_t k = 0; k < N; ++k) {
        const uint32_t* cur = &idx[perm[k] * order];
        bool same = !nval.empty() && std::equal(cur, cur + order, nidx.end() - order);
        if (same) {
            double v0 = nval.back(), v1 = val[perm[k]];
            if (std::fabs(v0 - v1) > tol) {
                *err = "the tensor is not symmetric: entry " + idx_str(cur, order, index_base) +
                       " (and its index permutations) is listed with different values " +
                       g6(v0) + " and " + g6(v1) +
                       " (tolerance " + g6(tol) + "); this solver needs a symmetric tensor";
                return false;
            }
        } else {
            nidx.insert(nidx.end(), cur, cur + order);
            nval.push_back(val[perm[k]]);
        }
    }
    idx.swap(nidx); val.swap(nval);
    return true;
}

// ---------------- text / CSV ----------------
bool load_text(const std::string& path, int index_base, double sym_tol, LoadedTensor* out, std::string* err) {
    std::ifstream f(path);
    if (!f) { *err = "cannot open '" + path + "'"; return false; }
    std::string line;
    int order = 0; size_t header_n = 0; size_t max_index = 0; size_t lineno = 0;
    bool seen_data = false, any = false;
    std::vector<uint32_t> idx; std::vector<double> val; double max_abs = 0.0;
    std::vector<std::string> tok;
    while (std::getline(f, line)) {
        ++lineno;
        size_t p = line.find_first_not_of(" \t\r");
        if (p == std::string::npos) continue;
        if (line[p] == '#' || line[p] == '%') {
            size_t q = line.find("n=");
            if (q != std::string::npos) header_n = (size_t)std::strtoull(line.c_str() + q + 2, nullptr, 10);
            continue;
        }
        tok.clear();
        std::string cur;
        for (char ch : line) {
            if (ch == ',' || ch == ';' || ch == ' ' || ch == '\t' || ch == '\r') { if (!cur.empty()) { tok.push_back(cur); cur.clear(); } }
            else cur.push_back(ch);
        }
        if (!cur.empty()) tok.push_back(cur);
        if (tok.empty()) continue;
        if (!seen_data) {
            // optional header row (e.g. "i,j,k,value"): any letter in the first token means it is not data
            bool letters = false;
            for (char ch : tok[0]) if (std::isalpha((unsigned char)ch)) letters = true;
            if (letters) continue;
        }
        if (tok.size() < 4) { *err = "line " + std::to_string(lineno) + ": expected 'i,j,k,...,value' with at least 3 indices, got " + std::to_string(tok.size()) + " columns"; return false; }
        int ord = (int)tok.size() - 1;
        if (!seen_data) { order = ord; seen_data = true; if (order > kMaxOrder) { *err = "order " + std::to_string(order) + " exceeds the maximum " + std::to_string(kMaxOrder); return false; } }
        else if (ord != order) { *err = "line " + std::to_string(lineno) + ": " + std::to_string(ord) + " indices but earlier lines have " + std::to_string(order); return false; }
        uint32_t ii[kMaxOrder];
        for (int m = 0; m < order; ++m) {
            char* end = nullptr; errno = 0;
            long long v = std::strtoll(tok[m].c_str(), &end, 10);
            if (end == tok[m].c_str() || *end != '\0') { *err = "line " + std::to_string(lineno) + ": index '" + tok[m] + "' is not an integer"; return false; }
            v -= index_base;
            if (v < 0) { *err = "line " + std::to_string(lineno) + ": index " + tok[m] + " is below the index base " + std::to_string(index_base) + " (use --index-base)"; return false; }
            ii[m] = (uint32_t)v; max_index = std::max(max_index, (size_t)v);
        }
        char* end = nullptr;
        double x = std::strtod(tok[order].c_str(), &end);
        if (end == tok[order].c_str() || *end != '\0' || !std::isfinite(x)) { *err = "line " + std::to_string(lineno) + ": value '" + tok[order] + "' is not a finite number"; return false; }
        std::sort(ii, ii + order);
        any = true;
        idx.insert(idx.end(), ii, ii + order);   // explicit zeros are kept until after the merge so that a
        val.push_back(x); max_abs = std::max(max_abs, std::fabs(x));   // zero/non-zero disagreement is detected
    }
    if (!any) { *err = "no tensor entries found in '" + path + "'"; return false; }
    size_t n = max_index + 1;
    if (header_n) { if (header_n < n) { *err = "an index (" + std::to_string(max_index + index_base) + ") is outside the declared n=" + std::to_string(header_n); return false; } n = header_n; }
    if (!merge_entries(order, idx, val, max_abs, sym_tol, index_base, err)) return false;
    { // drop exact zeros (implicit in BCSS)
        size_t w = 0;
        for (size_t k = 0; k < val.size(); ++k) if (val[k] != 0.0) {
            if (w != k) { std::copy(idx.begin() + k * order, idx.begin() + (k + 1) * order, idx.begin() + w * order); val[w] = val[k]; }
            ++w;
        }
        val.resize(w); idx.resize(w * order);
    }
    out->order = order; out->n = n; out->idx.swap(idx); out->val.swap(val); out->max_abs = max_abs; out->format = "text";
    return true;
}

// ---------------- .npy ----------------
bool load_npy(const std::string& path, double sym_tol, LoadedTensor* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { *err = "cannot open '" + path + "'"; return false; }
    char magic[6]; f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) { *err = "'" + path + "' is not a NumPy .npy file"; return false; }
    unsigned char ver[2]; f.read((char*)ver, 2);
    size_t hlen = 0;
    if (ver[0] == 1) { unsigned char b[2]; f.read((char*)b, 2); hlen = b[0] | (b[1] << 8); }
    else { unsigned char b[4]; f.read((char*)b, 4); hlen = (size_t)b[0] | ((size_t)b[1] << 8) | ((size_t)b[2] << 16) | ((size_t)b[3] << 24); }
    std::string header(hlen, '\0'); f.read(&header[0], hlen);
    if (!f) { *err = "truncated .npy header"; return false; }
    bool f8 = header.find("'<f8'") != std::string::npos || header.find("'|f8'") != std::string::npos || header.find("\"<f8\"") != std::string::npos;
    bool f4 = header.find("'<f4'") != std::string::npos || header.find("\"<f4\"") != std::string::npos;
    if (!f8 && !f4) { *err = "unsupported .npy dtype (need little-endian float64 or float32); header: " + header; return false; }
    bool fortran = header.find("'fortran_order': True") != std::string::npos;
    size_t sp = header.find("'shape'");
    if (sp == std::string::npos) { *err = "no shape in .npy header"; return false; }
    size_t lp = header.find('(', sp), rp = header.find(')', sp);
    std::vector<size_t> shape;
    { std::string s = header.substr(lp + 1, rp - lp - 1); std::stringstream ss(s); std::string t;
      while (std::getline(ss, t, ',')) { if (t.find_first_of("0123456789") != std::string::npos) shape.push_back((size_t)std::strtoull(t.c_str(), nullptr, 10)); } }
    int order = (int)shape.size();
    if (order < 3) { *err = "the array has " + std::to_string(order) + " dimensions; a tensor of order >= 3 is required"; return false; }
    if (order > kMaxOrder) { *err = "order " + std::to_string(order) + " exceeds the maximum " + std::to_string(kMaxOrder); return false; }
    size_t n = shape[0];
    for (size_t s : shape) if (s != n) { *err = "all modes must have the same length (shape is not (n,...,n))"; return false; }
    double total = 1.0; for (int m = 0; m < order; ++m) total *= (double)n;
    size_t N = (size_t)total;
    std::vector<double> data(N);
    if (f8) { f.read((char*)data.data(), N * sizeof(double)); }
    else { std::vector<float> tmp(N); f.read((char*)tmp.data(), N * sizeof(float)); for (size_t k = 0; k < N; ++k) data[k] = tmp[k]; }
    if (!f) { *err = "the .npy file is shorter than its header says (" + std::to_string(N) + " values expected)"; return false; }
    // strides (in elements) per mode for the stored layout
    std::vector<size_t> stride(order);
    if (!fortran) { size_t s = 1; for (int m = order - 1; m >= 0; --m) { stride[m] = s; s *= n; } }
    else          { size_t s = 1; for (int m = 0; m < order; ++m)  { stride[m] = s; s *= n; } }
    double max_abs = 0.0;
    for (size_t k = 0; k < N; ++k) { if (!std::isfinite(data[k])) { *err = "the array contains NaN or Inf"; return false; } max_abs = std::max(max_abs, std::fabs(data[k])); }
    double tol = sym_tol * std::max(max_abs, 1e-300);
    // symmetry check: invariance under every adjacent transposition generates all permutations
    std::vector<size_t> ix(order, 0);
    for (size_t cnt = 0; cnt < N; ++cnt) {
        size_t pos = 0; for (int m = 0; m < order; ++m) pos += ix[m] * stride[m];
        for (int m = 0; m + 1 < order; ++m) {
            if (ix[m] < ix[m + 1]) {
                size_t pos2 = pos - ix[m] * stride[m] - ix[m + 1] * stride[m + 1] + ix[m + 1] * stride[m] + ix[m] * stride[m + 1];
                if (std::fabs(data[pos] - data[pos2]) > tol) {
                    uint32_t a[kMaxOrder], b[kMaxOrder]; for (int q = 0; q < order; ++q) { a[q] = (uint32_t)ix[q]; b[q] = (uint32_t)ix[q]; }
                    std::swap(b[m], b[m + 1]);
                    *err = "the tensor is not symmetric: T" + idx_str(a, order, 0) + " = " + g6(data[pos]) + " but T" + idx_str(b, order, 0) + " = " + g6(data[pos2]) +
                           " (tolerance " + g6(tol) + "); this solver needs a symmetric tensor";
                    return false;
                }
            }
        }
        int m = order; while (m > 0 && ++ix[m - 1] == n) ix[--m] = 0;
    }
    // collect canonical entries (non-decreasing indices), skip exact zeros
    std::vector<uint32_t> idx; std::vector<double> val;
    std::fill(ix.begin(), ix.end(), 0);
    while (true) {
        size_t pos = 0; for (int m = 0; m < order; ++m) pos += ix[m] * stride[m];
        if (data[pos] != 0.0) { for (int m = 0; m < order; ++m) idx.push_back((uint32_t)ix[m]); val.push_back(data[pos]); }
        int p = order; while (p > 0 && ix[p - 1] == n - 1) p--;
        if (p == 0) break;
        ix[p - 1]++; for (int m = p; m < order; ++m) ix[m] = ix[p - 1];
    }
    out->order = order; out->n = n; out->idx.swap(idx); out->val.swap(val); out->max_abs = max_abs; out->format = "npy";
    return true;
}

}  // namespace

bool load_tensor_file(const std::string& path, int index_base, double sym_tol, LoadedTensor* out, std::string* err) {
    if (index_base != 0 && index_base != 1) { *err = "--index-base must be 0 or 1"; return false; }
    bool ok = ends_with(path, ".npy") ? load_npy(path, sym_tol, out, err) : load_text(path, index_base, sym_tol, out, err);
    if (!ok) return false;
    if (out->max_abs == 0.0) { *err = "the tensor is identically zero"; return false; }
    // Very large/small inputs: rescale by a power of two (exact in floating point, no rounding) so that
    // squares and products of entries cannot underflow/overflow in the norms and angle formulas.
    if (out->max_abs < 1e-6 || out->max_abs > 1e6) {
        int e = 0;
        std::frexp(out->max_abs, &e);
        for (double& v : out->val) v = std::ldexp(v, -e);
        out->max_abs = std::ldexp(out->max_abs, -e);
        out->scale_exp = e;
    }
    return true;
}

size_t largest_divisor_up_to(size_t n, size_t cap) {
    for (size_t d = std::min(n, cap); d >= 1; --d) if (n % d == 0) return d;
    return 1;
}

bool write_result_files(const std::string& prefix, const std::vector<double>& diag, const std::vector<double>& F, size_t n, std::string* err) {
    FILE* fd = fopen((prefix + "_diag.csv").c_str(), "w");
    FILE* ff = fopen((prefix + "_F.csv").c_str(), "w");
    if (!fd || !ff) { *err = "cannot write result files with prefix '" + prefix + "'"; if (fd) fclose(fd); if (ff) fclose(ff); return false; }
    fprintf(fd, "index,value\n");
    for (size_t i = 0; i < n; ++i) fprintf(fd, "%zu,%.17g\n", i, diag[i]);
    for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < n; ++j) fprintf(ff, j ? ",%.17g" : "%.17g", F[i * n + j]); fprintf(ff, "\n"); }
    fclose(fd); fclose(ff);
    return true;
}
