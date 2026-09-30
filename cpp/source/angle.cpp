#include "angle.hpp"
#include "tensor_utils.hpp"
#include <cmath>
#include <complex>
#include <vector>

namespace {
    constexpr double kCubicEps = 1e-14;
}

//Returns the roots of a cubic equation with coefficients a, b, c and d
//Functional only for third order Tensors
/* ----- CARDANO'S METHOD ----- */
int solve_cubic(double a, double b, double c, double d, double* roots){
    if (fabs(a) < kCubicEps) {
        if (fabs(b) < kCubicEps) {
            if (fabs(c) < kCubicEps) return 0;
            roots[0] = -d / c;
            return 1;
        }
        
        double disc = c * c - 4 * b * d;
        if (disc < 0) return 0;
        roots[0] = (-c + sqrt(disc)) / (2 * b);
        roots[1] = (-c - sqrt(disc)) / (2 * b);
        return 2;
    }
    
    double p = b / a;
    double q = c / a;
    double r = d / a;
    
    double p_third = p / 3.0;
    double Q = (3 * q - p * p) / 9.0;
    double R = (9 * p * q - 27 * r - 2 * p * p * p) / 54.0;
    double D = Q * Q * Q + R * R;
    
    int num_real = 0;
    
    if (D > kCubicEps) {
        double sqrtD = sqrt(D);
        double S = cbrt(R + sqrtD);
        double T = cbrt(R - sqrtD);
        roots[0] = S + T - p_third;
        num_real = 1;
    } else {
        double theta = acos(R / sqrt(-Q * Q * Q));
        double sqrtQ = sqrt(-Q);
        roots[0] = 2 * sqrtQ * cos(theta / 3.0) - p_third;
        roots[1] = 2 * sqrtQ * cos((theta + 2 * M_PI) / 3.0) - p_third;
        roots[2] = 2 * sqrtQ * cos((theta + 4 * M_PI) / 3.0) - p_third;
        num_real = 3;
    }
    
    return num_real;
}

//Evaluates objective function for 2*2*2 sub-tensor for angle \phi and coefficients
double eval_obj(double c, double s, double App, double Aqq, double Apq, double Aqp){
    double c2 = c * c;
    double c3 = c2 * c;
    double s2 = s * s;
    double s3 = s2 * s;
    
    return c3 * (App + Aqq) + 3 * c2 * s * (Apq - Aqp) + 3 * c * s2 * (Apq + Aqp) + s3 * (Aqq - App);
}

//Returns the rotation angle for pivot pair p,q
//Functional only for third order Tensors
void calculate_rotation_angle(const FLA_Obj T, const double* A_arr, dim_t order, double* c_out, double* s_out){
    //Retreive elements stored at above indices
    double Appp = A_arr[0];
    double Aqqq = A_arr[1];
    double Aqpp = A_arr[2];
    double Apqq = A_arr[3];

    // Maximize Appp'^2 + Aqqq'^2 (the new diagonal entries after rotation),
    // not their raw sum Appp'+Aqqq'. The local 2x2x2 pivot sub-cube is closed
    // under this rotation (only p,q are touched), so its Frobenius norm is
    // exactly invariant: Appp'^2+Aqqq'^2+3*Appq'^2+3*Apqq'^2 is constant in
    // theta. So maximizing the diagonal sum-of-squares is exactly equivalent
    // to minimizing the off-diagonal residual Appq'^2+Apqq'^2 - the actual
    // goal - unlike the raw-sum objective previously used here (matching
    // jactdiagangleTsym.m), which can be stationary at theta=0 while
    // significant off-diagonal mass remains (verified against real pivot
    // data near convergence: this closed form reduces the residual by
    // 7-9 orders of magnitude in cases where the old cubic approach picked
    // theta=0 and did nothing). As a function of theta, Appp'^2+Aqqq'^2
    // reduces to a0 + a*cos(4*theta) + b*sin(4*theta) - period pi/2, per
    // CLAUDE.md's precision notes - with maximizer theta = atan2(b,a)/4.
    double a = Appp*Appp + Aqqq*Aqqq - 3*Aqpp*Aqpp - 3*Apqq*Apqq
             - 2*Appp*Apqq - 2*Aqpp*Aqqq;
    double b = 4.0 * (Appp*Aqpp - Apqq*Aqqq);

    double theta = atan2(b, a) / 4.0;

    *c_out = cos(theta);
    *s_out = sin(theta);
}

//Embeds [p,q] sub-tensor contribution to rotation matrix
void embed_givens_rotation(FLA_Obj G, int p, int q, double c, double s){
    set_matrix_element_bccs(G, p, p, c);
    set_matrix_element_bccs(G, p, q, s);
    set_matrix_element_bccs(G, q, p, -s);
    set_matrix_element_bccs(G, q, q, c);
}

void apply_givens_rotation_to_columns(FLA_Obj F, int p, int q, double c, double s){
    double* buf = (double*) FLA_Obj_buffer_at_view(F);
    dim_t rs = FLA_Obj_row_stride(F);
    dim_t cs = FLA_Obj_col_stride(F);
    dim_t n  = FLA_Obj_length(F);

    double* col_p = buf + (dim_t)p * cs;
    double* col_q = buf + (dim_t)q * cs;

    for (dim_t row = 0; row < n; row++){
        double fp = col_p[row * rs];
        double fq = col_q[row * rs];
        col_p[row * rs] = c * fp + s * fq;
        col_q[row * rs] = -s * fp + c * fq;
    }
}

namespace {
    typedef std::complex<double> cplx;

    // X[k][m + d]: coefficient of e^{i m theta} in cos^(d-k)(theta) sin^k(theta).
    // Computed by a 64-point DFT, then rounded to a multiple of 2^-(d+2): the true
    // coefficients are dyadic rationals, so the rounding makes the table exact and
    // avoids an O(eps * |A|) error floor in the objective coefficients.
    struct HarmTable {
        bool ready = false;
        std::vector<cplx> X;
    };

    const HarmTable& harm_table(int d){
        static HarmTable cache[FLA_MAX_ORDER + 1];
        HarmTable& t = cache[d];
        if (t.ready) return t;
        const int N = 64, W = 2 * d + 1;
        t.X.assign((d + 1) * W, cplx(0, 0));
        const double q = std::ldexp(1.0, d + 2);
        for (int k = 0; k <= d; k++){
            std::vector<double> f(N);
            for (int j = 0; j < N; j++){
                double th = 2.0 * M_PI * j / N;
                f[j] = std::pow(std::cos(th), d - k) * std::pow(std::sin(th), k);
            }
            for (int m = -d; m <= d; m++){
                cplx acc(0, 0);
                for (int j = 0; j < N; j++)
                    acc += f[j] * std::polar(1.0, -2.0 * M_PI * m * j / N);
                acc /= (double)N;
                t.X[k * W + (m + d)] = cplx(std::round(acc.real() * q) / q,
                                            std::round(acc.imag() * q) / q);
            }
        }
        t.ready = true;
        return t;
    }

    double binom(int n, int k){
        double r = 1.0;
        for (int i = 1; i <= k; i++) r = r * (n - k + i) / i;
        return std::round(r);
    }
}

void calculate_rotation_angle_general(const double* A, dim_t order, double* c_out, double* s_out){
    const int d = (int)order;
    const int W = 2 * d + 1;
    const int M = d / 2;
    const HarmTable& tab = harm_table(d);

    // A_0'(theta) = sum_k C(d,k) c^(d-k) s^k A_k
    // A_d'(theta) = sum_k C(d,k) (-s)^(d-k) c^k A_k   (c^k s^(d-k) is table row d-k)
    std::vector<cplx> U(W, cplx(0, 0)), V(W, cplx(0, 0));
    for (int k = 0; k <= d; k++){
        double wu = binom(d, k) * A[k];
        double wv = binom(d, k) * ((d - k) % 2 ? -1.0 : 1.0) * A[k];
        for (int m = 0; m < W; m++){
            U[m] += wu * tab.X[k * W + m];
            V[m] += wv * tab.X[(d - k) * W + m];
        }
    }
    // f = U*U + V*V (convolution); keep the frequencies 4j, j=1..M, i.e. F[j].
    // (All other non-zero frequencies cancel exactly - see higher_order.md.)
    std::vector<cplx> F(M + 1, cplx(0, 0));
    for (int j = 1; j <= M; j++){
        int w = 4 * j;
        for (int a = 0; a < W; a++){
            int b = w + 2 * d - a; // index into the second factor: (a-d)+(b-d)=w
            if (b >= 0 && b < W) F[j] += U[a] * U[b] + V[a] * V[b];
        }
    }

    auto g = [&](double phi){
        double r = 0.0;
        for (int j = 1; j <= M; j++)
            r += F[j].real() * std::cos(j * phi) - F[j].imag() * std::sin(j * phi);
        return 2.0 * r;
    };
    auto dg = [&](double phi){
        double r = 0.0;
        for (int j = 1; j <= M; j++)
            r += j * (-F[j].real() * std::sin(j * phi) - F[j].imag() * std::cos(j * phi));
        return 2.0 * r;
    };

    const int NG = 32 * M;
    const double h = 2.0 * M_PI / NG;
    std::vector<double> gv(NG);
    for (int i = 0; i < NG; i++) gv[i] = g(-M_PI + i * h);

    double best_phi = 0.0, best_g = -INFINITY;
    for (int i = 0; i < NG; i++){
        if (!(gv[i] >= gv[(i + NG - 1) % NG] && gv[i] >= gv[(i + 1) % NG])) continue;
        double phi = -M_PI + i * h;
        double lo = phi - h, hi = phi + h;
        if (dg(lo) > 0 && dg(hi) < 0){
            for (int it = 0; it < 64; it++){
                double mid = 0.5 * (lo + hi);
                if (dg(mid) > 0) lo = mid; else hi = mid;
            }
            phi = 0.5 * (lo + hi);
        }
        double gval = g(phi);
        if (gval > best_g){ best_g = gval; best_phi = phi; }
    }

    double theta = best_phi / 4.0;
    *c_out = std::cos(theta);
    *s_out = std::sin(theta);
}
