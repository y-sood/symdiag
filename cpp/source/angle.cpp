#include "angle.hpp"
#include "tensor_utils.hpp"
#include <cmath>

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
void calculate_rotation_angle(FLA_Obj T, double* A_arr, dim_t order, double* c_out, double* s_out){
    //Retreive elements stored at above indices
    double Appp = A_arr[0];
    double Aqqq = A_arr[1];
    double Aqpp = A_arr[2];
    double Apqq = A_arr[3];

    //printf("          DEBUG: T[%d,%d,%d]=%.6f, T[%d,%d,%d]=%.6f, T[%d,%d,%d]=%.6f, T[%d,%d,%d]=%.6f\n", pp[0], pp[1], pp[2], App, qq[0], qq[1], qq[2], Aqq, qp[0], qp[1], qp[2], Aqp, pq[0], pq[1], pq[2], Apq);
    
    //Coefficients of cubic equation (matches jactdiagangleTsym.m: a=A(p,p,q)+A(p,q,q))
    double ca = Aqpp + Apqq;
    double cb = Appp - Aqqq + 2 * Aqpp - 2 * Apqq;
    double cc = Appp + Aqqq - 2 * Aqpp - 2 * Apqq;
    double cd = Apqq - Aqpp;

    //printf("          Cubic coeffs: ca=%.6f, cb=%.6f, cc=%.6f, cd=%.6f\n", ca, cb, cc, cd);
    
    //Get roots
    double roots[3];
    int num_roots = solve_cubic(ca, cb, cc, cd, roots);

    /*printf("          Found %d roots: ", num_roots);
    for (int i = 0; i < num_roots; i++) {
        printf("t[%d]=%.6f ", i, roots[i]);
    }
    printf("\n");*/
    
    //Pick the root that maximises objective function — baseline is no rotation (c=1, s=0)
    double best_c = 1.0, best_s = 0.0;
    double max_g = eval_obj(1.0, 0.0, Appp, Aqqq, Aqpp, Apqq);  // = Appp + Aqqq
    for (int i = 0; i < num_roots; i++) {
        double t = roots[i];
        double c_i = 1.0 / sqrt(1.0 + t * t);
        double s_i = t * c_i;
        
        // Apq=Aqpp=T[p,p,q], Aqp=Apqq=T[p,q,q] — matches MATLAB eval_obj sign convention
        double g = eval_obj(c_i, s_i, Appp, Aqqq, Aqpp, Apqq);
        
        if (g > max_g) {
            max_g = g;
            best_c = c_i;
            best_s = s_i;
        }
    }
    
    //Return best angles
    *c_out = best_c;
    *s_out = best_s;
}

//Embeds [p,q] sub-tensor contribution to rotation matrix
void embed_givens_rotation(FLA_Obj G, int p, int q, double c, double s){
    set_matrix_element_bccs(G, p, p, c);
    set_matrix_element_bccs(G, p, q, s);
    set_matrix_element_bccs(G, q, p, -s);
    set_matrix_element_bccs(G, q, q, c);
}

void embed_givens_rotation_dense(FLA_Obj G, int p, int q, double c, double s){
    set_dense_matrix_element(G, p, p, c);
    set_dense_matrix_element(G, p, q, s);
    set_dense_matrix_element(G, q, p, -s);
    set_dense_matrix_element(G, q, q, c);
}