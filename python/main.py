#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple


import numpy as np


@dataclass
class IterationRecord:
    iter: int
    diag_norm_sq: float
    offdiag_norm_sq: float
    ratio: float
    delta_diag: float
    trace: float
    rel_offdiag: float
    skipped_pairs: int
    total_pairs: int


@dataclass
class RunResult:
    T_final: np.ndarray
    U_final: np.ndarray
    history: List[IterationRecord]
    trace_history: List[float]
    rel_offdiag_history: List[float]
    final_record: IterationRecord
    T_reconstructed: np.ndarray
    reconstruction_error_abs: float
    reconstruction_error_rel: float
    reconstruction_max_abs : float
    reconstruction_max_rel : float


@dataclass
class TensorGenerationResult:
    T0: np.ndarray
    source_diag: Optional[np.ndarray] = None
    source_factor: Optional[np.ndarray] = None
    source_eigs: Optional[np.ndarray] = None


def gram_schmidt(M: np.ndarray) -> np.ndarray:
    n = M.shape[1]
    U = M.astype(float).copy()
    for j in range(n):
        for k in range(j):
            dot = float(np.dot(U[:, k], M[:, j]))
            U[:, j] -= dot * U[:, k]
        norm = float(np.linalg.norm(U[:, j]))
        if norm < 1e-15:
            raise ValueError("Gram-Schmidt produced a near-zero column")
        U[:, j] /= norm
    return U


def build_fixed_diag_tensor() -> TensorGenerationResult:
    lam = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    M = np.array(
        [
            [0.5, 0.3, 0.7, 0.2],
            [0.1, 0.8, 0.4, 0.6],
            [0.9, 0.2, 0.1, 0.7],
            [0.3, 0.6, 0.5, 0.8],
        ],
        dtype=float,
    )
    U = gram_schmidt(M)
    D = np.zeros((4, 4, 4), dtype=float)
    for i, val in enumerate(lam):
        D[i, i, i] = val
    T0 = tucker_sym(D, U)
    return TensorGenerationResult(T0=T0, source_diag=D, source_factor=U, source_eigs=lam)


def build_demo_tensor() -> TensorGenerationResult:
    T = np.zeros((4, 4, 4), dtype=float)
    T[0, 0, 0] = 1.0
    T[0, 0, 1] = T[0, 1, 0] = T[1, 0, 0] = 0.5
    T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = 0.3
    T[1, 1, 1] = 2.0
    T[0, 1, 2] = T[0, 2, 1] = T[1, 0, 2] = 0.2
    T[1, 2, 0] = T[2, 0, 1] = T[2, 1, 0] = 0.2
    T[0, 2, 2] = T[2, 0, 2] = T[2, 2, 0] = 0.4
    T[2, 2, 2] = 3.0
    T[3, 3, 3] = 4.0
    return TensorGenerationResult(T0=T)


def build_random_orthogonally_diagonalizable_tensor(n: int, seed: Optional[int]) -> TensorGenerationResult:
    rng = np.random.default_rng(seed)
    diag_vals = rng.random(n)
    D = np.zeros((n, n, n), dtype=float)
    for i, val in enumerate(diag_vals):
        D[i, i, i] = float(val)

    M = rng.random((n, n))
    U, _ = np.linalg.qr(M)
    T0 = tucker_sym(D, U)
    return TensorGenerationResult(T0=T0, source_diag=D, source_factor=U, source_eigs=diag_vals)


def tucker_sym(T: np.ndarray, G: np.ndarray) -> np.ndarray:
    return np.einsum("pqr,ip,jq,kr->ijk", T, G, G, G)


def gredcheck_tsym(T: np.ndarray, U: np.ndarray, p: int, q: int, eta: float) -> bool:
    n = T.shape[0]

    pom = np.einsum('abc,jb,kc->ajk', T, U, U)

    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            G[i, j] = 3.0 * pom[i, j, j]

    R = np.zeros((n, n), dtype=float)
    R[p, q] = -1.0
    R[q, p] = 1.0

    lhs = abs(np.trace(G @ (U @ R).T))
    return lhs > eta * np.linalg.norm(G)


def eval_obj(c: float, s: float, App: float, Aqq: float, Apq: float, Aqp: float) -> float:
    return (
        c**3 * (App + Aqq)
        + 3 * c * c * s * (Apq - Aqp)
        + 3 * c * s * s * (Apq + Aqp)
        + s**3 * (Aqq - App)
    )


def solve_cubic(a: float, b: float, c: float, d: float) -> List[float]:
    eps = 1e-14
    if abs(a) < eps:
        if abs(b) < eps:
            if abs(c) < eps:
                return []
            return [-d / c]
        disc = c * c - 4 * b * d
        if disc < 0:
            return []
        root_disc = math.sqrt(disc)
        return [(-c + root_disc) / (2 * b), (-c - root_disc) / (2 * b)]

    p = b / a
    q = c / a
    r = d / a
    p_third = p / 3.0
    Q = (3 * q - p * p) / 9.0
    R = (9 * p * q - 27 * r - 2 * p * p * p) / 54.0
    D = Q * Q * Q + R * R

    if D > 1e-14:
        sqrtD = math.sqrt(D)
        S = math.copysign(abs(R + sqrtD) ** (1 / 3), R + sqrtD)
        T_ = math.copysign(abs(R - sqrtD) ** (1 / 3), R - sqrtD)
        return [S + T_ - p_third]

    denom = math.sqrt(max(1e-300, -Q * Q * Q))
    arg = max(-1.0, min(1.0, R / denom))
    theta = math.acos(arg)
    sqrtQ = math.sqrt(max(0.0, -Q))
    return [
        2 * sqrtQ * math.cos(theta / 3.0) - p_third,
        2 * sqrtQ * math.cos((theta + 2 * math.pi) / 3.0) - p_third,
        2 * sqrtQ * math.cos((theta + 4 * math.pi) / 3.0) - p_third,
    ]


def calculate_rotation_angle(Appp: float, Aqqq: float, Aqpp: float, Apqq: float) -> Tuple[float, float]:
    ca = Aqpp + Apqq
    cb = Appp - Aqqq + 2 * Aqpp - 2 * Apqq
    cc = Appp + Aqqq - 2 * Aqpp - 2 * Apqq
    cd = Apqq - Aqpp
    roots = solve_cubic(ca, cb, cc, cd)

    best_c, best_s = 1.0, 0.0
    max_g = eval_obj(1.0, 0.0, Appp, Aqqq, Aqpp, Apqq)
    for t in roots:
        c_i = 1.0 / math.sqrt(1.0 + t * t)
        s_i = t * c_i
        g = eval_obj(c_i, s_i, Appp, Aqqq, Aqpp, Apqq)
        if g > max_g:
            max_g = g
            best_c, best_s = c_i, s_i
    return best_c, best_s


def embed_givens_rotation(G: np.ndarray, p: int, q: int, c: float, s: float) -> None:
    G[p, p] = c
    G[p, q] = s
    G[q, p] = -s
    G[q, q] = c


def partition_round_robin(N: int) -> List[List[Tuple[int, int]]]:
    n = N if N % 2 == 0 else N + 1
    nr_groups = n - 1

    def wrap(n_val: int, i: int) -> int:
        return ((i - 1) % (n_val - 1)) + 1

    groups: List[List[Tuple[int, int]]] = []
    for i in range(1, nr_groups + 1):
        group: List[Tuple[int, int]] = []
        first = (0, i)
        if first[0] < N and first[1] < N:
            group.append(tuple(sorted(first)))
        for j in range(1, ((nr_groups - 1) // 2) + 1):
            p = wrap(nr_groups + 1, i + j)
            q = wrap(nr_groups + 1, p + (nr_groups - 1) - (2 * j - 1))
            if p < N and q < N:
                group.append(tuple(sorted((p, q))))
        groups.append(group)
    return groups


def diag_norm_sq(T: np.ndarray) -> float:
    n = T.shape[0]
    return float(sum(T[i, i, i] ** 2 for i in range(n)))


def frob_norm_sq(T: np.ndarray) -> float:
    return float(np.sum(T * T))


def offdiag_norm_sq(T: np.ndarray) -> float:
    return frob_norm_sq(T) - diag_norm_sq(T)


def tensor_trace(T: np.ndarray) -> float:
    n = T.shape[0]
    return float(sum(T[i, i, i] for i in range(n)))


def matlab_name(name: str) -> str:
    return name.replace("-", "_")


def print_tensor_matlab(name: str, T: np.ndarray, tol: float = 1e-15) -> None:
    name = matlab_name(name)
    n1, n2, n3 = T.shape
    print(f"{name} = zeros({n1},{n2},{n3});")
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if abs(T[i, j, k]) > tol:
                    print(f"{name}({i+1},{j+1},{k+1}) = {T[i,j,k]:.15e};")


def print_matrix_matlab(name: str, M: np.ndarray, tol: float = 1e-15) -> None:
    name = matlab_name(name)
    n, _ = M.shape
    print(f"{name} = eye({n});")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j] - (1.0 if i == j else 0.0)) > tol:
                print(f"{name}({i+1},{j+1}) = {M[i,j]:.15e};")


def print_vector_matlab(name: str, values: Sequence[float]) -> None:
    name = matlab_name(name)
    arr = np.asarray(values, dtype=float)
    formatted = " ".join(f"{x:.15e}" for x in arr)
    print(f"{name} = [{formatted}];")


def format_groups(groups: List[List[Tuple[int, int]]]) -> str:
    lines = []
    for i, g in enumerate(groups, start=1):
        lines.append(f"Group {i}: " + " ".join(f"({p},{q})" for p, q in g))
    return "\n".join(lines)


def diagonal_tensor_from_tensor(T: np.ndarray) -> np.ndarray:
    D = np.zeros_like(T)
    n = T.shape[0]
    for i in range(n):
        D[i, i, i] = T[i, i, i]
    return D


def reconstruct_from_diagonal_and_factor(D: np.ndarray, U: np.ndarray) -> np.ndarray:
    return tucker_sym(D, U)


def run_symmetric_jacobi(
    T0: np.ndarray,
    max_iters: int,
    pivot_eps: float,
    stop_mode: str,
    tol: float,
    require_both: bool,
    verbose: bool,
) -> RunResult:
    n = T0.shape[0]
    groups = partition_round_robin(n)
    T = T0.copy()
    F = np.eye(n)
    history: List[IterationRecord] = []
    trace_history = [tensor_trace(T)]
    rel_offdiag_history = [math.sqrt(max(0.0, offdiag_norm_sq(T))) / math.sqrt(max(1e-300, frob_norm_sq(T)))]
    eta = 1.0/ (100.0 * n)
    

    for it in range(1, max_iters + 1):
        diag_old = diag_norm_sq(T)
        sweep_skipped = 0
        sweep_total = 0

        for g_idx, group in enumerate(groups, start=1):
            G = np.eye(n)
            active_pairs = 0
            for (p, q) in group:
                Appp = T[p, p, p]
                Aqqq = T[q, q, q]
                Aqpp = T[q, p, p]
                Apqq = T[p, q, q]
                sweep_total += 1
                if not gredcheck_tsym(T, F, p, q, eta):
                    sweep_skipped += 1
                    continue
                c, s = calculate_rotation_angle(Appp, Aqqq, Aqpp, Apqq)
                if abs(s) <= pivot_eps:
                    sweep_skipped += 1
                    continue
                embed_givens_rotation(G, p, q, c, s)
                active_pairs += 1

            T = tucker_sym(T, G)
            F = F @ G.T

            if verbose:
                print(
                    f"iter={it:4d} group={g_idx:3d}/{len(groups)} active_pairs={active_pairs} "
                    f"diag={diag_norm_sq(T):.12e} offdiag={offdiag_norm_sq(T):.12e} trace={tensor_trace(T):.12e}"
                )

        diag_new = diag_norm_sq(T)
        off_new = offdiag_norm_sq(T)
        ratio = off_new / max(diag_new, 1e-300)
        delta = abs(diag_new - diag_old)
        tr_new = tensor_trace(T)
        rel_off_new = math.sqrt(max(0.0, off_new)) / math.sqrt(max(1e-300, frob_norm_sq(T)))
        record = IterationRecord(
            iter=it,
            diag_norm_sq=diag_new,
            offdiag_norm_sq=off_new,
            ratio=ratio,
            delta_diag=delta,
            trace=tr_new,
            rel_offdiag=rel_off_new,
            skipped_pairs=sweep_skipped,
            total_pairs=sweep_total,
        )
        history.append(record)
        trace_history.append(tr_new)
        rel_offdiag_history.append(rel_off_new)

        print(
            f"iter {it:4d}/{max_iters} diag_norm_sq={diag_new:.12e} offdiag_norm_sq={off_new:.12e} "
            f"ratio={ratio:.12e} delta_diag={delta:.12e} trace={tr_new:.12e} "
            f"rel_offdiag={rel_off_new:.12e} skipped={sweep_skipped}/{sweep_total}"
        )

        cond_delta = delta < tol
        cond_ratio = ratio < tol
        if stop_mode == "diag":
            stop = cond_delta
        elif stop_mode == "ratio":
            stop = cond_ratio
        elif stop_mode == "both":
            stop = cond_delta and cond_ratio
        elif stop_mode == "either":
            stop = cond_delta or cond_ratio
        else:
            raise ValueError(f"Unknown stop_mode: {stop_mode}")

        if require_both:
            stop = cond_delta and cond_ratio

        if stop:
            D_final = diagonal_tensor_from_tensor(T)
            T_reconstructed = reconstruct_from_diagonal_and_factor(D_final, F)
            reconstruction_error_abs = float(np.linalg.norm(T_reconstructed-T0))
            reconstruction_error_rel = reconstruction_error_abs / max(float(np.linalg.norm(T)), 1e-300)
            reconstruction_max_abs = float(np.max(np.abs(T_reconstructed - T0)))
            reconstruction_max_rel = reconstruction_max_abs / max(float(np.max(np.abs(T0))), 1e-300)
            return RunResult(
                T_final=T,
                U_final=F,
                history=history,
                trace_history=trace_history,
                rel_offdiag_history=rel_offdiag_history,
                final_record=record,
                T_reconstructed=T_reconstructed,
                reconstruction_error_abs=reconstruction_error_abs,
                reconstruction_error_rel=reconstruction_error_rel,
                reconstruction_max_abs = reconstruction_max_abs,
                reconstruction_max_rel = reconstruction_max_rel,
                
            )
    D_final = diagonal_tensor_from_tensor(T)
    T_reconstructed = reconstruct_from_diagonal_and_factor(D_final, F)
    reconstruction_error_abs = float(np.linalg.norm(T_reconstructed-T0))
    reconstruction_error_rel = reconstruction_error_abs / max(float(np.linalg.norm(T0)), 1e-300)
    reconstruction_max_abs = float(np.max(np.abs(T_reconstructed - T0)))
    reconstruction_max_rel = reconstruction_max_abs / max(float(np.max(np.abs(T0))), 1e-300)
    return RunResult(
        T_final=T,
        U_final=F,
        history=history,
        trace_history=trace_history,
        rel_offdiag_history=rel_offdiag_history,
        final_record=history[-1],
        T_reconstructed=T_reconstructed,
        reconstruction_error_abs=reconstruction_error_abs,
        reconstruction_error_rel=reconstruction_error_rel,
        reconstruction_max_abs = reconstruction_max_abs,
        reconstruction_max_rel = reconstruction_max_rel,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grouped Python oracle for symmetric Jacobi tensor diagonalisation with MATLAB-ready output"
    )
    parser.add_argument("--input-mode", choices=["demo", "fixed-diag", "symdiag"], default="demo")
    parser.add_argument("--n", type=int, default=4, help="Tensor size for --input-mode symdiag")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --input-mode symdiag")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--pivot-eps", type=float, default=1e-12, help="Strict pivot skip threshold")
    parser.add_argument("--tol", type=float, default=1e-12, help="Convergence tolerance")
    parser.add_argument(
        "--stop-mode",
        choices=["diag", "ratio", "both", "either"],
        default="diag",
        help="diag=|diag_new-diag_old|<tol, ratio=offdiag/diag<tol, both=both tests, either=either test",
    )
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="Override stop-mode and require both delta-diag and ratio tests",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-groups", action="store_true")
    parser.add_argument("--print-initial-tensor", action="store_true", help="Print T_initial in MATLAB syntax")
    parser.add_argument("--print-final-tensor", action="store_true", help="Print T_final in MATLAB syntax")
    parser.add_argument("--print-final-factor", action="store_true", help="Print U_final in MATLAB syntax")
    parser.add_argument("--print-reconstructed-tensor", action="store_true", help="Print the tensor reconstructed from diag(T_final) and U_final")
    parser.add_argument(
        "--print-generator",
        action="store_true",
        help="For symdiag/fixed-diag, also print the source diagonal tensor and generator matrix in MATLAB syntax",
    )
    return parser.parse_args(argv)


def build_input_tensor(args: argparse.Namespace) -> TensorGenerationResult:
    if args.input_mode == "demo":
        return build_demo_tensor()
    if args.input_mode == "fixed-diag":
        return build_fixed_diag_tensor()
    if args.input_mode == "symdiag":
        if args.n < 2:
            raise ValueError("--n must be at least 2 for --input-mode symdiag")
        return build_random_orthogonally_diagonalizable_tensor(args.n, args.seed)
    raise ValueError(f"Unsupported input mode: {args.input_mode}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    generated = build_input_tensor(args)
    T0 = generated.T0
    groups = partition_round_robin(T0.shape[0])

    print("%% ============================================")
    print("%% GROUPED JACOBI ORACLE DEBUG OUTPUT")
    print("%% ============================================")
    print(f"%% input_mode = {args.input_mode}")
    print(f"%% n = {T0.shape[0]}")
    if args.seed is not None:
        print(f"%% seed = {args.seed}")
    print()

    if args.print_groups:
        print("%% Round-robin groups (0-based indices)")
        print(format_groups(groups))
        print()

    if args.print_initial_tensor:
        print("=== Initial tensor ===")
        print_tensor_matlab("T_initial", T0)
        print()

    if args.print_generator and generated.source_diag is not None:
        print("=== Source diagonal tensor ===")
        print_tensor_matlab("D_source", generated.source_diag)
        print()
    if args.print_generator and generated.source_factor is not None:
        print("=== Source generator matrix ===")
        print_matrix_matlab("U_source", generated.source_factor)
        print()
    if args.print_generator and generated.source_eigs is not None:
        print("=== Source diagonal entries ===")
        print_vector_matlab("lambda_source", generated.source_eigs)
        print()

    result = run_symmetric_jacobi(
        T0=T0,
        max_iters=args.max_iters,
        pivot_eps=args.pivot_eps,
        stop_mode=args.stop_mode,
        tol=args.tol,
        require_both=args.require_both,
        verbose=args.verbose,
    )

    final = result.final_record
    print("\n=== Final summary ===")
    print(f"iterations_run = {final.iter};")
    print(f"final_diag_norm_sq = {final.diag_norm_sq:.15e};")
    print(f"final_offdiag_norm_sq = {final.offdiag_norm_sq:.15e};")
    print(f"final_ratio = {final.ratio:.15e};")
    print(f"final_delta_diag = {final.delta_diag:.15e};")
    print(f"final_trace = {final.trace:.15e};")
    print(f"final_rel_offdiag = {final.rel_offdiag:.15e};")
    print(f"frob_norm_sq_initial = {frob_norm_sq(T0):.15e};")
    print(f"frob_norm_sq_final = {frob_norm_sq(result.T_final):.15e};")
    print(f"orthogonality_error_U_final = {np.linalg.norm(result.U_final.T @ result.U_final - np.eye(result.U_final.shape[0])):.15e};")
    print(f"reconstruction_error_abs = {result.reconstruction_error_abs:.15e};")
    print(f"reconstruction_error_rel = {result.reconstruction_error_rel:.15e};")
    print(f"reconstruction_max_abs = {result.reconstruction_max_abs:.15e};")
    print(f"reconstruction_max_rel = {result.reconstruction_max_rel:.15e};")
    print_vector_matlab("final_diagonal", np.array([result.T_final[i, i, i] for i in range(result.T_final.shape[0])], dtype=float))
    print_vector_matlab("trace_history", result.trace_history)
    print_vector_matlab("rel_offdiag_history", result.rel_offdiag_history)
    print()

    if args.print_final_tensor:
        print("=== Final tensor ===")
        print_tensor_matlab("T_final", result.T_final)
        print()

    if args.print_final_factor:
        print("=== Final factor matrix ===")
        print_matrix_matlab("U_final", result.U_final)
        print()

    if args.print_reconstructed_tensor:
        print("=== Reconstructed tensor from diag(T_final) and U_final ===")
        print_tensor_matlab("T_reconstructed", result.T_reconstructed)
        print()

    print("=== Final record as Python dict ===")
    print(asdict(final))


if __name__ == "__main__":
    main() 
