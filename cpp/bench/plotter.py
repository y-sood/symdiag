#!/usr/bin/env python3
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def line_plot(df, x, y, out, hue=None, logy=False, title=None):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    if hue and hue in df.columns:
        for key, g in df.groupby(hue):
            g = g.sort_values(x)
            plt.plot(g[x], g[y], marker="o", label=str(key))
        plt.legend(title=hue)
    else:
        d = df.sort_values(x)
        plt.plot(d[x], d[y], marker="o")
    if logy:
        plt.yscale("log")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x}")
    savefig(out)

def scatter_plot(df, x, y, out, hue=None, logx=False, logy=False, title=None):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    if hue and hue in df.columns:
        for key, g in df.groupby(hue):
            plt.scatter(g[x], g[y], label=str(key), alpha=0.8)
        plt.legend(title=hue)
    else:
        plt.scatter(df[x], df[y], alpha=0.8)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x}")
    savefig(out)

def convergence_plot(sweeps, metric, out, logy=True):
    if sweeps.empty or metric not in sweeps.columns:
        return
    plt.figure(figsize=(8, 5))
    for run_id, g in sweeps.groupby("run_id"):
        g = g.sort_values("iter")
        plt.plot(g["iter"], g[metric], marker="o", alpha=0.8, label=run_id)
    if logy:
        plt.yscale("log")
    plt.xlabel("iter")
    plt.ylabel(metric)
    plt.title(f"{metric} vs iteration")
    if sweeps["run_id"].nunique() <= 12:
        plt.legend(fontsize=8)
    savefig(out)

def grouped_mean_plot(df, group_col, metric, out, logy=False):
    if df.empty or group_col not in df.columns or metric not in df.columns:
        return
    d = df.groupby(group_col, dropna=True)[metric].mean().reset_index().sort_values(group_col)
    plt.figure(figsize=(8, 5))
    plt.plot(d[group_col], d[metric], marker="o")
    if logy:
        plt.yscale("log")
    plt.xlabel(group_col)
    plt.ylabel(f"mean {metric}")
    plt.title(f"mean {metric} vs {group_col}")
    savefig(out)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_jacobi_results.py <benchmark_output_dir>")
        sys.exit(1)

    out_dir = sys.argv[1]
    summary_path = os.path.join(out_dir, "summary.csv")
    sweeps_path = os.path.join(out_dir, "sweeps.csv")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary = pd.read_csv(summary_path)
    sweeps = pd.read_csv(sweeps_path)

    summary = ensure_numeric(summary, [
        "n", "block_size", "max_iters", "seed", "tol_delta", "tol_ratio", "min_iters",
        "elapsed_s", "user_s", "sys_s", "maxrss_kb", "exit_code", "converged", "stop_iter",
        "diag_norm_sq", "offdiag_norm_sq", "frob_norm_sq", "ratio", "rel_offdiag", "trace",
        "avg_diag_abs", "avg_offdiag_abs", "max_offdiag_abs",
        "orthogonality_error_U_final", "reconstruction_error_abs", "reconstruction_error_rel",
        "reconstruction_max_abs", "reconstruction_max_rel"
    ])

    sweeps = ensure_numeric(sweeps, [
        "iter", "diag_norm_sq", "offdiag_norm_sq", "ratio",
        "rel_offdiag", "delta", "trace", "stop"
    ])

    summary_ok = summary[summary["exit_code"] == 0].copy()

    line_plot(summary_ok, "n", "elapsed_s", os.path.join(plots_dir, "runtime_vs_n.png"),
              hue="stop_mode", title="Runtime vs n")
    line_plot(summary_ok, "n", "maxrss_kb", os.path.join(plots_dir, "memory_vs_n.png"),
              hue="stop_mode", title="Peak RSS vs n")
    line_plot(summary_ok, "block_size", "elapsed_s", os.path.join(plots_dir, "runtime_vs_block_size.png"),
              hue="stop_mode", title="Runtime vs block size")
    line_plot(summary_ok, "block_size", "maxrss_kb", os.path.join(plots_dir, "memory_vs_block_size.png"),
              hue="stop_mode", title="Peak RSS vs block size")

    scatter_plot(summary_ok, "elapsed_s", "reconstruction_error_rel",
                 os.path.join(plots_dir, "recon_error_vs_runtime.png"),
                 hue="stop_mode", logy=True, title="Relative reconstruction error vs runtime")
    scatter_plot(summary_ok, "elapsed_s", "orthogonality_error_U_final",
                 os.path.join(plots_dir, "orthogonality_vs_runtime.png"),
                 hue="stop_mode", logy=True, title="Orthogonality error vs runtime")
    scatter_plot(summary_ok, "reconstruction_error_rel", "orthogonality_error_U_final",
                 os.path.join(plots_dir, "orthogonality_vs_reconstruction_error.png"),
                 hue="stop_mode", logx=True, logy=True,
                 title="Orthogonality error vs reconstruction error")
    scatter_plot(summary_ok, "elapsed_s", "maxrss_kb",
                 os.path.join(plots_dir, "memory_vs_runtime.png"),
                 hue="stop_mode", title="Peak RSS vs runtime")

    grouped_mean_plot(summary_ok, "n", "elapsed_s", os.path.join(plots_dir, "mean_runtime_vs_n.png"))
    grouped_mean_plot(summary_ok, "n", "maxrss_kb", os.path.join(plots_dir, "mean_memory_vs_n.png"))
    grouped_mean_plot(summary_ok, "n", "reconstruction_error_rel",
                      os.path.join(plots_dir, "mean_recon_error_vs_n.png"), logy=True)

    convergence_plot(sweeps, "rel_offdiag", os.path.join(plots_dir, "convergence_rel_offdiag.png"), logy=True)
    convergence_plot(sweeps, "ratio", os.path.join(plots_dir, "convergence_ratio.png"), logy=True)
    convergence_plot(sweeps, "offdiag_norm_sq", os.path.join(plots_dir, "convergence_offdiag_norm_sq.png"), logy=True)
    convergence_plot(sweeps, "diag_norm_sq", os.path.join(plots_dir, "convergence_diag_norm_sq.png"), logy=False)
    convergence_plot(sweeps, "delta", os.path.join(plots_dir, "convergence_delta.png"), logy=True)
    convergence_plot(sweeps, "trace", os.path.join(plots_dir, "convergence_trace.png"), logy=False)

    # Per-run convergence overlays for a few representative runs
    sample_runs = summary_ok["run_id"].dropna().astype(str).head(6).tolist()
    sample_sweeps = sweeps[sweeps["run_id"].isin(sample_runs)].copy()
    if not sample_sweeps.empty:
        convergence_plot(sample_sweeps, "rel_offdiag",
                         os.path.join(plots_dir, "sample_rel_offdiag_runs.png"), logy=True)
        convergence_plot(sample_sweeps, "ratio",
                         os.path.join(plots_dir, "sample_ratio_runs.png"), logy=True)

    print(f"Plots written to: {plots_dir}")

if __name__ == "__main__":
    main()