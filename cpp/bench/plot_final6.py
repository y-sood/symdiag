#!/usr/bin/env python3
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    PALETTE = sns.color_palette("deep", 8)
except Exception:
    PALETTE = plt.cm.tab10.colors

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 240,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "semibold",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
})


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_summary(run_dir):
    path = os.path.join(run_dir, "summary.csv")
    df = pd.read_csv(path)
    numeric_cols = [
        "n", "block_size", "max_iters", "seed", "tol_delta", "tol_ratio", "min_iters",
        "elapsed_s", "user_s", "sys_s", "maxrss_kb", "exit_code", "converged", "stop_iter",
        "diag_norm_sq", "offdiag_norm_sq", "frob_norm_sq", "ratio", "rel_offdiag", "trace",
        "avg_diag_abs", "avg_offdiag_abs", "max_offdiag_abs",
        "orthogonality_error_U_final", "reconstruction_error_abs", "reconstruction_error_rel",
        "reconstruction_max_abs", "reconstruction_max_rel"
    ]
    df = ensure_numeric(df, numeric_cols)
    if "exit_code" in df.columns:
        df = df[df["exit_code"] == 0].copy()
    return df


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def fmt_seconds(x, _):
    return f"{x:.0f} s" if abs(x) >= 1 else f"{x:.2f} s"


def fmt_kb(x, _):
    if x >= 1024 * 1024:
        return f"{x / (1024*1024):.1f}"
    if x >= 1024:
        return f"{x / 1024:.0f}"
    return f"{x:.0f} KB"


def int_x(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def summarise(df, xcol, ycol):
    d = (df.groupby(xcol, dropna=True)[ycol]
           .agg(mean="mean", std="std", q10=lambda s: s.quantile(0.10), q90=lambda s: s.quantile(0.90), count="count")
           .reset_index()
           .sort_values(xcol))
    d["lo"] = np.where(d["count"] > 1, d["q10"], d["mean"])
    d["hi"] = np.where(d["count"] > 1, d["q90"], d["mean"])
    d["std"] = d["std"].fillna(0.0)
    return d


def mean_line_plot(df, xcol, ycol, out, title, xlabel, ylabel, yfmt=None, logy=False, annotate_best=False):
    if df.empty or xcol not in df.columns or ycol not in df.columns:
        return
    d = summarise(df, xcol, ycol)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    color = PALETTE[0]
    ax.plot(d[xcol], d["mean"], color=color, marker="o", linewidth=2.6, markersize=6)
    if annotate_best and not d.empty:
        best = d.loc[d["mean"].idxmin()]
        ax.scatter([best[xcol]], [best["mean"]], s=80, color="#c44e52", zorder=4)
        label_x = int(best[xcol]) if float(best[xcol]).is_integer() else best[xcol]
        ax.annotate(f"best = {label_x}", xy=(best[xcol], best["mean"]), xytext=(8, -14),
                    textcoords="offset points", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if yfmt is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    int_x(ax)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, which="major", axis="both", alpha=0.32, linestyle="--")
    ax.grid(True, which="minor", axis="y", alpha=0.16, linestyle=":")
    savefig(out)


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 plot_final6.py <size_sweep_dir> <block_sweep_dir> [plots_out_dir]")
        sys.exit(1)

    size_dir = sys.argv[1]
    block_dir = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) == 4 else os.path.join(block_dir, "plots_final6")
    os.makedirs(out_dir, exist_ok=True)

    size_df = read_summary(size_dir)
    block_df = read_summary(block_dir)

    if size_df.empty:
        raise RuntimeError(f"No successful rows found in {size_dir}/summary.csv")
    if block_df.empty:
        raise RuntimeError(f"No successful rows found in {block_dir}/summary.csv")

    size_bs_vals = sorted(size_df["block_size"].dropna().unique().tolist()) if "block_size" in size_df.columns else []
    if len(size_bs_vals) != 1:
        print(f"Warning: size sweep has block sizes {size_bs_vals}; plotting all rows together.")
    else:
        fixed_bs = int(size_bs_vals[0]) if float(size_bs_vals[0]).is_integer() else size_bs_vals[0]

    block_n_vals = sorted(block_df["n"].dropna().unique().tolist()) if "n" in block_df.columns else []
    if len(block_n_vals) != 1:
        print(f"Warning: block sweep has n values {block_n_vals}; plotting all rows together.")
    else:
        fixed_n = int(block_n_vals[0]) if float(block_n_vals[0]).is_integer() else block_n_vals[0]

    title_suffix_size = f" (fixed block size = {fixed_bs})" if len(size_bs_vals) == 1 else ""
    title_suffix_block = f" (fixed n = {fixed_n})" if len(block_n_vals) == 1 else ""

    mean_line_plot(
        size_df, "n", "elapsed_s", os.path.join(out_dir, "01_runtime_vs_problem_size.png"),
        title=f"Runtime vs problem size{title_suffix_size}",
        xlabel="Number of elements per dimension", ylabel="Runtime (s)", yfmt=fmt_seconds, logy=False
    )

    mean_line_plot(
        size_df, "n", "maxrss_kb", os.path.join(out_dir, "02_peak_memory_vs_problem_size.png"),
        title=f"Peak memory vs problem size{title_suffix_size}",
        xlabel="Number of elements per dimension", ylabel="Peak RSS (MB)", yfmt=fmt_kb, logy=False
    )

    mean_line_plot(
        size_df, "n", "stop_iter", os.path.join(out_dir, "03_converged_iteration_vs_problem_size.png"),
        title=f"Converged iteration vs problem size{title_suffix_size}",
        xlabel="Number of elements per dimension", ylabel="Iteration of Convergence", yfmt=None, logy=False
    )

    mean_line_plot(
        block_df, "block_size", "elapsed_s", os.path.join(out_dir, "04_runtime_vs_block_size.png"),
        title=f"Runtime vs block size{title_suffix_block}",
        xlabel="Block size", ylabel="Runtime (s)", yfmt=fmt_seconds, logy=False, annotate_best=True
    )

    ratio_metric = "ratio" if "ratio" in block_df.columns else "rel_offdiag"
    ratio_label = "Final off/diag ratio" if ratio_metric == "ratio" else "Final relative off-diagonal norm"
    mean_line_plot(
        block_df, "block_size", ratio_metric, os.path.join(out_dir, "05_final_ratio_vs_block_size.png"),
        title=f"{ratio_label} vs block size{title_suffix_block}",
        xlabel="Block size", ylabel=ratio_label, yfmt=None, logy=True
    )

    mean_line_plot(
        block_df, "block_size", "maxrss_kb", os.path.join(out_dir, "06_peak_memory_vs_block_size.png"),
        title=f"Peak memory vs block size{title_suffix_block}",
        xlabel="Block size", ylabel="Peak RSS (MB)", yfmt=fmt_kb, logy=False
    )

    print(f"Wrote 6 plots to: {out_dir}")


if __name__ == "__main__":
    main()