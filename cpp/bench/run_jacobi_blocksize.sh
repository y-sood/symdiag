#!/usr/bin/env bash
set -euo pipefail

PROG="${1:-../main}"
OUT_DIR="${2:-bench_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/time"

SUMMARY_CSV="$OUT_DIR/summary.csv"
SWEEPS_CSV="$OUT_DIR/sweeps.csv"

echo "run_id,n,block_size,max_iters,seed,tol_delta,tol_ratio,min_iters,stop_mode,elapsed_s,user_s,sys_s,cpu_percent,maxrss_kb,exit_code,converged,stop_iter,diag_norm_sq,offdiag_norm_sq,frob_norm_sq,ratio,rel_offdiag,trace,avg_diag_abs,avg_offdiag_abs,max_offdiag_abs,orthogonality_error_U_final,reconstruction_error_abs,reconstruction_error_rel,reconstruction_max_abs,reconstruction_max_rel" > "$SUMMARY_CSV"

echo "run_id,iter,diag_norm_sq,offdiag_norm_sq,ratio,rel_offdiag,delta,trace,stop" > "$SWEEPS_CSV"

get_metric() {
    local key="$1"
    local file="$2"
    awk -F' = ' -v key="$key" '
        $1 ~ ("^" key "[[:space:]]*$") {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            print $2
            exit
        }
    ' "$file"
}

get_time_field() {
    local key="$1"
    local file="$2"
    awk -F'=' -v key="$key" '$1==key {print $2; exit}' "$file"
}

parse_sweeps() {
    local run_id="$1"
    local log_file="$2"

    awk -v run_id="$run_id" '
    /^SWEEP iter=/ {
        iter=diag=off=ratio=rel=delta=trace=stop=""
        for (i=1; i<=NF; i++) {
            split($i, a, "=")
            if (a[1]=="iter") iter=a[2]
            else if (a[1]=="diag_norm_sq") diag=a[2]
            else if (a[1]=="offdiag_norm_sq") off=a[2]
            else if (a[1]=="ratio") ratio=a[2]
            else if (a[1]=="rel_offdiag") rel=a[2]
            else if (a[1]=="delta") delta=a[2]
            else if (a[1]=="trace") trace=a[2]
            else if (a[1]=="stop") stop=a[2]
        }
        print run_id "," iter "," diag "," off "," ratio "," rel "," delta "," trace "," stop
    }' "$log_file" >> "$SWEEPS_CSV"
}

run_case() {
    local n="$1"
    local bs="$2"
    local max_iters="$3"
    local seed="$4"
    local tol_delta="$5"
    local tol_ratio="$6"
    local min_iters="$7"
    local stop_mode="$8"

    local run_id="n${n}_bs${bs}_mi${max_iters}_seed${seed}_td${tol_delta}_tr${tol_ratio}_min${min_iters}_mode${stop_mode}"
    local log_file="$OUT_DIR/logs/${run_id}.log"
    local time_file="$OUT_DIR/time/${run_id}.time"

    echo "Running $run_id"

    /usr/bin/time -f "elapsed_s=%e
user_s=%U
sys_s=%S
cpu_percent=%P
maxrss_kb=%M
exit_code=%x" \
        -o "$time_file" \
        "$PROG" \
        --n "$n" \
        --block-size "$bs" \
        --max-iters "$max_iters" \
        --seed "$seed" \
        --tol-delta "$tol_delta" \
        --tol-ratio "$tol_ratio" \
        --min-iters "$min_iters" \
        --stop-mode "$stop_mode" \
        > "$log_file" 2>&1

    parse_sweeps "$run_id" "$log_file"

    local elapsed_s user_s sys_s cpu_percent maxrss_kb exit_code
    elapsed_s="$(get_time_field elapsed_s "$time_file")"
    user_s="$(get_time_field user_s "$time_file")"
    sys_s="$(get_time_field sys_s "$time_file")"
    cpu_percent="$(get_time_field cpu_percent "$time_file")"
    maxrss_kb="$(get_time_field maxrss_kb "$time_file")"
    exit_code="$(get_time_field exit_code "$time_file")"

    local converged stop_iter
    converged="$(awk -F'converged=' '/Jacobi finished after/ {print $2; exit}' "$log_file")"
    stop_iter="$(awk '/Converged at iter/ {gsub(":", "", $4); print $4; exit}' "$log_file")"

    local diag_norm_sq offdiag_norm_sq frob_norm_sq ratio rel_offdiag trace
    local avg_diag_abs avg_offdiag_abs max_offdiag_abs
    local orthogonality_error_U_final reconstruction_error_abs reconstruction_error_rel
    local reconstruction_max_abs reconstruction_max_rel

    diag_norm_sq="$(get_metric diag_norm_sq "$log_file")"
    offdiag_norm_sq="$(get_metric offdiag_norm_sq "$log_file")"
    frob_norm_sq="$(get_metric frob_norm_sq "$log_file")"
    ratio="$(get_metric ratio "$log_file")"
    rel_offdiag="$(get_metric rel_offdiag "$log_file")"
    trace="$(get_metric trace "$log_file")"
    avg_diag_abs="$(get_metric avg_diag_abs "$log_file")"
    avg_offdiag_abs="$(get_metric avg_offdiag_abs "$log_file")"
    max_offdiag_abs="$(get_metric max_offdiag_abs "$log_file")"
    orthogonality_error_U_final="$(get_metric orthogonality_error_U_final "$log_file")"
    reconstruction_error_abs="$(get_metric reconstruction_error_abs "$log_file")"
    reconstruction_error_rel="$(get_metric reconstruction_error_rel "$log_file")"
    reconstruction_max_abs="$(get_metric reconstruction_max_abs "$log_file")"
    reconstruction_max_rel="$(get_metric reconstruction_max_rel "$log_file")"

    echo "$run_id,$n,$bs,$max_iters,$seed,$tol_delta,$tol_ratio,$min_iters,$stop_mode,$elapsed_s,$user_s,$sys_s,$cpu_percent,$maxrss_kb,$exit_code,$converged,$stop_iter,$diag_norm_sq,$offdiag_norm_sq,$frob_norm_sq,$ratio,$rel_offdiag,$trace,$avg_diag_abs,$avg_offdiag_abs,$max_offdiag_abs,$orthogonality_error_U_final,$reconstruction_error_abs,$reconstruction_error_rel,$reconstruction_max_abs,$reconstruction_max_rel" >> "$SUMMARY_CSV"
}

# ---- parameter grid ----
NS=(48)
BLOCK_SIZES=(2 3 4 6 8 12 16 24 48)
MAX_ITERS=(100)
SEEDS=(42 0 100)
TOL_DELTAS=(1e-10)
TOL_RATIOS=(1e-6)
MIN_ITERS=(2)
STOP_MODES=(either)

for n in "${NS[@]}"; do
  for bs in "${BLOCK_SIZES[@]}"; do
    for mi in "${MAX_ITERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for td in "${TOL_DELTAS[@]}"; do
          for tr in "${TOL_RATIOS[@]}"; do
            for mn in "${MIN_ITERS[@]}"; do
              for sm in "${STOP_MODES[@]}"; do
                run_case "$n" "$bs" "$mi" "$seed" "$td" "$tr" "$mn" "$sm"
              done
            done
          done
        done
      done
    done
  done
done

echo ""
echo "Done."
echo "Summary: $SUMMARY_CSV"
echo "Sweeps : $SWEEPS_CSV"
echo "Logs   : $OUT_DIR/logs/"