#!/usr/bin/env/bash
ORACLE="${1:-./main.py}"
OUTDIR="${2:-.}"
MAX_ITERS="${MAX_ITERS:-1000}"
TOL="${TOL:-1e-12}"
STOP_MODE="${STOP_MODE:-diag}"
PIVOT_EPS="${PIVOT_EPS:-1e-12}"
START_N="${START_N:-3}"
END_N="${END_N:-30}"
SEEDS="${SEEDS:-0 1 2}"
REQUIRE_BOTH="${REQUIRE_BOTH:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "$OUTDIR/logs"
SUMMARY_TSV="$OUTDIR/summary.tsv"
DICT_JSONL="$OUTDIR/final_record_dicts.jsonl"
RUN_CFG="$OUTDIR/run_config.txt"
cat > "$RUN_CFG" <<CFG
ORACLE=$ORACLE
MAX_ITERS=$MAX_ITERS
TOL=$TOL
STOP_MODE=$STOP_MODE
PIVOT_EPS=$PIVOT_EPS
START_N=$START_N
END_N=$END_N
SEEDS=$SEEDS
REQUIRE_BOTH=$REQUIRE_BOTH
EXTRA_ARGS=$EXTRA_ARGS
CFG
printf 'n\tseed\truntime_seconds\truntime_hms\titerations_run\tfinal_diag_norm_sq\tfinal_offdiag_norm_sq\tfinal_ratio\tfinal_delta_diag\tfinal_trace\tfinal_rel_offdiag\treconstruction_error_abs\treconstruction_error_rel\treconstruction_max_abs\treconstruction_max_rel\torthogonality_error_U_final\tlog_file\n' > "$SUMMARY_TSV"
: > "$DICT_JSONL"
if [[ ! -f "$ORACLE" ]]; then
  echo "Oracle file not found: $ORACLE" >&2
  exit 1
fi
for n in $(seq "$START_N" "$END_N")
do
  for seed in $SEEDS
  do
    LOG_FILE="$OUTDIR/logs/n_${n}_seed_${seed}.log"
    echo "Running n=$n seed=$seed ..."
    cmd=(python3 "$ORACLE" \
      --input-mode symdiag \
      --n "$n" \
      --seed "$seed" \
      --max-iters "$MAX_ITERS" \
      --pivot-eps "$PIVOT_EPS" \
      --tol "$TOL" \
      --stop-mode "$STOP_MODE")
    if [[ "$REQUIRE_BOTH" == "1" ]]; then
      cmd+=(--require-both)
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
      # shellcheck disable=SC2206
      extra=( $EXTRA_ARGS )
      cmd+=("${extra[@]}")
    fi
    start_ts=$(date +%s)
    "${cmd[@]}" > "$LOG_FILE"
    end_ts=$(date +%s)
    runtime_seconds=$((end_ts - start_ts))
    runtime_hms=$(printf '%02d:%02d:%02d' $((runtime_seconds/3600)) $(((runtime_seconds%3600)/60)) $((runtime_seconds%60)))
    python3 - "$LOG_FILE" "$n" "$seed" "$runtime_seconds" "$runtime_hms" "$SUMMARY_TSV" "$DICT_JSONL" <<'PY'
import ast
import json
import pathlib
import re
import sys

log_file, n, seed, runtime_seconds, runtime_hms, summary_tsv, dict_jsonl = sys.argv[1:]
text = pathlib.Path(log_file).read_text()

keys = [
    "iterations_run",
    "final_diag_norm_sq",
    "final_offdiag_norm_sq",
    "final_ratio",
    "final_delta_diag",
    "final_trace",
    "final_rel_offdiag",
    "orthogonality_error_U_final",
    "reconstruction_error_abs",
    "reconstruction_error_rel",
    "reconstruction_max_abs",
    "reconstruction_max_rel",
]

vals = {}
for key in keys:
    m = re.search(rf"^{re.escape(key)}\s*=\s*([^;\n]+);", text, re.M)
    vals[key] = m.group(1).strip() if m else "nan"

record = None
lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
for line in reversed(lines):
    if line.startswith("{") and line.endswith("}"):
        record = ast.literal_eval(line)
        break

entry = {
    "n": int(n),
    "seed": int(seed),
    "runtime_seconds": int(runtime_seconds),
    "runtime_hms": runtime_hms,
    "log_file": log_file,
    "metrics": vals,
    "final_record": record,
}

with open(dict_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")

row = [
    str(n),
    str(seed),
    str(runtime_seconds),
    runtime_hms,
    vals["iterations_run"],
    vals["final_diag_norm_sq"],
    vals["final_offdiag_norm_sq"],
    vals["final_ratio"],
    vals["final_delta_diag"],
    vals["final_trace"],
    vals["final_rel_offdiag"],
    vals["reconstruction_error_abs"],
    vals["reconstruction_error_rel"],
    vals["reconstruction_max_abs"],
    vals["reconstruction_max_rel"],
    vals["orthogonality_error_U_final"],
    log_file,
]

with open(summary_tsv, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY
  done
done

echo
echo "Results written to: $OUTDIR"
echo "Summary table: $SUMMARY_TSV"
echo "Dict JSONL:    $DICT_JSONL"
echo

echo "===== Raw summary table ====="
column -t -s $'\t' "$SUMMARY_TSV"
echo

echo "===== Aggregated by n ====="
python3 - "$SUMMARY_TSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

summary_tsv = sys.argv[1]
rows = []
with open(summary_tsv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

by_n = defaultdict(list)
for row in rows:
    by_n[int(row['n'])].append(row)

headers = [
    'n',
    'runs',
    'avg_runtime_s',
    'max_runtime_s',
    'avg_final_rel_offdiag',
    'max_final_rel_offdiag',
    'avg_recon_abs',
    'max_recon_abs',
    'avg_recon_max_abs',
    'max_recon_max_abs',
]
print("\t".join(headers))
for n in sorted(by_n):
    grp = by_n[n]
    runt = [float(r['runtime_seconds']) for r in grp]
    frel = [float(r['final_rel_offdiag']) for r in grp]
    rabs = [float(r['reconstruction_error_abs']) for r in grp]
    rmax = [float(r['reconstruction_max_abs']) for r in grp]
    print("\t".join([
        str(n),
        str(len(grp)),
        f"{statistics.fmean(runt):.2f}",
        f"{max(runt):.2f}",
        f"{statistics.fmean(frel):.6e}",
        f"{max(frel):.6e}",
        f"{statistics.fmean(rabs):.6e}",
        f"{max(rabs):.6e}",
        f"{statistics.fmean(rmax):.6e}",
        f"{max(rmax):.6e}",
    ]))
PY 

