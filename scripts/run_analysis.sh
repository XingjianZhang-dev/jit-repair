#!/usr/bin/env bash
# Analyses, tables and figures from the per-question records (CPU). Records: $JIT_RECORDS (default records/).
# Experiment tables go to records/tables; paper tables, figure data and figures go to $JIT_OUTPUTS (default outputs/).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REC="${JIT_RECORDS:-$REPO/records}"
mkdir -p "$REC/tables" "$REC/figures"
export MPLBACKEND=Agg

cd "$REPO/experiments"
python x7_alloc.py discovery
python x7_alloc.py confirmation
python x7_confirm.py selftest
python x7_confirm.py
python x7_boundary.py
python alloc_controls.py
python prop1_check.py
python c2_report.py
python c1_analyze.py --root "$REC/results/pw" --out "$REC/tables/C1_T1_analysis.json"
python c1_report.py "$REC/tables/C1_T1_analysis.json"
python c1_llama_second_attempt_score.py
python x1_residuals.py
python x2_errortype.py "$REC/results/pw" "$REC/tables/X2_error_type.json"

cd "$REPO/experiments/archive"
python t4_global_holm.py
python t3_like_for_like_cost.py
python t6_ablation.py

cd "$REPO/analysis"
python allocation_theory.py
python allocation_lens.py
python make_tables.py
python make_supp_tables.py
python figure_data.py

cd "$REPO/figures"
python make_figures.py
