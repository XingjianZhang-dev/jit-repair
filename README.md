# Just-in-Time Repair: Spending Test-Time Compute Where LLM-Generated Programs and SQL Fail

Code for the paper of the same title by Xingjian Zhang.

JIT-Repair makes three decisions for every output a language model writes:

- **gate**: flag the first output when it fails a deterministic check, that is, when it does not parse, validate or
  execute, or when a question–SQL consistency rule (the critic) fires;
- **spend**: improve only the flagged outputs, with one revision call, with a vote over the execution results of fresh
  samples (self-consistency restricted to flagged outputs), or with both;
- **selector**: replace the first output only when the check prefers the candidate, so checker validity never
  decreases for any spend.

This repository contains the method, every experiment driver, the statistics, and the code that produces each table
and figure of the paper and its supplementary material. Data are not redistributed; `DATA.md` describes every input,
how to obtain or rebuild it, and the sha256 of the files used in the paper.

## Layout

```
jit_repair/          method core: SQL prompts and read-only execution checker, question–SQL critic, selector,
                     ProofWriter logic-program prompts, validator and executor, input preparation, paths
experiments/         generation and scoring drivers (vLLM) and the per-experiment analyses
experiments/archive/ analyses of the earlier runs on the second hardware stack (Supplement S4.6)
analysis/            allocation identities and checks, fix–break ledger, paper tables, figure data
figures/             figure rendering (matplotlib) and Fig. 1 (TikZ)
scripts/             environment setup, data and model download, end-to-end run scripts
tests/               unit tests and replay checks
```

Experiment identifiers follow the list of pre-registered predictions in Supplement S5.

## Installation

Python 3.10 or later.

```sh
python -m pip install -e .                       # method core and all CPU analyses
python -m pip install -r requirements-gpu.txt    # generation: vLLM 0.11.0, PyTorch 2.8.0, Transformers 4.57.6
```

The paper's generation ran on one NVIDIA A100-SXM4-80GB with CUDA 12.8 and Python 3.12 (`scripts/setup_env.sh`).

## Directory layout and environment variables

`jit_repair/paths.py` defines every location. Generation uses `JIT_WORK` (default `/workspace`) with `inbox/`
(prepared inputs), `outbox/results/` (outputs), `data/` (benchmarks and evaluators) and `models/` (weights). The
analyses read the per-question records from `JIT_RECORDS` (default `records/`; a copy of `outbox/results` as
`records/results`, plus `records/tables`, `records/inputs` and `records/LEDGER.csv`) and the second-stack records from
`JIT_ARCHIVE` (default `records/archive`). Tables, figure data and figures are written to `JIT_OUTPUTS` (default
`outputs/`).

## Reproducing the experiments (GPU)

1. `scripts/setup_env.sh`, then `scripts/download_data.sh` and `scripts/download_models.sh`.
2. Prepare the inputs in `$JIT_WORK/inbox` as described in `DATA.md` and check their sha256.
3. `scripts/run_experiments.sh` runs every generation and scoring step in the order used for the paper. Each driver
   writes its records as it goes and resumes from them.

References are never used for generation, checking, gating or selection; the scoring steps read them separately.

## Reproducing the tables and figures (CPU)

Copy `$JIT_WORK/outbox/results` to `records/results`, place the prepared Dr.Spider subset at
`records/inputs/drspider/safe_subset.jsonl`, and run

```sh
scripts/run_analysis.sh
```

It recomputes the experiment tables in `records/tables` and writes the paper's tables (`outputs/tables`,
`outputs/supp_tables`), the figure data (`outputs/figure_data`) and the figures (`outputs/figures`). Fig. 1 is
`figures/fig1_method.tex` (`pdflatex fig1_method.tex`).

## Where each result comes from

| Paper | Experiment | Code |
|---|---|---|
| Sec. 6.1, Fig. 2, Table 2, Supplement S3 | X7, allocation on held-out BIRD | `prepare_heldout_bird.py`, `x7_birddev.py`, `lp_score.py`, `x7_alloc.py`, `x7_confirm.py` |
| Sec. 6.2, Fig. 3, Fig. 4, Table S9 | fix–break ledger | `analysis/allocation_lens.py`, `analysis/allocation_theory.py` |
| Sec. 6.3, Fig. 5, Supplement S4.2 | X3b and X3c, Dr.Spider | `x3b_drspider.py`, `x3b_peritem.py` |
| Sec. 6.3, Supplement S4.1 | C2, accuracy–calls frontier | `generate_c1_c2.py`, `c2_analyze.py`, `c2_report.py` |
| Sec. 6.4, Table 3, Supplement S4.3 | X3, revising the vote winner; spend choice | `x3_combo.py`, `x3_score.py`, `x7_boundary.py`, `analysis/allocation_lens.py` |
| Sec. 6.5, Fig. 6, Table 4, Supplement S4.5 | C1, programs against reasoning | `generate_c1_c2.py`, `llama_cot_second_attempt.py`, `c1_analyze.py`, `c1_report.py`, `c1_llama_second_attempt_score.py` |
| Sec. 6.6, Supplement S4.7 | G2 and G3, execution-only gate and Proposition 1 audit | `alloc_controls.py`, `prop1_check.py` |
| Sec. 4, Supplement S2 | numeric checks of the propositions and corollaries; allocation curves | `analysis/allocation_theory.py`, `analysis/allocation_lens.py` |
| Supplement S4.6 | archive results (second stack) | `experiments/archive/`, `t9_testsuite.py` |
| Supplement S5 | X1 and X2 | `x1_residuals.py`, `x2_errortype.py` |
| Supplement S6 | G1, output identity across stacks | `anchor_spider.py` |
| Tables 2–4, Tables S1–S17 | LaTeX tables | `analysis/make_tables.py`, `analysis/make_supp_tables.py` |
| Figs. 2–6, Fig. S1 | figures | `analysis/figure_data.py`, `figures/make_figures.py` |

The earlier runs on the second stack (Hugging Face Transformers on an NVIDIA RTX PRO 6000 Blackwell) used the
Dr.Spider pipeline in `jit_repair/`: `prepare_drspider_oneuse`, `run_drspider_oneuse` (initial generation, generic
revision and the two Self-Refine stages), `build_drspider_oneuse_arms`, `evaluate_drspider_execution`,
`analyze_drspider_oneuse` and `evaluate_drspider_oneuse_gate`; each module documents its arguments (`--help`).

## Tests

```sh
python tests/test_core.py            # executor, SQL checker, critic, selector
python tests/test_sql_tools.py       # SQL helpers on a temporary database
python tests/test_pw_extractor.py    # answer extractor for free-form reasoning
python tests/test_paired_stats.py    # paired statistics against a stored result (needs the archive records)
python tests/test_pw_frozen_replay.py <depth-5 set> <four-label set>   # replay of stored ProofWriter outputs (archive records)
```

## Citation

If you use this code, please cite the paper; bibliographic details will be added on publication.
