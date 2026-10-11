#!/usr/bin/env bash
# Every generation and scoring step of the paper on one GPU, in the order used for the paper.
# Inputs: $JIT_WORK/inbox and $JIT_WORK/data (DATA.md). Outputs: $JIT_WORK/outbox/results. Each driver resumes.
set -euo pipefail
WORK="${JIT_WORK:-/workspace}"
cd "$(dirname "$0")/../experiments"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# C1 and C2: ProofWriter arms (direct, step-by-step, thinking mode, logic programs with revision) and the
# Spider / BIRD Mini-Dev frontier arms (greedy, JIT-Repair with the revise spend, 8 samples per question)
for spec in "llama31_8b sql,pw 12288" "mistral7b sql,pw 12288" "qwen3_14b pw 18432" "qwen3_8b pw 18432"; do
  set -- $spec
  python generate_c1_c2.py --model-tag "$1" --tasks "$2" --max-model-len "$3"
done
python llama_cot_second_attempt.py        # C1: declared second attempt for Llama-3.1-8B step-by-step outputs
for bench in spider bird; do
  for model in llama31_8b mistral7b; do python c2_analyze.py --bench $bench --model $model --workers 40; done
done

# X3, discovery split: self-consistency, then gating and revising the winner
python x3_combo.py --model llama31_8b --benches spider --split discovery --Ns 3,5
python x3_score.py --split discovery

# X3b: Dr.Spider subset, gated voting against revision and self-consistency
for model in llama31_8b mistral7b; do python x3b_drspider.py --phase gen --model $model; done
python x3b_drspider.py --phase analyze

# X3, confirmation split
for model in llama31_8b mistral7b; do
  python x3_combo.py --model $model --benches spider,bird --split confirmation --Ns 3
done
python x3_score.py --split confirmation

# X3c: Qwen3-8B on the Dr.Spider subset
python x3b_drspider.py --phase gen --model qwen3_8b
python x3b_drspider.py --phase analyze --models qwen3_8b --out-name x3c_qwen3_8b_summary.json
python x3b_peritem.py

# Token confidence of the greedy outputs for the allocation comparators (cells analysed before X7)
python lp_score.py --model llama31_8b --cells spider,bird,drspider
python lp_score.py --model mistral7b --cells spider,bird,drspider
python lp_score.py --model qwen3_8b --cells drspider

# X7: pre-registered allocation test on held-out BIRD
python prepare_heldout_bird.py
for model in llama31_8b mistral7b qwen3_8b qwen3_14b; do
  python x7_birddev.py --phase gen --model $model
  python lp_score.py --model $model --cells birddev
  python x7_birddev.py --phase analyze --models $model
done

# Steps that use outputs of the earlier runs on the second stack (DATA.md)
python anchor_spider.py --frozen "$WORK/inbox/frozen/spider_llama31" \
  --db-dir "$WORK/data/spider/spider_data/test_database" --test-suite "$WORK/data/dl/test-suite-sql-eval" \
  --model "$WORK/models/llama31_8b" --out "$WORK/outbox/results/anchor_spider_llama31" --stacks hf,vllm   # G1
python t9_testsuite.py 40                                                                                   # T9
