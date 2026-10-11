# Data

No benchmark data, model weights or model outputs are redistributed here. This file lists every input, where it comes
from, how it is prepared, and the sha256 of the files used in the paper. Paths are relative to `$JIT_WORK`
(`jit_repair/paths.py`).

## Benchmarks and evaluators (`data/`)

`scripts/download_data.sh` downloads, checks and unpacks the following.

| Item | Source | Checksum or revision |
|---|---|---|
| Spider (`spider_data.zip`; test split, 2,147 questions, test databases) | official release | sha256 `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` |
| BIRD dev (`dev.zip`; 1,534 questions, 11 databases) | official release | sha256 `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630` |
| BIRD Mini-Dev (500 questions; `dev_prompt.jsonl` schemas) | github.com/bird-bench/mini_dev | commit `b3d4bcbbae9a96934ad812551eb400c7a3b23c12` |
| Spider test-suite evaluator | github.com/taoyds/test-suite-sql-eval | commit `e97acc546ecbee8fa27fa8dbf025ef61493a876c`; `evaluation.py` sha256 `7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779`, `process_sql.py` sha256 `927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1` |
| Distilled test-suite databases (`testsuitedatabases.zip`) | same repository | sha256 `9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96` |
| Dr.Spider (`data.official.tar.gz`, extracted to `data/drspider/data_official`) | github.com/awslabs/diagnostic-robustness-text-to-sql | sha256 `d0f47e4d2c9202f4fd2d956a34e9dd1ff8180c1c9e58cf5b530c13fefaaa3ba9` |
| ProofWriter (structured theories and questions, OWA depth splits) | Tafjord et al. (2021) | used to build the two ProofWriter sets below |
| Logic-LM ProofWriter test file (600 depth-5 questions) | github.com/teacherpeterpan/Logic-LLM | used to build the depth-5 set below |

Benchmark data remain under their original licenses.

## Models (`models/`)

`scripts/download_models.sh` downloads the four instruction-tuned models at fixed revisions and records the revision
and the sha256 of every weight shard in `models/MODEL_REVISIONS.json`. Revisions and weight-shard hashes used in the
paper:

| Model | Repository @ revision |
|---|---|
| Llama-3.1-8B-Instruct | `unsloth/Meta-Llama-3.1-8B-Instruct` @ `a2856192dd7c25b842431f39c179a6c2c2f627d1` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Qwen3-8B | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen3-14B | `Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18` |

```
llama31_8b  model-00001-of-00004.safetensors  2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668
llama31_8b  model-00002-of-00004.safetensors  09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15
llama31_8b  model-00003-of-00004.safetensors  fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa
llama31_8b  model-00004-of-00004.safetensors  92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b
mistral7b   consolidated.safetensors          76d5729be995fb6510c7f103c5b8825d3d82f8126ac949db6be367a638694d1f
mistral7b   model-00001-of-00003.safetensors  ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19
mistral7b   model-00002-of-00003.safetensors  8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0
mistral7b   model-00003-of-00003.safetensors  905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca
qwen3_8b    model-00001-of-00005.safetensors  31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f
qwen3_8b    model-00002-of-00005.safetensors  5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282
qwen3_8b    model-00003-of-00005.safetensors  c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836
qwen3_8b    model-00004-of-00005.safetensors  b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a
qwen3_8b    model-00005-of-00005.safetensors  20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff
qwen3_14b   model-00001-of-00008.safetensors  e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e
qwen3_14b   model-00002-of-00008.safetensors  f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9
qwen3_14b   model-00003-of-00008.safetensors  dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9
qwen3_14b   model-00004-of-00008.safetensors  eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d
qwen3_14b   model-00005-of-00008.safetensors  97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6
qwen3_14b   model-00006-of-00008.safetensors  9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e
qwen3_14b   model-00007-of-00008.safetensors  0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750
qwen3_14b   model-00008-of-00008.safetensors  0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0
```

## Prepared inputs (`inbox/`)

Each builder writes the questions without references (`safe`) and the references in a separate file read only by the
scoring steps.

| File | Questions | Builder | sha256 of the file used in the paper |
|---|---|---|---|
| `inbox/frozen/spider_llama31/prepared/safe.jsonl` (Spider test) | 2,147 | `python -m jit_repair.prepare_spider_feedback_split --split test` | `d1b53ff71c7104d9840526135424479e4fb041be7bf2e1ff71756842033c1de6` |
| `inbox/frozen/bird_assets/safe.jsonl` (BIRD Mini-Dev) | 500 | `python -m jit_repair.prepare_bird_minidev_generic_forward` | `c6a68f11436cade9b77c85b824174db361e410836cdd34eb83b9808a03dc1703` |
| `inbox/drspider/safe_subset.jsonl` (Dr.Spider subset, 780 source questions) | 1,556 | `python -m jit_repair.prepare_drspider_oneuse` | `f92beb1ed4a292d4bb818a1c32f80114b9f55c7313afb77f34aed65ac4266733` |
| `outbox/results/x7/birddev_assets/safe.jsonl` (held-out BIRD) | 1,033 | `python experiments/prepare_heldout_bird.py` | `4a86c3c45090a0a061ea5af5422da46d36bd0a8ebfafc8fa7311dbd8d029a39c` |
| `inbox/data/proofwriter/proofwriter_fourval_test_1000.jsonl` (four-label set, 250 per label) | 1,000 | `python -m jit_repair.build_phase0` | `58a8fdafe91acf0bf7680d5826a48f87e76723f26ce802f6806f8645c487526a` |
| `inbox/data/proofwriter/logiclm_d5_600.jsonl` (depth-5 set) | 600 | `python -m jit_repair.build_logiclm_proofwriter` | `dfb6e5770b394dffbcfbc70a3523b69ef3ba39828eccae0cf56219f20e5ea687` |

The held-out BIRD set holds the BIRD dev questions outside Mini-Dev, after removing any question whose normalized
text equals, or has token-Jaccard similarity of at least 0.9 with, a Mini-Dev question (`prepare_heldout_bird.py`,
which also writes `audit.json`). Every builder documents its arguments (`--help`); compare the sha256 of a rebuilt file
with the table above.

Two steps use outputs of the earlier runs on the second hardware stack: the output-identity check G1
(`anchor_spider.py --frozen inbox/frozen/spider_llama31`) and the test-suite re-scoring T9 (`inbox/t9/<model>.jsonl`,
the stored Dr.Spider predictions of that stack).

## Per-question records

Every driver writes JSON Lines with one record per question (`case_id` plus the fields of that step), together with
summary JSON files. The analyses read them from `records/results/<experiment>/…` with the same relative paths as
under `outbox/results`:

| Directory | Contents |
|---|---|
| `c2/<bench>/<model>/` | greedy outputs, JIT-Repair (revise) outputs, 8 samples per question, checked samples, selections, per-question correctness |
| `pw/<set>/<model>/` | ProofWriter direct, step-by-step and thinking-mode outputs; logic programs and their revisions per round |
| `x3/<split>/<bench>/<model>/` | self-consistency winners and their gated revision |
| `x3b/<model>/` | Dr.Spider greedy, JIT-Repair (revise) and sample records, selections and per-question correctness |
| `x7/birddev/<model>/`, `x7/logprobs/` | held-out BIRD records and teacher-forced token log-probabilities |
| `t9/` | test-suite re-scoring |

`records/LEDGER.csv` lists the pre-registered predictions with the rule fixed in advance and the verdict;
`analysis/make_supp_tables.py` turns it into the table of Supplement S5.

The analyses of the second-stack runs (`experiments/archive/`, `x1_residuals.py`, `x7_alloc.py` for the archive
cells, and the replay tests) read the records of those runs from `records/archive/`: `evidence/` (per-run output
directories), `drspider_oneuse/<model>/` (Dr.Spider arms, cases and analysis) and `results/` (summary JSON files).
