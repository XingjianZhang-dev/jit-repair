#!/usr/bin/env bash
# Downloads the four models at fixed revisions into $JIT_WORK/models and records the weight-shard sha256 values.
# Gated repositories need a Hugging Face access token in the HF_TOKEN environment variable.
set -euo pipefail
WORK="${JIT_WORK:-/workspace}"
mkdir -p "$WORK/models"
JIT_MODELS_DIR="$WORK/models" python - <<'PY'
import json, os
from huggingface_hub import HfApi, snapshot_download

root = os.environ["JIT_MODELS_DIR"]
api = HfApi()
specs = [("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1", "llama31_8b"),
         ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", "mistral7b"),
         ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen3_8b"),
         ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "qwen3_14b")]
record = {}
for repo, revision, name in specs:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    path = snapshot_download(repo, revision=info.sha, local_dir=os.path.join(root, name),
                             allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja"],
                             token=os.environ.get("HF_TOKEN") or None)
    shards = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings if s.rfilename.endswith(".safetensors")}
    record[name] = {"repo": repo, "revision": info.sha, "path": path, "lfs_sha256": shards}
    with open(os.path.join(root, "MODEL_REVISIONS.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print("downloaded", name, info.sha, flush=True)
PY
