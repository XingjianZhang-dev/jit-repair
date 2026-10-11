"""Generation engine (vLLM backend) shared by the experiment drivers.

Prompts are rendered with the model's own chat template exactly as the frozen HF
runners did (apply_chat_template(tokenize=False, add_generation_prompt=True,
enable_thinking=...)) and then tokenized with tokenizer(text) -- default
add_special_tokens=True, i.e. the frozen input ids (including the duplicated BOS
that Llama/Mistral templates receive in the frozen runners).
Output text = tokenizer.decode(token_ids, skip_special_tokens=True).strip(), as frozen.

Rows: {"id": str, "system": str|None, "user": str}
Output rows (appended per chunk, resumable by id):
  {"id", "input_tokens", "samples": [{"text","output_tokens","finish_reason","seed"}], meta...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time


def stable_seed(base: int, key: str) -> int:
    return (base * 1000003 + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**31 - 1)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


class Engine:
    def __init__(self, model: str, model_tag: str, max_model_len: int = 8192, gpu_mem: float = 0.90,
                 max_num_seqs: int = 256, seed: int = 20261011):
        from transformers import AutoTokenizer
        from vllm import LLM
        import torch, vllm, transformers
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        # frozen runners: tokenizer.pad_token_id, or eos_token_id when the tokenizer has no pad;
        # frozen output_tokens = number of generated ids != that pad id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.llm = LLM(model=model, dtype="bfloat16", seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem, enable_prefix_caching=True,
                       max_num_seqs=max_num_seqs, tokenizer=model)
        self.max_model_len = max_model_len
        self.meta = {"model_tag": model_tag, "model_path": model, "backend": "vllm", "vllm": vllm.__version__,
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0),
                     "frozen_pad_id": self.pad_id}

    def render(self, row, enable_thinking: str):
        msgs = []
        if row.get("system") is not None:
            msgs.append({"role": "system", "content": row["system"]})
        msgs.append({"role": "user", "content": row["user"]})
        kw = {} if enable_thinking == "omit" else {"enable_thinking": enable_thinking == "true"}
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return text, self.tok(text)["input_ids"]

    def run(self, rows, output: str, *, n: int = 1, temperature: float = 0.0, top_p: float = 1.0,
            top_k: int = -1, min_p: float = 0.0, max_tokens: int, seed: int = 20261011,
            enable_thinking: str = "false", chunk: int = 1024, tag: str = ""):
        from vllm import SamplingParams
        done = {r["id"] for r in read_jsonl(output)}
        pending = [r for r in rows if r["id"] not in done]
        print(f"[{tag}] rows={len(rows)} done={len(done)} pending={len(pending)}", flush=True)
        meta = dict(self.meta, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, n=n,
                    max_tokens=max_tokens, enable_thinking=enable_thinking, base_seed=seed)
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        example = output + ".rendered_example.txt"
        with open(output, "a", encoding="utf-8") as handle:
            for start in range(0, len(pending), chunk):
                part = pending[start: start + chunk]
                prompts, params, rendered = [], [], []
                for r in part:
                    text, ids = self.render(r, enable_thinking)
                    mt = max_tokens
                    if len(ids) + mt > self.max_model_len:
                        raise ValueError(f"prompt too long for max_model_len: {r['id']} {len(ids)}")
                    s = stable_seed(seed, r["id"]) if temperature > 0 else None
                    rendered.append((text, ids))
                    prompts.append({"prompt_token_ids": ids})
                    params.append(SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k,
                                                 min_p=min_p, max_tokens=mt, seed=s, skip_special_tokens=True))
                if not os.path.exists(example):
                    with open(example, "w", encoding="utf-8") as ex:
                        ex.write(rendered[0][0] + "\n\n#input_ids[:12]=" + json.dumps(rendered[0][1][:12]) + "\n")
                t0 = time.perf_counter()
                outs = self.llm.generate(prompts, params, use_tqdm=False)
                dt = time.perf_counter() - t0
                for r, (text, ids), o, sp in zip(part, rendered, outs, params):
                    samples = [{"text": self.tok.decode(list(c.token_ids), skip_special_tokens=True).strip(),
                                "output_tokens": len(c.token_ids),
                                "output_tokens_frozen_convention": sum(1 for t in c.token_ids if t != self.pad_id),
                                "finish_reason": c.finish_reason,
                                "seed": sp.seed} for c in o.outputs]
                    rec = {"id": r["id"], "input_tokens": len(ids), "samples": samples,
                           "chunk_seconds": dt, "chunk_size": len(part), **meta}
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{tag}] {start + len(part)}/{len(pending)} chunk_s={dt:.1f}", flush=True)
        return {r["id"]: r for r in read_jsonl(output)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--enable-thinking", choices=["true", "false", "omit"], default="false")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    rows = read_jsonl(a.input)
    if a.limit:
        rows = rows[: a.limit]
    eng = Engine(a.model, a.model_tag, max_model_len=a.max_model_len, seed=a.seed)
    eng.run(rows, a.output, n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, min_p=a.min_p,
            max_tokens=a.max_tokens, seed=a.seed, enable_thinking=a.enable_thinking, chunk=a.chunk, tag=a.model_tag)


if __name__ == "__main__":
    main()
