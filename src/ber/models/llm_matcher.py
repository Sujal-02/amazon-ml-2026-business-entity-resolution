"""LoRA-tuned decoder LLM as a pair classifier on the UNCERTAIN band only (optional zoo member).

Bases (Apache-2.0, <= 8B params): Qwen/Qwen3-4B-Instruct-2507 (4.02B), Qwen/Qwen2.5-7B-Instruct (7.62B).
(Qwen3-8B = 8.19B params and Jellyfish = CC-BY-NC are NOT allowed by the challenge constraints.)

Sequence-classification head (1 logit) on the last token + LoRA r=16 on all linear layers; bf16.
Train: pairs from deep_train_folds whose GBDT OOF p_blend lies in cfg.llm.<name>.band (hard cases) + all band positives.
Predict: band pairs of the holdout folds (for the stacker) and of the test split -> work/<split>/score_llm_<name>.parquet

  python -m ber.models.llm_matcher train   --config configs/aws.yaml --name qwen3_4b
  python -m ber.models.llm_matcher predict --config configs/aws.yaml --name qwen3_4b --split train|test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROMPT = ("Do these two business records refer to the same real-world business? Names may be abbreviated, "
          "transliterated, misspelled or reordered; addresses may be partial or reformatted.\n"
          "Record A: {a}\nRecord B: {b}\nAnswer:")


def _wp(cfg, split, name):
    return os.path.join(cfg["work_dir"], split, name)


def _texts(rec, i1, i2, tok, per_side=110):
    """Truncate EACH record (not the whole prompt) so the 'Answer:' cue is never cut off."""
    from .cross_encoder import pair_texts

    A, B = pair_texts(rec, i1, i2)
    cut = lambda xs: tok.batch_decode([ids[:per_side] for ids in tok(xs, add_special_tokens=False)["input_ids"]])
    return [PROMPT.format(a=a, b=b) for a, b in zip(cut(A), cut(B))]


def _band(cfg, split, lo, hi):
    s = pd.read_parquet(_wp(cfg, split, "oof.parquet" if split == "train" else "scores_gbdt.parquet"))
    return s[(s.p_blend >= lo) & (s.p_blend <= hi)]


def train(cfg, name):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

    lc = cfg["llm"][name]
    lo, hi = lc.get("band", [0.02, 0.98])
    rec = pd.read_parquet(_wp(cfg, "train", "records.parquet"))
    df = _band(cfg, "train", lo, hi)
    df = df[df["fold"].isin(set(cfg.get("deep_train_folds", [0, 1, 2])))]
    df = df.sample(min(len(df), lc.get("max_train_pairs", 600000)), random_state=0).reset_index(drop=True)
    log.info("LLM %s: %d band training pairs (pos rate %.3f)", name, len(df), df.y.mean())
    tok = AutoTokenizer.from_pretrained(lc["base"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(lc["base"], num_labels=1, dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=lc.get("lora_r", 16), lora_alpha=lc.get("lora_alpha", 32),
                                             lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 master weights for LoRA + score head (bf16 AdamW updates underflow)
    from .cross_encoder import _device

    dev = _device(cfg)
    model.to(dev).train()
    model.gradient_checkpointing_enable() if lc.get("grad_ckpt", True) else None
    texts = _texts(rec, df.i1.to_numpy(), df.i2.to_numpy(), tok)
    y = df.y.to_numpy(np.float32)
    bs, accum = lc.get("batch_size", 16), lc.get("grad_accum", 4)
    steps = lc.get("epochs", 1) * math.ceil(len(df) / (bs * accum))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lc.get("lr", 1e-4), weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    step, t0 = 0, time.time()
    for ep in range(lc.get("epochs", 1)):
        order = np.random.default_rng(ep).permutation(len(df))
        for bi, s in enumerate(range(0, len(order), bs)):
            idx = order[s:s + bs]
            t = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=lc.get("max_len", 256), return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logit = model(**t, use_cache=False).logits.squeeze(-1).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[idx]).to(dev)) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 100 == 0:
                    log.info("LLM %s step %d/%d loss %.4f (%.1f pairs/s)", name, step, steps, loss.item() * accum, (s + bs) / (time.time() - t0))
    out = os.path.join(cfg["artifacts_dir"], f"llm_{name}")
    model.save_pretrained(out); tok.save_pretrained(out)
    json.dump({"base": lc["base"]}, open(os.path.join(out, "ber_meta.json"), "w"))
    return out


def predict(cfg, name, split):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    lc = cfg["llm"][name]
    lo, hi = lc.get("band", [0.02, 0.98])
    out = os.path.join(cfg["artifacts_dir"], f"llm_{name}")
    rec = pd.read_parquet(_wp(cfg, split, "records.parquet"))
    df = _band(cfg, split, lo, hi)
    if split == "train":
        df = df[~df["fold"].isin(set(cfg.get("deep_train_folds", [0, 1, 2])))]
    df = df.reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(out)
    tok.padding_side = "left"
    base = AutoModelForSequenceClassification.from_pretrained(lc["base"], num_labels=1, dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    from .cross_encoder import _device

    dev = _device(cfg)
    model = PeftModel.from_pretrained(base, out).merge_and_unload().to(dev).eval()
    texts = _texts(rec, df.i1.to_numpy(), df.i2.to_numpy(), tok)
    order = np.argsort([len(t) for t in texts])
    res = np.empty(len(df), np.float32)
    bs = lc.get("infer_batch", 64)
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            t = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=lc.get("max_len", 256), return_tensors="pt").to(dev)
            res[idx] = model(**t, use_cache=False).logits.squeeze(-1).float().cpu().numpy()
    path = _wp(cfg, split, f"score_llm_{name}.parquet")
    pd.DataFrame({"i1": df.i1, "i2": df.i2, f"llm_{name}": res}).to_parquet(path)
    log.info("LLM %s predicted %d band pairs on %s", name, len(df), split)
    return path


def main(argv=None):
    from ..pipeline import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(a.config)
    print(train(cfg, a.name) if a.cmd == "train" else predict(cfg, a.name, a.split))


if __name__ == "__main__":
    main()
