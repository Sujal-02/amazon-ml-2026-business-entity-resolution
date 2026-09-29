"""Multilingual cross-encoder matcher (Ditto-style serialisation + numeric-feature head, Foursquare-1st-place style).

Supported bases (all MIT / Apache-2.0, <= 8B params):
  FacebookAI/xlm-roberta-large (MIT, 561M)      BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M)
  microsoft/mdeberta-v3-base (MIT, 278M)         jhu-clsp/mmBERT-base (MIT, 307M)
  FacebookAI/xlm-roberta-base (MIT, 278M)        Alibaba-NLP/gte-multilingual-reranker-base (Apache-2.0, 306M)

Holdout stacking: trained on S1 folds cfg.deep_train_folds (default 0,1,2); predicts the remaining train folds and the
test split; the stacker (models/stack.py) is fit on the remaining folds only -> no leakage of CE scores.

  python -m ber.models.cross_encoder train   --config configs/aws.yaml --name xlmr_large
  python -m ber.models.cross_encoder predict --config configs/aws.yaml --name xlmr_large --split train|test
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# numeric side-features concatenated to the [CLS] vector (country-agnostic; NaN -> 0 plus a missing flag)
NUM_FEATS = ["p1", "nm_core_tset", "nm_core_ratio", "nm_core_jw", "nm_key_tset", "nm_skel_tset", "nm_idf_jac",
             "nm_idf_cov_s1", "nm_idf_cov_c", "nm_max_extra_c", "nm_n_extra_c", "nm_extra_harm_max",
             "nm_extra_unknown", "nm_parts_best_tset", "nm_concat_full_partial", "nm_legal_conflict", "ad_tset",
             "ad_idf_jac", "ad_num_jac", "ad_num_first_eq", "ad_num_first_lev", "ad_house_eq", "ad_street_tset",
             "ad_loc_tset", "ad_state_eq", "ad_empty_c", "cos_lsa_name", "cos_lsa_addr", "cos_lsa_both"]


def _wp(cfg, split, name):
    return os.path.join(cfg["work_dir"], split, name)


def pair_texts(rec: pd.DataFrame, i1: np.ndarray, i2: np.ndarray):
    name_raw = rec["raw_name"].to_numpy() if "raw_name" in rec else rec["name_norm"].to_numpy()
    addr_raw = rec["raw_addr"].to_numpy() if "raw_addr" in rec else rec["addr_norm"].to_numpy()
    core = rec["name_core"].to_numpy(); had = rec["name_had_indic"].to_numpy()
    anorm = rec["addr_norm"].to_numpy()

    def side(i, tag):
        n = name_raw[i]
        if had[i]:
            n = f"{n} ({core[i]})"
        return f"{tag} {n} [ADDR] {addr_raw[i]} [NORM] {anorm[i]}"

    # role tags survive the A/B swap augmentation: the model can tell which side carries an inserted word
    return [side(a, "[REF]") for a in i1], [side(b, "[CAND]") for b in i2]


def _device(cfg):
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if not cfg.get("allow_cpu", False):
        raise RuntimeError("CUDA not available; set allow_cpu: true in the config only for smoke runs")
    return "cpu"


class CEModel:
    """transformer + numeric-feature MLP head; built lazily so the module imports without torch."""

    @staticmethod
    def build(base: str, n_num: int, dropout: float = 0.1):
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = AutoModel.from_pretrained(base)
                h = self.enc.config.hidden_size
                self.num = nn.Sequential(nn.Linear(2 * n_num, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h + 128, 256), nn.GELU(), nn.Linear(256, 1))

            def forward(self, input_ids, attention_mask, num, token_type_ids=None):
                kw = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None and getattr(self.enc.config, "type_vocab_size", 1) > 1:
                    kw["token_type_ids"] = token_type_ids
                h = self.enc(**kw).last_hidden_state[:, 0]
                return self.head(torch.cat([h.float(), self.num(num)], -1)).squeeze(-1)

        return _M()


def _num_matrix(df: pd.DataFrame, cols: List[str], stats: dict = None):
    X = np.stack([df[c].to_numpy(np.float32) if c in df else np.full(len(df), np.nan, np.float32) for c in cols], 1)
    miss = np.isnan(X).astype(np.float32)
    if stats is None:
        stats = {"mu": np.nanmean(X, 0).tolist(), "sd": (np.nanstd(X, 0) + 1e-6).tolist()}
    X = (np.nan_to_num(X - np.array(stats["mu"], np.float32)) / np.array(stats["sd"], np.float32)).astype(np.float32)
    return np.concatenate([X, miss], 1), stats


def _load_pairs(cfg, split, folds=None, exclude_folds=None):
    """Only the columns the CE needs (i1,i2,y,fold + NUM_FEATS) + p1/p_blend (OOF on train, full model on test)."""
    from .. import store

    have = set(store.columns(cfg, split))
    cols = ["i1", "i2"] + [c for c in ("y", "fold") if c in have] + [c for c in NUM_FEATS if c != "p1" and c in have]
    filt = None
    if folds is not None:
        filt = lambda d: d["fold"].isin(set(folds)).to_numpy()
    elif exclude_folds is not None:
        filt = lambda d: ~d["fold"].isin(set(exclude_folds)).to_numpy()
    df = store.load_all(cfg, split, columns=cols, row_filter=filt if "fold" in have else None)
    sp = _wp(cfg, split, "oof.parquet" if split == "train" else "scores_gbdt.parquet")
    s = pd.read_parquet(sp, columns=["i1", "i2", "p1", "p_blend"])
    return df.merge(s, on=["i1", "i2"], how="left")


def train(cfg: dict, name: str) -> str:
    import torch
    import torch.nn.functional as Fn
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    cc = cfg["cross_encoders"][name]
    base = cc["base"]
    out_dir = os.path.join(cfg["artifacts_dir"], f"ce_{name}")
    max_len, bs, accum = cc.get("max_len", 160), cc.get("batch_size", 32), cc.get("grad_accum", 4)
    epochs, lr = cc.get("epochs", 2), cc.get("lr", 1e-5)
    rec = pd.read_parquet(_wp(cfg, "train", "records.parquet"))
    tf = set(cfg.get("deep_train_folds", [0, 1, 2]))
    df = _load_pairs(cfg, "train", folds=tf)
    band = cc.get("train_min_p1", 0.001)
    pos = df[df.y == 1]
    neg = df[(df.y == 0) & (df.p1 >= band)]
    ratio = cc.get("neg_ratio", 4)
    if len(neg) > ratio * len(pos):
        neg = neg.sample(ratio * len(pos), random_state=0)
    max_pairs = cc.get("max_train_pairs", 4_000_000)
    tr = pd.concat([pos, neg]).sample(frac=1.0, random_state=0).head(max_pairs).reset_index(drop=True)
    log.info("CE %s: base=%s train pairs=%d (pos %d)", name, base, len(tr), int(tr.y.sum()))
    ta, tb = pair_texts(rec, tr.i1.to_numpy(), tr.i2.to_numpy())
    num, stats = _num_matrix(tr, NUM_FEATS)
    y = tr.y.to_numpy(np.float32)

    dev = _device(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 runs (mdeberta) use TF32 tensor cores
    tok = AutoTokenizer.from_pretrained(base)
    model = CEModel.build(base, len(NUM_FEATS)).to(dev)
    ema = copy.deepcopy(model).eval() if cc.get("ema", 0.999) else None
    decay = cc.get("ema", 0.999)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.enc.named_parameters() if not n.endswith(no_decay)], "weight_decay": 0.01, "lr": lr},
        {"params": [p for n, p in model.enc.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0, "lr": lr},
        {"params": list(model.num.parameters()) + list(model.head.parameters()), "weight_decay": 0.01, "lr": lr * 20},
    ]
    opt = torch.optim.AdamW(groups)
    steps = epochs * math.ceil(len(tr) / (bs * accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    fp = cc.get("precision", "bf16")  # mdeberta: use "fp32" or "fp16" (bf16 NaNs reported)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(fp)
    scaler = torch.amp.GradScaler("cuda", enabled=fp == "fp16" and dev == "cuda")
    rng = random.Random(0)
    step = 0
    model.train()
    for ep in range(epochs):
        order = np.random.default_rng(ep).permutation(len(tr))
        t0 = time.time()
        for bi, s in enumerate(range(0, len(order), bs)):
            idx = order[s:s + bs]
            A = [ta[i] for i in idx]; Bx = [tb[i] for i in idx]
            if cc.get("swap_aug", True):  # order-robustness (RobEM): random A/B swap
                for j in range(len(A)):
                    if rng.random() < 0.5:
                        A[j], Bx[j] = Bx[j], A[j]
            t = tok(A, Bx, padding=True, truncation="longest_first", max_length=max_len, return_tensors="pt").to(dev)
            nm = torch.from_numpy(num[idx]).to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None and dev == "cuda"):
                logit = model(t["input_ids"], t["attention_mask"], nm, t.get("token_type_ids"))
            loss = Fn.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[idx]).to(dev)) / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
                if ema is not None:
                    d = min(decay, (1 + step) / (10 + step))  # EMA warm-up
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                        for be, bm in zip(ema.buffers(), model.buffers()):
                            be.copy_(bm)
                if step % 200 == 0:
                    log.info("CE %s ep %d step %d/%d loss %.4f %.0f pairs/s", name, ep, step, steps,
                             loss.item() * accum, (s + bs) / (time.time() - t0))
    os.makedirs(out_dir, exist_ok=True)
    final = ema if ema is not None else model
    torch.save(final.state_dict(), os.path.join(out_dir, "model.pt"))
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"base": base, "max_len": max_len, "num_feats": NUM_FEATS, "stats": stats}, f)
    return out_dir


def predict(cfg: dict, name: str, split: str) -> str:
    import torch
    from transformers import AutoTokenizer

    cc = cfg["cross_encoders"][name]
    out_dir = os.path.join(cfg["artifacts_dir"], f"ce_{name}")
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    rec = pd.read_parquet(_wp(cfg, split, "records.parquet"))
    df = _load_pairs(cfg, split, exclude_folds=set(cfg.get("deep_train_folds", [0, 1, 2])) if split == "train" else None)
    band = cc.get("predict_min_p", 0.002)
    todo = df[df["p_blend"].fillna(df["p1"]) >= band].reset_index(drop=True)
    log.info("CE %s predict %s: %d of %d pairs in band", name, split, len(todo), len(df))
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = CEModel.build(meta["base"], len(meta["num_feats"]))
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu"))
    dev = _device(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = model.to(dev).eval()
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cc.get("precision", "bf16"))
    ta, tb = pair_texts(rec, todo.i1.to_numpy(), todo.i2.to_numpy())
    num, _ = _num_matrix(todo, meta["num_feats"], meta["stats"])
    out = np.empty(len(todo), np.float32)
    bs = cc.get("infer_batch", 512)
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)])
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            nm = torch.from_numpy(num[idx]).to(dev)
            logits = 0
            for A, Bx in (([ta[i] for i in idx], [tb[i] for i in idx]), ([tb[i] for i in idx], [ta[i] for i in idx])):
                t = tok(A, Bx, padding=True, truncation="longest_first", max_length=meta["max_len"], return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None and dev == "cuda"):
                    logits = logits + model(t["input_ids"], t["attention_mask"], nm, t.get("token_type_ids")).float()
            out[idx] = (logits / 2).cpu().numpy()  # symmetric test-time averaging
    res = pd.DataFrame({"i1": todo.i1, "i2": todo.i2, f"ce_{name}": out})
    path = _wp(cfg, split, f"score_ce_{name}.parquet")
    res.to_parquet(path)
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
