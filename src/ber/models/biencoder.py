"""Multilingual bi-encoder for dense blocking (contrastive fine-tuning on train positives).

Default base: intfloat/multilingual-e5-base (MIT, 278M). Alternatives: intfloat/multilingual-e5-small (MIT, 118M),
BAAI/bge-m3 (MIT, 568M), sentence-transformers/LaBSE (Apache-2.0).

Training
  * anchor = S1 text, positive = one random matched S2/S3 record per S1 per epoch -> every batch holds distinct S1
    entities, so there are no in-batch false negatives (S1 is deduplicated, so other S1s are true negatives);
  * optional hard negatives = top-ranked non-matching blocking candidates of the same S1 (from cands/feats parquet);
  * symmetric InfoNCE (temperature 0.05) over [positives ; hard negatives], AdamW, linear warmup/decay, AMP.
Text: "query: <name> | <address>" on both sides (romanised name when the raw name is in an Indic script, raw
otherwise, so the model sees what the rest of the pipeline sees).

  python -m ber.models.biencoder train --config configs/aws.yaml
  (encode() is used by pipeline.step_block when blocking.dense_model points to the trained directory)
"""
from __future__ import annotations

import argparse
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


def record_text(rec: pd.DataFrame) -> List[str]:
    name = rec["name_core"].where(rec["name_core"] != "", rec["name_norm"])
    legal = rec["name_legal"]
    addr = rec["addr_norm"]
    return ("query: " + name + " " + legal + " | " + addr).str.strip().tolist()


def _mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


def train(cfg: dict) -> str:
    import torch
    import torch.nn.functional as Fn
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    bc = cfg.get("biencoder", {})
    base = bc.get("base", "intfloat/multilingual-e5-base")
    out_dir = bc.get("out_dir", os.path.join(cfg["artifacts_dir"], "biencoder"))
    max_len, bs, epochs, lr = bc.get("max_len", 64), bc.get("batch_size", 256), bc.get("epochs", 2), bc.get("lr", 3e-5)
    tau, n_hard = bc.get("temperature", 0.05), bc.get("hard_negatives", 1)
    wd = os.path.join(cfg["work_dir"], "train")
    rec = pd.read_parquet(os.path.join(wd, "records.parquet"))
    with open(os.path.join(wd, "gt.json")) as f:
        gt = json.load(f)
    text = record_text(rec)
    pos_of = {e: i for i, e in enumerate(rec.entity_id)}
    anchors = [(pos_of[s], ps) for s, cs in gt.items() if s in pos_of
               for ps in [[pos_of[c] for c in cs if c in pos_of]] if ps]
    # fold filter: deep models train on folds listed in cfg.deep_train_folds (holdout stacking)
    from ..features import fold_of

    tf = set(cfg.get("deep_train_folds", [0, 1, 2]))
    keep = fold_of([rec.entity_id.iat[a] for a, _ in anchors], cfg.get("n_folds", 5))
    anchors = [x for x, k in zip(anchors, keep) if k in tf]
    hard = {}
    hp = os.path.join(wd, "cands.parquet")
    if n_hard > 0 and os.path.exists(hp):
        c = pd.read_parquet(hp, columns=["i1", "i2", "blk_prescore"])
        c = c[np.isin(c.i1.to_numpy(), np.fromiter((a for a, _ in anchors), np.int64))]
        n = int(len(rec))
        tkey = np.fromiter((a * n + b for a, bs_ in anchors for b in bs_), np.int64)
        c = c[~np.isin(c.i1.to_numpy().astype(np.int64) * n + c.i2.to_numpy().astype(np.int64), tkey)]
        c = c.sort_values(["i1", "blk_prescore"], ascending=[True, False]).groupby("i1").head(5)
        hard = c.groupby("i1")["i2"].agg(list).to_dict()
    log.info("biencoder: %d anchors, %d with hard negatives, base=%s", len(anchors), len(hard), base)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModel.from_pretrained(base).to(dev)
    model.gradient_checkpointing_enable() if bc.get("grad_ckpt", False) else None
    steps = epochs * math.ceil(len(anchors) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and not use_bf16))

    def enc(batch_texts):
        t = tok(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=dev == "cuda"):
            h = model(**t).last_hidden_state
        return Fn.normalize(_mean_pool(h.float(), t["attention_mask"]), dim=-1)

    rng = random.Random(cfg.get("seed", 0))
    step = 0
    model.train()
    for ep in range(epochs):
        rng.shuffle(anchors)
        t0 = time.time()
        for s in range(0, len(anchors), bs):
            batch = anchors[s:s + bs]
            a_txt = [text[a] for a, _ in batch]
            p_txt = [text[rng.choice(ps)] for _, ps in batch]
            n_txt = []
            for a, _ in batch:
                hn = hard.get(a, [])
                if hn and n_hard > 0:
                    n_txt.extend(text[x] for x in rng.sample(hn, min(n_hard, len(hn))))
            ea, ep_ = enc(a_txt), enc(p_txt)
            cand = torch.cat([ep_, enc(n_txt)], 0) if n_txt else ep_
            logits = ea @ cand.T / tau
            labels = torch.arange(len(batch), device=dev)
            loss = Fn.cross_entropy(logits, labels) + Fn.cross_entropy(ep_ @ ea.T / tau, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            step += 1
            if step % 200 == 0:
                log.info("ep %d step %d/%d loss %.4f (%.0f anchors/s)", ep, step, steps, loss.item(),
                         (s + bs) / (time.time() - t0))
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "ber_biencoder.json"), "w") as f:
        json.dump({"base": base, "max_len": max_len}, f)
    return out_dir


_CACHE = {}


def encode(model_dir: str, texts: List[str], batch: int = 1024) -> np.ndarray:
    """L2-normalised float16 embeddings. `texts` are the pipeline text views; the 'query: ' prefix is added here."""
    import torch
    import torch.nn.functional as Fn
    from transformers import AutoModel, AutoTokenizer

    if model_dir not in _CACHE:
        meta = json.load(open(os.path.join(model_dir, "ber_biencoder.json")))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModel.from_pretrained(model_dir).to(dev).eval()
        if dev == "cuda":
            m = m.half()
        _CACHE[model_dir] = (AutoTokenizer.from_pretrained(model_dir), m, dev, meta["max_len"])
    tok, m, dev, max_len = _CACHE[model_dir]
    out = np.empty((len(texts), m.config.hidden_size), np.float16)
    order = np.argsort([len(t) for t in texts])  # length-sorted batching = less padding
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            batch_txt = [texts[i] if texts[i].startswith("query: ") else "query: " + texts[i] for i in idx]
            t = tok(batch_txt, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            h = m(**t).last_hidden_state
            e = Fn.normalize(_mean_pool(h.float(), t["attention_mask"]), dim=-1)
            out[idx] = e.cpu().numpy().astype(np.float16)
    return out


def main(argv=None):
    from ..pipeline import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(train(load_cfg(a.config)))


if __name__ == "__main__":
    main()
