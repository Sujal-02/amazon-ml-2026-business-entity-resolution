import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

t0 = time.time()
D = os.path.dirname(sorted(glob.glob("/kaggle/input/**/cal.parquet", recursive=True))[0]); D2 = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tr2.parquet", recursive=True))[0])
print("data dir:", D, flush=True)
OUT = "/kaggle/working"
assert torch.cuda.is_available(), "NO GPU"
dev = "cuda"
print("gpu:", torch.cuda.get_device_name(0), "n:", torch.cuda.device_count(), flush=True)

tr = pd.read_parquet(f"{D2}/tr2.parquet")
cal = pd.read_parquet(f"{D}/cal.parquet")
te_files = sorted(glob.glob(f"{D}/te_*.parquet"))
print(f"tr {len(tr):,} (pos {tr.y.mean():.3f})  cal {len(cal):,}  te files {len(te_files)}", flush=True)

NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=1).to(dev)
MAXLEN = 128


def enc(a, b):
    return tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")


EPOCHS, BS, LR = 3, 256, 3e-5
TRAIN_BUDGET_S = 45 * 60
n = len(tr)
steps = EPOCHS * (n // BS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = nn.BCEWithLogitsLoss()
A, B, Y = tr.a.to_numpy(), tr.b.to_numpy(), tr.y.to_numpy(np.float32)
model.train()
step, stop = 0, False
for ep in range(EPOCHS):
    perm = np.random.default_rng(ep).permutation(n)
    run = 0.0
    for s in range(0, n - BS + 1, BS):
        idx = perm[s:s + BS]
        b = enc(A[idx], B[idx]).to(dev)
        y = torch.from_numpy(Y[idx]).to(dev)
        with torch.cuda.amp.autocast():
            logit = model(**b).logits.squeeze(-1)
        loss = lossf(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run = 0.98 * run + 0.02 * float(loss)
        if step % 500 == 0:
            print(f"ep {ep} step {step}/{steps} loss {run:.4f} t {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > TRAIN_BUDGET_S:
            stop = True
            break
    if stop:
        print("train time budget hit", flush=True)
        break


@torch.no_grad()
def predict(df, bs=1024):
    model.eval()
    a, b = df.a.to_numpy(), df.b.to_numpy()
    out = np.empty(len(df), np.float32)
    for s in range(0, len(df), bs):
        x = enc(a[s:s + bs], b[s:s + bs]).to(dev)
        with torch.cuda.amp.autocast():
            out[s:s + bs] = model(**x).logits.squeeze(-1).float().cpu().numpy()
    return out


cl = predict(cal)
pd.DataFrame({"pid": cal.pid.to_numpy(), "ce": cl}).to_parquet(f"{OUT}/cal_ce.parquet", index=False)
yc = cal.y.to_numpy()
pc = 1 / (1 + np.exp(-cl))
print(f"cal done t {time.time() - t0:.0f}s | CE acc@0.5 {((pc > 0.5) == (yc > 0.5)).mean():.4f} "
      f"GBDT acc@0.5 {((cal.p.to_numpy() > 0.5) == (yc > 0.5)).mean():.4f}", flush=True)

parts = []
for f in te_files:
    te = pd.read_parquet(f)
    parts.append(pd.DataFrame({"pid": te.pid.to_numpy(), "ce": predict(te)}))
    print(f"{os.path.basename(f)} {len(te):,} done t {time.time() - t0:.0f}s", flush=True)
pd.concat(parts, ignore_index=True).to_parquet(f"{OUT}/te_ce.parquet", index=False)
print("ALL DONE", time.time() - t0, flush=True)
