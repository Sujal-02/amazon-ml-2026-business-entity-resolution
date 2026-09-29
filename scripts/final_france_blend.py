import sys, os, json, shutil, glob, subprocess, numpy as np, pandas as pd, yaml
sys.path.insert(0, "src")
from ber import decision as Dm, io as bio
cfg = yaml.safe_load(open("configs/local_final_tuned.yaml", encoding="utf-8"))
W, ART, OUT = cfg["work_dir"], cfg["artifacts_dir"], cfg["out_dir"]
bl = json.load(open("D:/er_data/ce_out2/blend.json")); w = np.array(bl["w"]); b = bl["b"]; print("weights", w, b)
dcfg = {k: v for k, v in json.load(open(ART + "/decision.json")).items() if k != "score"}
seen = set(json.load(open(ART + "/train_countries.json"))); ub = float(cfg.get("decision_unseen_bias", -0.3))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
tm = pd.read_parquet(W + "/ce_te_map.parquet"); tm = tm[tm.split.str.startswith("test@france")]
for nm, d in (("c2", "ce_out2"), ("c4", "ce_out4"), ("f1", "ce_outF1"), ("f2", "ce_outF2")):
    tm = tm.merge(pd.read_parquet(f"D:/er_data/{d}/te_ce.parquet").rename(columns={"ce": nm}), on="pid", how="inner")
tm["c4m"] = 0.5 * tm.c4 + 0.25 * (tm.f1 + tm.f2)
FIN = OUT + "_FINAL"
for s in glob.glob(OUT + "_CE2/test@*"):
    if "france" not in s: shutil.copytree(s, os.path.join(FIN, os.path.basename(s)), dirs_exist_ok=True)
for split, g in tm.groupby("split"):
    r = pd.read_parquet(f"{W}/{split}/records.parquet", columns=["entity_id", "src", "country_norm"]); eid = r.entity_id.to_numpy()
    sc = pd.read_parquet(f"{W}/{split}/scores_gbdt.parquet", columns=["i1", "i2", "p", "p_raw"]).merge(g[["i1", "i2", "c2", "c4m"]], on=["i1", "i2"], how="left")
    has = sc.c2.notna().to_numpy()
    z = w[0] * lg(sc.p_raw) + w[1] * sc.c2.fillna(0) + w[2] * sc.c4m.fillna(0) + b + (lg(sc.p) - lg(sc.p_raw))
    pn = np.where(has, 1 / (1 + np.exp(-z)), sc.p)
    un = ~np.isin(r.country_norm.to_numpy()[sc.i1.to_numpy()], list(seen))
    pi = Dm.decide(pd.DataFrame({"s1": sc.i1, "cand": sc.i2, "p": pn, "_unseen": un}), {**dcfg, "unseen_bias": ub}, "p")
    pred = {eid[x]: [eid[c] for c in cs] for x, cs in pi.items()}; s1 = r.entity_id[r.src == 1].tolist()
    od = os.path.join(FIN, split); os.makedirs(od, exist_ok=True)
    bio.write_matching(od + "/matching_results.tsv", s1, pred); shutil.copyfile(f"{OUT}/{split}/candidate_pairs.tsv", od + "/candidate_pairs.tsv")
    print(split, "links", sum(len(v) for v in pred.values()), "empty", round(sum(1 for x in s1 if not pred.get(x)) / len(s1), 4), flush=True)
print("shards:", len(glob.glob(FIN + "/test@*")))
vc = dict(cfg); vc["out_dir"] = FIN; vc["finalize"] = {**cfg.get("finalize", {}), "out_dir": FIN + "/final"}
yaml.safe_dump(vc, open("configs/local_final_FINAL.yaml", "w", encoding="utf-8"), sort_keys=False)
rr = subprocess.run([sys.executable, "-m", "ber.pipeline", "finalize", "--config", os.path.abspath("configs/local_final_FINAL.yaml")], cwd="src", capture_output=True, text=True)
print("\n".join([l for l in (rr.stdout + rr.stderr).splitlines() if "PASS" in l or "FAIL" in l or "rows" in l][-5:]))
