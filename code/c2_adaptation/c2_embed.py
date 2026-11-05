"""C2 (CPU, fast): adaptation on cached frozen embeddings from c1_pilot (results/c1_cross/emb).

For each backbone x target (child, autism) x seed x target-label fraction:
  source_only : probe trained on adult FER2013 only (C1 baseline)
  coral       : unsupervised CORAL - align source feature covariance to target (no target labels)
  target_only : probe on the labelled target fraction only
  combined    : probe on source + labelled target (target up-weighted)
Target split: 70% adaptation pool / 30% test, grouped by perceptual hash so near-duplicate
images never cross the split (important for the autism set).

Usage: python c2_embed.py --emb results/c1_cross/emb --out results/c2_real
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from PIL import Image
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "c1_benchmark"))
from c1_pilot import EMO, ALIASES

BACKBONES = ["mobilenetv3_large_100", "efficientnet_b0", "resnet50", "vit_small_patch16_224"]
AUT = "data/autism_emotions/Autism emotion recogition dataset/Autism emotion recogition dataset/train"
TARGETS = {"child": "data/fer2013_child90", "autism": AUT}


def folder_files(folder):   # same order as c1_pilot.folder_split
    return [(p, i) for i, e in enumerate(EMO) for a in ALIASES.get(e, [e])
            for p in sorted(glob.glob(os.path.join(folder, a, "*")))]


def ahash(p):
    px = np.asarray(Image.open(p).convert("L").resize((8, 8)), dtype=float).ravel()
    return "".join("1" if v > px.mean() else "0" for v in px)


def source_labels(seed=42, n=6000):
    ag = pd.read_csv("data/fer2013_age/age_predictions.csv")
    ex = ag[(ag.split == "train") & ((ag.group == "child") | (ag.p_teen > .3))].idx.tolist()
    d = pd.read_parquet("data/fer2013/train.parquet").drop(index=ex)
    idx = [i for _, g in d.groupby("label") for i in g.sample(max(1, round(n * len(g) / len(d))), random_state=seed).index]
    return d.loc[idx].label.values


def coral(Xs, Xt, eps=1.0):
    cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xs.shape[1])
    return Xs @ np.real(linalg.fractional_matrix_power(cs, -0.5)) @ np.real(linalg.fractional_matrix_power(ct, 0.5))


def fit_eval(Xtr, ytr, Xte, yte, w=None):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(sc.transform(Xtr), ytr, sample_weight=w)
    p = m.predict(sc.transform(Xte)); labs = sorted(set(yte))
    return {"acc": float(accuracy_score(yte, p)), "macro_f1": float(f1_score(yte, p, average="macro", labels=labs))}


def group_split(groups, y, seed):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rng.shuffle(ug)
    test_g = set(ug[: int(.3 * len(ug))])
    te = np.array([g in test_g for g in groups])
    return np.where(~te)[0], np.where(te)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="results/c1_cross/emb"); ap.add_argument("--out", default="results/c2_real")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--upweight", type=float, default=5.0); ap.add_argument("--pca", type=int, default=256)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ys = source_labels()
    tinfo = {}
    for t, f in TARGETS.items():
        files = folder_files(f)
        tinfo[t] = (np.array([y for _, y in files]), np.array([ahash(p) for p, _ in files]))
        print(t, len(files), "unique hashes", len(set(tinfo[t][1])), flush=True)
    rows = []
    for bb in BACKBONES:
        Xs_raw = np.load(os.path.join(a.emb, f"{bb}_fer_train_6000.npy"))
        # PCA fitted on source only keeps ~576 probe fits CPU-tractable
        pca = PCA(a.pca, random_state=0).fit(StandardScaler().fit_transform(Xs_raw))
        sc0 = StandardScaler().fit(Xs_raw)
        Xs = pca.transform(sc0.transform(Xs_raw))
        for t, (yt, gt) in tinfo.items():
            key = "autism_train" if t == "autism" else t
            Xt = pca.transform(sc0.transform(np.load(os.path.join(a.emb, f"{bb}_{key}_{len(yt)}.npy"))))
            for s in a.seeds:
                pool, te = group_split(gt, yt, s)
                rng = np.random.default_rng(s); pool = rng.permutation(pool)
                base = dict(backbone=bb, target=t, seed=s)
                rows.append({**base, "method": "source_only", "frac": 0.0, **fit_eval(Xs, ys, Xt[te], yt[te])})
                rows.append({**base, "method": "coral", "frac": 0.0,
                             **fit_eval(coral(Xs, Xt[pool]), ys, Xt[te], yt[te])})
                for fr in a.fracs:
                    lab = pool[: max(len(EMO), int(round(fr * len(pool))))]
                    rows.append({**base, "method": "target_only", "frac": fr, "n_lab": len(lab),
                                 **fit_eval(Xt[lab], yt[lab], Xt[te], yt[te])})
                    w = np.r_[np.ones(len(ys)), np.full(len(lab), a.upweight)]
                    rows.append({**base, "method": "combined", "frac": fr, "n_lab": len(lab),
                                 **fit_eval(np.r_[Xs, Xt[lab]], np.r_[ys, yt[lab]], Xt[te], yt[te], w)})
            print(bb, t, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "c2_embed_results.csv"), index=False)
    summ = df.groupby(["target", "backbone", "method", "frac"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(3)
    summ.to_csv(os.path.join(a.out, "c2_embed_summary.csv")); print(summ.to_string())
