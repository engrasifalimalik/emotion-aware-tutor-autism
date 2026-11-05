"""C3 step 2: map facial features to learning-centred states on DAiSEE.

Binary tasks (positive class): bored (Boredom>=1), confused (Confusion>=1), frustrated (Frustration>=1),
low_engaged (Engagement<=1). Plus 4-level engagement (accuracy, for comparison with the literature).
Feature sets: emo (7 FER emotion probabilities), emb (frozen ViT-S mean+std), emo+emb.
Hyper-parameters and decision thresholds are selected on Validation; metrics reported on Test.
Bootstrap 95% CIs over test clips. Also exports the emotion->state logistic coefficients.

Usage: python c3_classify.py --feat results/c3/features --out results/c3
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TASKS = {"bored": lambda L: L[:, 0] >= 1, "low_engaged": lambda L: L[:, 1] <= 1,
         "confused": lambda L: L[:, 2] >= 1, "frustrated": lambda L: L[:, 3] >= 1}


def load(d, s):
    z = np.load(os.path.join(d, f"{s}.npz"), allow_pickle=True)
    return {"emo": np.log(z["emo"] + 1e-4), "emb": z["emb"], "labels": z["labels"]}


def feats(D, kind, pca):
    if kind == "emo": return D["emo"]
    e = pca.transform(D["emb"])
    return e if kind == "emb" else np.c_[e, D["emo"] * 3]


def boot(y, p, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed); A, F = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        A.append(roc_auc_score(y[i], p[i])); F.append(f1_score(y[i], p[i] >= thr, average="macro"))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(A), q(F)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="results/c3/features"); ap.add_argument("--out", default="results/c3")
    a = ap.parse_args()
    tr, va, te = (load(a.feat, s) for s in ("Train", "Validation", "Test"))
    pca = make_pipeline(StandardScaler(), PCA(min(128, len(tr["emb"]) - 1), random_state=0)).fit(tr["emb"])
    res, coefs = {}, {}
    for kind in ("emo", "emb", "emo+emb"):
        Xtr, Xva, Xte = (feats(D, kind, pca) for D in (tr, va, te))
        for task, fn in TASKS.items():
            ytr, yva, yte = (fn(D["labels"]).astype(int) for D in (tr, va, te))
            best = None
            for C in (0.01, 0.1, 1.0):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(Xtr, ytr)
                auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
                if best is None or auc > best[0]: best = (auc, C, m)
            _, C, m = best
            pva = m.predict_proba(Xva)[:, 1]
            thr = max(np.linspace(.05, .95, 91), key=lambda t: f1_score(yva, pva >= t, average="macro"))
            pte = m.predict_proba(Xte)[:, 1]
            ci_auc, ci_f1 = boot(yte, pte, thr)
            res[f"{kind}/{task}"] = {"C": C, "thr": float(thr), "test_pos_rate": float(yte.mean()),
                                     "auc": float(roc_auc_score(yte, pte)), "auc_ci": ci_auc,
                                     "macro_f1": float(f1_score(yte, pte >= thr, average="macro")), "f1_ci": ci_f1,
                                     "majority_macro_f1": float(f1_score(yte, np.full_like(yte, int(ytr.mean() > .5)), average="macro"))}
            if kind == "emo":
                coefs[task] = dict(zip(EMO, m[-1].coef_[0].round(3).tolist()))
            print(kind, task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"{kind}/{task}"].items()}, flush=True)
        # 4-level engagement (multiclass), comparable to DAiSEE literature
        ytr4, yva4, yte4 = (D["labels"][:, 1] for D in (tr, va, te))
        m4 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr, ytr4)
        p4 = m4.predict(Xte)
        res[f"{kind}/engagement_4level"] = {"acc": float(accuracy_score(yte4, p4)),
                                            "balanced_acc": float(balanced_accuracy_score(yte4, p4)),
                                            "majority_acc": float((yte4 == np.bincount(ytr4).argmax()).mean())}
        print(kind, "engagement_4level", res[f"{kind}/engagement_4level"], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"results": res, "emotion_coefficients": coefs,
               "n": {s: int(len(D["labels"])) for s, D in zip(("train", "val", "test"), (tr, va, te))}},
              open(os.path.join(a.out, "c3_metrics.json"), "w"), indent=2)
    print(json.dumps(coefs, indent=1))
