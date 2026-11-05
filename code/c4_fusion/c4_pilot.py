"""C4 pilot: affect-cognition fusion on ASSISTments 2012-13 (with affect predictions).

Task: predict first-attempt correctness of a student's NEXT problem from the history
before it. Feature sets: cognitive-only, affect-only, fused. Split is by student.
Also evaluates a rule-based engine offline: do rule triggers flag upcoming failure?

Usage: python c4_pilot.py --csv <path> --students 4000 --out results/c4
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

AFF = {"Average_confidence(FRUSTRATED)": "frus", "Average_confidence(CONFUSED)": "conf",
       "Average_confidence(CONCENTRATING)": "conc", "Average_confidence(BORED)": "bored"}
COLS = ["user_id", "skill_id", "start_time", "original", "correct", "hint_count",
        "attempt_count", "ms_first_response", *AFF]


def load(csv, n_students, seed):
    # Pass 1: pick a random student subset; pass 2: read only their rows, in chunks.
    ids = pd.read_csv(csv, usecols=["user_id"])["user_id"].unique()
    keep = set(np.random.default_rng(seed).choice(ids, min(n_students, len(ids)), replace=False))
    parts = [c[c.user_id.isin(keep)] for c in pd.read_csv(csv, usecols=COLS, chunksize=500_000)]
    df = pd.concat(parts).rename(columns=AFF)
    df = df[df.original == 1].copy()                      # main problems only
    df["correct"] = (df.correct >= 1).astype(int)          # binarise partial credit
    df["start_time"] = pd.to_datetime(df.start_time, errors="coerce")
    return df.dropna(subset=["start_time"]).sort_values(["user_id", "start_time"])


def features(df):
    g = df.groupby("user_id")
    s = lambda c: g[c].shift(1)                           # strictly past information
    df["n_prev"] = g.cumcount()
    df["acc_prev_all"] = g["correct"].transform(lambda x: x.shift(1).expanding().mean())
    df["acc_prev5"] = g["correct"].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df["att_prev5"] = g["attempt_count"].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df["hint_prev5"] = g["hint_count"].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df["rt_prev"] = np.log1p(s("ms_first_response").clip(0, 6e5))
    df["skill_prev_acc"] = df.groupby(["user_id", "skill_id"])["correct"].transform(
        lambda x: x.shift(1).expanding().mean())
    for a in AFF.values():
        df[f"{a}_prev"] = s(a)
        df[f"{a}_prev5"] = g[a].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df = df[df.n_prev >= 3].fillna({"skill_prev_acc": -1})
    return df.dropna()


COG = ["n_prev", "acc_prev_all", "acc_prev5", "att_prev5", "hint_prev5", "rt_prev", "skill_prev_acc"]
AFFF = [f"{a}{s}" for a in AFF.values() for s in ("_prev", "_prev5")]


def evaluate(df, seed):
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(df, groups=df.user_id))
    tr, te = df.iloc[tr], df.iloc[te]
    out = {}
    for name, cols in {"cognitive": COG, "affect": AFFF, "fused": COG + AFFF}.items():
        for mname, m in {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                         "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                  subsample=0.8, n_jobs=-1, random_state=seed)}.items():
            m.fit(tr[cols], tr.correct)
            p = m.predict_proba(te[cols])[:, 1]
            out[f"{name}/{mname}"] = {"auc": roc_auc_score(te.correct, p),
                                      "acc": accuracy_score(te.correct, p > .5),
                                      "f1_incorrect": f1_score(1 - te.correct, p <= .5)}
    out["majority_baseline_acc"] = float(max(te.correct.mean(), 1 - te.correct.mean()))
    return out, te


def rules(te):
    """Offline rule engine: P(next incorrect | rule fires) vs base rate."""
    base = 1 - te.correct.mean()
    r = {"perf_only (acc5<0.5 & att5>1.5)": (te.acc_prev5 < .5) & (te.att_prev5 > 1.5)}
    hi = lambda a: te[f"{a}_prev"] > te[f"{a}_prev"].quantile(.8)
    r["affect_only (frustrated or confused high)"] = hi("frus") | hi("conf")
    r["emotion_aware (perf rule OR (acc5<0.6 & frus/conf high))"] = r["perf_only (acc5<0.5 & att5>1.5)"] | (
        (te.acc_prev5 < .6) & (hi("frus") | hi("conf")))
    res = {"base_rate_incorrect": float(base)}
    for k, m in r.items():
        fail = 1 - te.correct
        res[k] = {"coverage": float(m.mean()), "precision": float(fail[m].mean()),
                  "recall": float(fail[m].sum() / fail.sum()), "lift": float(fail[m].mean() / base)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--students", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="results/c4")
    a = ap.parse_args(); t = time.time(); os.makedirs(a.out, exist_ok=True)
    df = features(load(a.csv, a.students, a.seed))
    print(f"rows={len(df)} students={df.user_id.nunique()} load+feat={time.time()-t:.0f}s")
    models, te = evaluate(df, a.seed)
    res = {"n_rows": len(df), "n_students": int(df.user_id.nunique()),
           "models": models, "rules": rules(te), "runtime_s": round(time.time() - t)}
    json.dump(res, open(os.path.join(a.out, "c4_pilot_metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
