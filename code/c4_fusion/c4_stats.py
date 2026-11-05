"""C4 robustness: 5 student-level splits (seeds) x {XGBoost, DKT} x {cognitive, fused},
plus a student-clustered paired bootstrap of the AUC gain (fused - cognitive) and of the
rule-engine recall gain (emotion-aware - perf-only) on each seed's test set.

Usage: python c4_stats.py --features results/c4_full/features.pkl --seeds 0 1 2 3 4 --boot 300
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from c4_pilot import COG, AFFF
from c4_refine import sweep, perf_rule, emo_rule
import c4_dkt


def xgb_preds(tr, te, cols, seed):
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      n_jobs=-1, random_state=seed)
    return m.fit(tr[cols], tr.correct).predict_proba(te[cols])[:, 1]


def cluster_boot(users, stat, n, seed):
    """Resample students with replacement; stat(idx) -> float. Returns mean, 95% CI, P(<=0)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(users)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    vals = np.array([stat(np.concatenate([groups[g] for g in rng.integers(0, len(uniq), len(uniq))]))
                     for _ in range(n)])
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "p_le_0": float((vals <= 0).mean())}


def dkt_users(df_te, L):
    # Mirrors c4_dkt.sequences order: one prediction per step after the first in each chunk.
    out = []
    for u, g in df_te.groupby("user_id", sort=False):
        for i in range(0, len(g) - 1, L):
            j = min(i + L + 1, len(g))
            if j - i >= 2:
                out.append(np.full(j - i - 1, u))
    return np.concatenate(out)


def one_seed(df, seed, n_boot, epochs, L):
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(df, groups=df.user_id))
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    y, users = te.correct.values, te.user_id.values
    r = {}
    # XGBoost
    pc, pf = xgb_preds(tr, te, COG, seed), xgb_preds(tr, te, COG + AFFF, seed)
    r["xgb_cog_auc"], r["xgb_fused_auc"] = roc_auc_score(y, pc), roc_auc_score(y, pf)
    r["xgb_gain_boot"] = cluster_boot(users, lambda i: roc_auc_score(y[i], pf[i]) - roc_auc_score(y[i], pc[i]),
                                      n_boot, seed)
    # DKT
    train, test = c4_dkt.sequences(tr, L), c4_dkt.sequences(te, L)
    n_sk = int(df.sk.max()) + 1
    P0, Y0 = c4_dkt.run(train, test, n_sk, False, epochs, seed, return_preds=True)
    P1, _ = c4_dkt.run(train, test, n_sk, True, epochs, seed, return_preds=True)
    du = dkt_users(te, L)
    r["dkt_auc"], r["dkt_affect_auc"] = roc_auc_score(Y0, P0), roc_auc_score(Y0, P1)
    r["dkt_gain_boot"] = cluster_boot(du, lambda i: roc_auc_score(Y0[i], P1[i]) - roc_auc_score(Y0[i], P0[i]),
                                      n_boot, seed)
    # Rules (thresholds tuned on validation students from train)
    sw = sweep(tr, te, seed)
    p = sw["perf_only_best"]["params"]; e = sw["emotion_aware_best"]["params"]
    val_q = e["affect_quantile"]
    qf, qc = tr.frus_prev.quantile(val_q), tr.conf_prev.quantile(val_q)
    fail = 1 - y
    mp = perf_rule(te, p["acc5_lt"], p["att5_gt"]).values
    me = emo_rule(te, p["acc5_lt"], p["att5_gt"], qf, qc, e["acc5_lt_when_affect"], te).values
    rec = lambda m, i: fail[i][m[i]].sum() / fail[i].sum()
    prec = lambda m, i: fail[i][m[i]].mean()
    allidx = np.arange(len(y))
    r["rule_perf_recall"], r["rule_emo_recall"] = float(rec(mp, allidx)), float(rec(me, allidx))
    r["rule_perf_precision"], r["rule_emo_precision"] = float(prec(mp, allidx)), float(prec(me, allidx))
    r["rule_params"] = {"perf": p, "emo": e}
    r["rule_recall_gain_boot"] = cluster_boot(users, lambda i: rec(me, i) - rec(mp, i), n_boot, seed)
    r["rule_precision_change_boot"] = cluster_boot(users, lambda i: prec(me, i) - prec(mp, i), n_boot, seed)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--boot", type=int, default=300); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq_len", type=int, default=100); ap.add_argument("--out", default="results/c4_full")
    a = ap.parse_args(); t0 = time.time()
    df = pd.read_pickle(a.features)
    df["sk"] = df.skill_id.fillna(-1).astype("category").cat.codes.astype(np.int64)
    per_seed = {}
    for s in a.seeds:
        t = time.time(); per_seed[s] = one_seed(df, s, a.boot, a.epochs, a.seq_len)
        json.dump(per_seed, open(os.path.join(a.out, "c4_stats_partial.json"), "w"), indent=2)
        print(f"seed {s} done in {time.time()-t:.0f}s: " + json.dumps(
            {k: round(v, 4) for k, v in per_seed[s].items() if isinstance(v, float)}), flush=True)
    keys = [k for k, v in per_seed[a.seeds[0]].items() if isinstance(v, float)]
    summary = {k: {"mean": float(np.mean([per_seed[s][k] for s in a.seeds])),
                   "std": float(np.std([per_seed[s][k] for s in a.seeds], ddof=1))} for k in keys}
    for k in ("xgb_fused_auc", "dkt_affect_auc", "rule_emo_recall", "rule_emo_precision"):
        base = {"xgb_fused_auc": "xgb_cog_auc", "dkt_affect_auc": "dkt_auc",
                "rule_emo_recall": "rule_perf_recall", "rule_emo_precision": "rule_perf_precision"}[k]
        d = [per_seed[s][k] - per_seed[s][base] for s in a.seeds]
        summary[f"gain_{k}"] = {"mean": float(np.mean(d)), "std": float(np.std(d, ddof=1)),
                                "seeds_positive": int(sum(x > 0 for x in d)), "n_seeds": len(d)}
    res = {"per_seed": per_seed, "summary": summary, "n_boot": a.boot, "runtime_s": round(time.time() - t0)}
    json.dump(res, open(os.path.join(a.out, "c4_stats_metrics.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
