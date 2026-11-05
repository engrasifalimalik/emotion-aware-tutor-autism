"""C4 refinement: full-data fusion models + rule-threshold sweep tuned on validation students.

Reuses load/features/evaluate from c4_pilot. Caches features to <out>/features.pkl so the
DKT script and reruns don't re-parse the 3 GB CSV.

Usage: python c4_refine.py --csv <path> --students 100000 --out results/c4_full
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from c4_pilot import load, features, evaluate


def rule_stats(d, mask):
    fail = 1 - d.correct
    return {"coverage": float(mask.mean()), "precision": float(fail[mask].mean()),
            "recall": float(fail[mask].sum() / fail.sum()), "lift": float(fail[mask].mean() / fail.mean())}


def perf_rule(d, acc_t, att_t):
    return (d.acc_prev5 < acc_t) & (d.att_prev5 > att_t)


def emo_rule(d, acc_t, att_t, q_frus, q_conf, acc_e, ref):
    hi = (d.frus_prev > q_frus) | (d.conf_prev > q_conf)
    return perf_rule(d, acc_t, att_t) | ((d.acc_prev5 < acc_e) & hi)


def sweep(train_pool, test, seed):
    # Split the training students again: thresholds are picked on validation, reported on test.
    a, b = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed + 1).split(
        train_pool, groups=train_pool.user_id))
    val = train_pool.iloc[b]
    grid_perf = [(acc, att) for acc in (.4, .5, .6) for att in (1.25, 1.5, 2.0)]
    perf = [(k, rule_stats(val, perf_rule(val, *k))) for k in grid_perf]
    # Best perf-only rule by F1 on validation
    f1 = lambda s: 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"] + 1e-9)
    best_p = max(perf, key=lambda x: f1(x[1]))[0]
    emo = []
    for q in (.6, .7, .8, .9, .95):
        qf, qc = val.frus_prev.quantile(q), val.conf_prev.quantile(q)
        for acc_e in (.4, .5, .6, .7):
            s = rule_stats(val, emo_rule(val, *best_p, qf, qc, acc_e, val))
            emo.append(((q, acc_e, qf, qc), s))
    best_e = max(emo, key=lambda x: f1(x[1]))[0]
    q, acc_e, qf, qc = best_e
    return {
        "perf_only_best": {"params": {"acc5_lt": best_p[0], "att5_gt": best_p[1]},
                           "test": rule_stats(test, perf_rule(test, *best_p))},
        "emotion_aware_best": {"params": {"affect_quantile": q, "acc5_lt_when_affect": acc_e},
                               "test": rule_stats(test, emo_rule(test, *best_p, qf, qc, acc_e, test))},
        "val_grid_emotion": [{"q": k[0], "acc_e": k[1], **v} for k, v in emo],
        "selection": "F1(incorrect) on validation students",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--students", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="results/c4_full")
    a = ap.parse_args(); t = time.time(); os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "features.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        df = features(load(a.csv, a.students, a.seed)); df.to_pickle(cache)
    print(f"rows={len(df)} students={df.user_id.nunique()} t={time.time()-t:.0f}s", flush=True)
    models, te = evaluate(df, a.seed)
    train_pool = df[~df.user_id.isin(te.user_id.unique())]
    res = {"n_rows": len(df), "n_students": int(df.user_id.nunique()), "models": models,
           "rules": sweep(train_pool, te, a.seed), "runtime_s": round(time.time() - t)}
    json.dump(res, open(os.path.join(a.out, "c4_full_metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "rules"}, indent=2))
    print(json.dumps({k: v for k, v in res["rules"].items() if k != "val_grid_emotion"}, indent=2))
