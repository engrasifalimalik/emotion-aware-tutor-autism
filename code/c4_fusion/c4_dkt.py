"""C4: Deep Knowledge Tracing (LSTM) baseline vs DKT + affect, on cached C4 features.

Input at step t: embedding of (skill_t, correct_t) [+ 4 affect values of step t].
Target: correctness at t+1, given the embedding of skill_{t+1}.
Uses the same student-level test split as c4_pilot.evaluate (GroupShuffleSplit seed 42, 30%).

Usage: python c4_dkt.py --features results/c4_full/features.pkl --epochs 3 --out results/c4_full
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

AFF = ["frus", "conf", "conc", "bored"]


def sequences(df, L):
    out = []
    for _, g in df.groupby("user_id", sort=False):
        s, c, a = g.sk.values, g.correct.values, g[AFF].values.astype(np.float32)
        for i in range(0, len(g) - 1, L):
            j = min(i + L + 1, len(g))
            if j - i >= 2:
                out.append((s[i:j], c[i:j], a[i:j]))
    return out


def batches(seqs, bs, shuffle, rng):
    idx = rng.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for k in range(0, len(idx), bs):
        chunk = [seqs[i] for i in idx[k:k + bs]]
        T = max(len(s) for s, _, _ in chunk) - 1
        S, C, A, M = (np.zeros((len(chunk), T + 1), np.int64), np.zeros((len(chunk), T + 1), np.int64),
                      np.zeros((len(chunk), T + 1, 4), np.float32), np.zeros((len(chunk), T), bool))
        for b, (s, c, a) in enumerate(chunk):
            n = len(s); S[b, :n], C[b, :n], A[b, :n], M[b, :n - 1] = s, c, a, True
        yield map(torch.from_numpy, (S, C, A, M))


class DKT(nn.Module):
    def __init__(self, n_skills, use_affect, d=64):
        super().__init__()
        self.use_affect = use_affect
        self.inter = nn.Embedding(2 * n_skills, d)
        self.skill = nn.Embedding(n_skills, d)
        self.lstm = nn.LSTM(d + (4 if use_affect else 0), d, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, S, C, A):
        x = self.inter(S[:, :-1] * 2 + C[:, :-1])
        if self.use_affect:
            x = torch.cat([x, A[:, :-1]], -1)
        h, _ = self.lstm(x)
        return self.head(torch.cat([h, self.skill(S[:, 1:])], -1)).squeeze(-1)


def run(train, test, n_skills, use_affect, epochs, seed, return_preds=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DKT(n_skills, use_affect); opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        m.train(); t = time.time()
        for S, C, A, M in batches(train, 64, True, rng):
            loss = lossf(m(S, C, A)[M], C[:, 1:][M].float())
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  affect={use_affect} epoch {ep+1} loss={loss.item():.3f} {time.time()-t:.0f}s", flush=True)
    m.eval(); P, Y = [], []
    with torch.no_grad():
        for S, C, A, M in batches(test, 256, False, rng):
            P.append(torch.sigmoid(m(S, C, A))[M].numpy()); Y.append(C[:, 1:][M].numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    if return_preds:
        return P, Y
    return {"auc": roc_auc_score(Y, P), "acc": accuracy_score(Y, P > .5), "n_test_preds": int(len(Y))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq_len", type=int, default=100); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/c4_full")
    a = ap.parse_args(); t0 = time.time()
    df = pd.read_pickle(a.features)
    df["sk"] = df.skill_id.fillna(-1).astype("category").cat.codes.astype(np.int64)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=a.seed).split(df, groups=df.user_id))
    train, test = sequences(df.iloc[tr], a.seq_len), sequences(df.iloc[te], a.seq_len)
    n_sk = int(df.sk.max()) + 1
    print(f"train seqs={len(train)} test seqs={len(test)} skills={n_sk}", flush=True)
    res = {"dkt": run(train, test, n_sk, False, a.epochs, a.seed),
           "dkt_affect": run(train, test, n_sk, True, a.epochs, a.seed),
           "epochs": a.epochs, "runtime_s": round(time.time() - t0)}
    json.dump(res, open(os.path.join(a.out, "c4_dkt_metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
