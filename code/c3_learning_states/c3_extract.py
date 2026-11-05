"""C3 step 1: DAiSEE clip features.

For each labelled clip: sample --frames frames uniformly, detect the largest face (OpenCV Haar;
fall back to a centre crop), embed with a frozen timm backbone, and mean+std pool over frames.
Also stores per-frame-mean emotion probabilities from the C1 FER probe (adult FER2013, child-free),
so C3 can test emotion-only features as an interpretable bridge to learning states.

Outputs <out>/{split}.npz with clip_ids, emb (2*D), emo (7), labels (B, E, C, F).

Usage: python c3_extract.py --daisee data/daisee/DAiSEE --out results/c3/features --frames 4
"""
import argparse, glob, os, sys, time
import cv2, numpy as np, pandas as pd, timm, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "c1_benchmark"))
from c1_pilot import EMO

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def frames(path, k):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    out = []
    for t in np.linspace(0, n - 1, k + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = cap.read()
        if ok: out.append(f)
    cap.release(); return out


def face(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    det = CASCADE.detectMultiScale(g, 1.2, 5, minSize=(60, 60))
    if len(det):
        x, y, w, h = max(det, key=lambda d: d[2] * d[3]); m = int(.15 * w)
        f = f[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        hit = True
    else:
        H, W = g.shape; s = min(H, W); f = f[(H - s) // 2:(H + s) // 2, (W - s) // 2:(W + s) // 2]; hit = False
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), hit


def fer_probe(emb_dir, bb):
    """Re-fit the C1 cross-domain probe (adult FER2013, children excluded) from cached embeddings."""
    ag = pd.read_csv("data/fer2013_age/age_predictions.csv")
    ex = ag[(ag.split == "train") & ((ag.group == "child") | (ag.p_teen > .3))].idx.tolist()
    d = pd.read_parquet("data/fer2013/train.parquet").drop(index=ex)
    idx = [i for _, g in d.groupby("label") for i in g.sample(max(1, round(6000 * len(g) / len(d))), random_state=42).index]
    X = np.load(os.path.join(emb_dir, f"{bb}_fer_train_6000.npy"))
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=2000, C=.5, class_weight="balanced").fit(sc.transform(X), d.loc[idx].label.values)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daisee", default="data/daisee/DAiSEE"); ap.add_argument("--out", default="results/c3/features")
    ap.add_argument("--backbone", default="vit_small_patch16_224"); ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--fer_emb", default="results/c1_cross/emb"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(os.cpu_count())
    m = timm.create_model(a.backbone, pretrained=True, num_classes=0).eval()
    tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
    sc, probe = fer_probe(a.fer_emb, a.backbone)
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(a.daisee, "DataSet", "*", "*", "*", "*.*"))}
    for split in ("Train", "Validation", "Test"):
        lab = pd.read_csv(os.path.join(a.daisee, "Labels", f"{split}Labels.csv")); lab.columns = [c.strip() for c in lab.columns]
        lab = lab[lab.ClipID.isin(paths)]
        if a.limit: lab = lab.head(a.limit)
        E, P, hits, t0 = [], [], 0, time.time()
        for i, cid in enumerate(lab.ClipID):
            fs = [face(f) for f in frames(paths[cid], a.frames)]
            hits += sum(h for _, h in fs)
            with torch.no_grad():
                z = m(torch.stack([tf(im) for im, _ in fs])).numpy()
            E.append(np.r_[z.mean(0), z.std(0)]); P.append(probe.predict_proba(sc.transform(z)).mean(0))
            if i % 200 == 0:
                print(f"{split} {i}/{len(lab)} {time.time()-t0:.0f}s face_rate={hits/max(1,(i+1)*a.frames):.2f}", flush=True)
        np.savez(os.path.join(a.out, f"{split}.npz"), clip_ids=lab.ClipID.values, emb=np.array(E), emo=np.array(P),
                 labels=lab[["Boredom", "Engagement", "Confusion", "Frustration"]].values)
        print(f"{split} saved {len(lab)} clips, face_rate={hits/max(1,len(lab)*a.frames):.2f}", flush=True)
