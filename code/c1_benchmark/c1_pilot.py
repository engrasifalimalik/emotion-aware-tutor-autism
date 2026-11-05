"""C1 pilot: FER backbone benchmark with frozen ImageNet embeddings + linear probe (CPU-friendly).

Source: FER2013 (parquet: label, image{bytes}); train subset -> fit probe, privateTest -> in-domain test.
Optional cross-domain targets: --target name=folder, where folder has one subfolder per emotion
(angry, disgust, fear, happy, sad, surprise, neutral), e.g. CAFE or DE-ENIGMA frames once downloaded.
Embeddings are cached per backbone/dataset in <out>/emb/.

Usage: python c1_pilot.py --fer data/fer2013 --n_train 6000 --out results/c1
       [--target cafe=data/cafe_faces --target deenigma=data/deenigma_faces]
"""
import argparse, glob, io, json, os, time
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]   # FER2013 label order
BACKBONES = ["mobilenetv3_large_100", "efficientnet_b0", "resnet50", "vit_small_patch16_224"]


def fer_split(folder, split, n, seed, exclude=()):
    d = pd.read_parquet(os.path.join(folder, f"{split}.parquet"))
    d = d.drop(index=[i for i in exclude if i in d.index])   # keep target faces out of training
    if n and n < len(d):   # stratified subsample
        idx = [i for _, g in d.groupby("label")
               for i in g.sample(max(1, round(n * len(g) / len(d))), random_state=seed).index]
        d = d.loc[idx]
    imgs = [Image.open(io.BytesIO(b["bytes"])).convert("RGB") for b in d.image]
    return imgs, d.label.values


ALIASES = {"neutral": ["neutral", "Natural"], "happy": ["happy", "joy"], "sad": ["sad", "sadness"],
           "angry": ["angry", "anger"]}   # folder names used by other datasets (e.g. Talaat autism set)


def folder_split(folder):
    imgs, ys = [], []
    for i, e in enumerate(EMO):
        for name in ALIASES.get(e, [e]):
            for p in sorted(glob.glob(os.path.join(folder, name, "*"))):
                imgs.append(Image.open(p).convert("RGB")); ys.append(i)
    return imgs, np.array(ys)


@torch.no_grad()
def embed(name, imgs, bs=64):
    m = timm.create_model(name, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**cfg)
    out, t = [], time.time()
    for i in range(0, len(imgs), bs):
        out.append(m(torch.stack([tf(x) for x in imgs[i:i + bs]])).numpy())
    fps = len(imgs) / (time.time() - t)
    return np.concatenate(out), fps, sum(p.numel() for p in m.parameters()) / 1e6


def metrics(y, p):
    return {"acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "per_class_recall": dict(zip(EMO, recall_score(y, p, average=None, labels=range(7), zero_division=0).round(3).tolist())),
            "confusion": confusion_matrix(y, p, labels=range(7)).tolist(), "n": int(len(y))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True); ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=0); ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--target", action="append", default=[]); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_csv", default=None); ap.add_argument("--out", default="results/c1")
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "emb"), exist_ok=True); torch.set_num_threads(os.cpu_count())
    excl = []
    if a.exclude_csv:   # age_predictions.csv from make_child_subset.py: drop child/teen train faces
        ag = pd.read_csv(a.exclude_csv)
        excl = ag[(ag.split == "train") & ((ag.group == "child") | (ag.p_teen > 0.3))].idx.tolist()
    data = {"fer_train": fer_split(a.fer, "train", a.n_train, a.seed, excl),
            "fer_test": fer_split(a.fer, "privateTest", a.n_test, a.seed)}
    for t in a.target:
        k, v = t.split("="); data[k] = folder_split(v)
    print({k: len(v[1]) for k, v in data.items()}, flush=True)
    res = {}
    for bb in a.backbones:
        E, info = {}, {}
        for k, (imgs, y) in data.items():
            f = os.path.join(a.out, "emb", f"{bb}_{k}_{len(y)}.npy")
            if os.path.exists(f):
                E[k] = np.load(f)
            else:
                E[k], fps, params = embed(bb, imgs); np.save(f, E[k]); info = {"cpu_fps": fps, "params_m": params}
        sc = StandardScaler().fit(E["fer_train"])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(E["fer_train"]), data["fer_train"][1])
        res[bb] = {**info, **{k: metrics(data[k][1], clf.predict(sc.transform(E[k])))
                              for k in data if k != "fer_train"}}
        print(bb, {k: (round(v["acc"], 3), round(v["macro_f1"], 3)) for k, v in res[bb].items() if isinstance(v, dict)},
              {k: round(v, 1) for k, v in info.items()}, flush=True)
    json.dump({"n": {k: int(len(v[1])) for k, v in data.items()}, "results": res},
              open(os.path.join(a.out, "c1_pilot_metrics.json"), "w"), indent=2)
