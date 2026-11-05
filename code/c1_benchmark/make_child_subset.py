"""Build an age-filtered child subset of FER2013 using an open age classifier
(nateraw/vit-age-classifier, classes 0-2, 3-9, 10-19, 20-29, ...).

Child = predicted '0-2' or '3-9' with P(child) >= --thr. Adult = P(20+) >= --thr.
Writes <out>/{child,adult}/<emotion>/<split>_<idx>.png and age_predictions.csv.

Usage: python make_child_subset.py --fer data/fer2013 --out data/fer2013_age --thr 0.9
"""
import argparse, io, os
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True); ap.add_argument("--out", default="data/fer2013_age")
    ap.add_argument("--thr", type=float, default=0.9); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args(); torch.set_num_threads(os.cpu_count())
    proc = ViTImageProcessor.from_pretrained("nateraw/vit-age-classifier")
    m = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier").eval()
    rows = []
    for split in ("train", "publicTest", "privateTest"):
        d = pd.read_parquet(os.path.join(a.fer, f"{split}.parquet"))
        imgs = [Image.open(io.BytesIO(b["bytes"])).convert("RGB") for b in d.image]
        P = []
        with torch.no_grad():
            for i in range(0, len(imgs), a.bs):
                P.append(m(**proc(imgs[i:i + a.bs], return_tensors="pt")).logits.softmax(-1).numpy())
                if i % (a.bs * 50) == 0: print(split, i, len(imgs), flush=True)
        P = np.concatenate(P)
        for i, (p, y) in enumerate(zip(P, d.label.values)):
            pc, pa = p[0] + p[1], p[3:].sum()
            grp = "child" if pc >= a.thr else "adult" if pa >= a.thr else None
            rows.append({"split": split, "idx": i, "label": EMO[y], "p_child": pc, "p_teen": p[2],
                         "p_adult": pa, "group": grp})
            if grp:
                dst = os.path.join(a.out, grp, EMO[y]); os.makedirs(dst, exist_ok=True)
                imgs[i].save(os.path.join(dst, f"{split}_{i}.png"))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "age_predictions.csv"), index=False)
    print(df.groupby(["group", "label"]).size().unstack(0))
