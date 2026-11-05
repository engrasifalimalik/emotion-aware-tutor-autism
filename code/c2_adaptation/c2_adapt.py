"""C2: data-efficient adaptation of FER models from adult (source) to child/autistic (target) faces.

Strategies (--strategy):
  probe  - freeze backbone, train linear head
  lastk  - unfreeze the last K blocks + head (--k)
  lora   - LoRA adapters on ViT attention qkv/proj (rank --r), backbone frozen
  full   - full fine-tuning
  dann   - full fine-tuning + domain-adversarial loss on unlabeled target images (gradient reversal)
Loss: cross-entropy with class weights, or focal (--loss focal).
Training data = all source labels + --target_frac of target labels (learning curve: 0, .05, .1, .25, .5).
Evaluation: held-out target test split (and source test).

Data:
  --fer   folder with FER2013 parquet files (train.parquet, privateTest.parquet)
  --target folder with <emotion>/*.jpg (angry, disgust, fear, happy, sad, surprise, neutral).
           If omitted, FER2013 publicTest is used as a stand-in target (pipeline check only).

GPU (Kaggle T4): python c2_adapt.py --fer ... --target ... --backbone vit_small_patch16_224 --strategy lora --epochs 8
CPU pilot:      python c2_adapt.py --fer ... --backbone mobilenetv3_large_100 --img 112 --n_src 4000 --epochs 2
"""
import argparse, glob, io, json, math, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ---------------- data ----------------
def fer(folder, split, n, seed):
    d = pd.read_parquet(os.path.join(folder, f"{split}.parquet"))
    if n and n < len(d):
        d = d.sample(n, random_state=seed)
    return [b["bytes"] for b in d.image], d.label.values.astype(np.int64)


ALIASES = {"neutral": ["neutral", "Natural"], "happy": ["happy", "joy"], "sad": ["sad", "sadness"],
           "angry": ["angry", "anger"]}


def folder(path):
    xs, ys = [], []
    for i, e in enumerate(EMO):
        for name in ALIASES.get(e, [e]):
            for p in sorted(glob.glob(os.path.join(path, name, "*"))):
                xs.append(p); ys.append(i)
    return xs, np.array(ys, np.int64)


def ahash(x):
    img = Image.open(io.BytesIO(x) if isinstance(x, bytes) else x).convert("L").resize((8, 8))
    px = np.asarray(img, dtype=float).ravel()
    return "".join("1" if v > px.mean() else "0" for v in px)


class Faces(Dataset):
    def __init__(self, xs, ys, tf): self.xs, self.ys, self.tf = xs, ys, tf
    def __len__(self): return len(self.xs)
    def __getitem__(self, i):
        x = self.xs[i]
        img = Image.open(io.BytesIO(x) if isinstance(x, bytes) else x).convert("RGB")
        return self.tf(img), self.ys[i]


def split_target(xs, ys, frac, seed):
    """70% train pool / 30% test, grouped by perceptual hash so near-duplicates never cross the split;
    labeled = frac of the pool, rest unlabeled."""
    rng = np.random.default_rng(seed)
    g = np.array([ahash(x) for x in xs]); ug = np.unique(g); rng.shuffle(ug)
    test_g = set(ug[: int(.3 * len(ug))])
    is_te = np.array([h in test_g for h in g])
    tr, te = np.where(~is_te)[0], np.where(is_te)[0]
    tr = rng.permutation(tr); n_lab = int(round(frac * len(tr)))
    pick = lambda I: ([xs[i] for i in I], ys[I])
    return pick(tr[:n_lab]), pick(tr[n_lab:]), pick(np.array(te))


# ---------------- model ----------------
class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x): return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None


class Net(nn.Module):
    def __init__(self, name, strategy, k, r):
        super().__init__()
        self.bb = timm.create_model(name, pretrained=True, num_classes=0)
        with torch.no_grad():   # num_features can differ from the pooled output (e.g. MobileNetV3 head)
            d = self.bb.eval()(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.head = nn.Linear(d, len(EMO))
        self.dom = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 2)) if strategy == "dann" else None
        if strategy in ("probe", "lastk", "lora"):
            for p in self.bb.parameters(): p.requires_grad = False
        if strategy == "lastk":
            blocks = getattr(self.bb, "blocks", None) or getattr(self.bb, "stages", None) or \
                     [m for n, m in self.bb.named_children() if n.startswith("layer")]
            for b in list(blocks)[-k:]:
                for p in b.parameters(): p.requires_grad = True
        if strategy == "lora":
            assert hasattr(self.bb, "blocks"), "LoRA here is implemented for ViT backbones"
            for blk in self.bb.blocks:
                blk.attn.qkv = LoRA(blk.attn.qkv, r); blk.attn.proj = LoRA(blk.attn.proj, r)

    def forward(self, x, lam=0.0):
        f = self.bb(x)
        return self.head(f), (self.dom(GradRev.apply(f, lam)) if self.dom is not None else None)


def focal(logits, y, w, gamma=2.0):
    ce = F.cross_entropy(logits, y, weight=w, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()


@torch.no_grad()
def evaluate(m, dl, dev):
    m.eval(); P, Y = [], []
    for x, y in dl:
        P.append(m(x.to(dev))[0].argmax(1).cpu()); Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return {"acc": float(accuracy_score(Y, P)), "macro_f1": float(f1_score(Y, P, average="macro")), "n": int(len(Y))}


def main(a):
    torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Net(a.backbone, a.strategy, a.k, a.r).to(dev)
    cfg = timm.data.resolve_data_config({}, model=m.bb); cfg["input_size"] = (3, a.img, a.img)
    tf_tr = timm.data.create_transform(**cfg, is_training=True, hflip=0.5, auto_augment=None)
    tf_te = timm.data.create_transform(**cfg)

    sx, sy = fer(a.fer, "train", a.n_src, a.seed)
    stx, sty = fer(a.fer, "privateTest", a.n_test, a.seed)
    txs, tys = folder(a.target) if a.target else fer(a.fer, "publicTest", 0, a.seed)
    (lx, ly), (ux, uy), (tex, tey) = split_target(list(txs), tys, a.target_frac, a.seed)

    tr_x, tr_y = list(sx) + lx, np.concatenate([sy, ly])
    counts = np.bincount(tr_y, minlength=len(EMO)) + 1
    w = torch.tensor(len(tr_y) / (len(EMO) * counts), dtype=torch.float32, device=dev)
    # Up-weight scarce target labels so they aren't drowned out by the source set.
    samp_w = np.r_[np.ones(len(sy)), np.full(len(ly), a.target_upweight)]
    sampler = torch.utils.data.WeightedRandomSampler(samp_w, len(samp_w), replacement=True)
    kw = dict(num_workers=a.workers, pin_memory=dev == "cuda")
    dl_tr = DataLoader(Faces(tr_x, tr_y, tf_tr), a.bs, sampler=sampler, drop_last=True, **kw)
    dl_u = DataLoader(Faces(ux + lx, np.r_[uy, ly], tf_tr), a.bs, shuffle=True, drop_last=True, **kw) \
        if a.strategy == "dann" else None
    dl_te = DataLoader(Faces(tex, tey, tf_te), a.bs * 2, **kw)
    dl_ste = DataLoader(Faces(list(stx), sty, tf_te), a.bs * 2, **kw)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    print(f"dev={dev} trainable={sum(p.numel() for p in params)/1e6:.2f}M src={len(sy)} "
          f"tgt_lab={len(ly)} tgt_unlab={len(uy)} tgt_test={len(tey)} steps={steps}", flush=True)

    t0, step, u_iter = time.time(), 0, iter(dl_u) if dl_u else None
    for ep in range(a.epochs):
        m.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                lam = 2 / (1 + math.exp(-10 * step / steps)) - 1     # DANN schedule
                logit, d_s = m(x, lam)
                loss = focal(logit, y, w) if a.loss == "focal" else F.cross_entropy(logit, y, weight=w)
                if dl_u:
                    try: xu, _ = next(u_iter)
                    except StopIteration: u_iter = iter(dl_u); xu, _ = next(u_iter)
                    _, d_t = m(xu.to(dev), lam)
                    dl = torch.cat([d_s, d_t]); dy = torch.cat([torch.zeros(len(d_s)), torch.ones(len(d_t))]).long().to(dev)
                    loss = loss + a.dann_w * F.cross_entropy(dl, dy)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"  ep{ep+1} step {step}/{steps} loss={loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    res = {"args": vars(a), "trainable_m": sum(p.numel() for p in params) / 1e6, "train_s": round(time.time() - t0),
           "target_test": evaluate(m, dl_te, dev), "source_test": evaluate(m, dl_ste, dev)}
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.backbone}_{a.strategy}_tf{a.target_frac}_s{a.seed}"
    json.dump(res, open(os.path.join(a.out, f"c2_{tag}.json"), "w"), indent=2)
    print(tag, json.dumps({k: res[k] for k in ("target_test", "source_test", "train_s")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True); ap.add_argument("--target", default=None)
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--strategy", default="full", choices=["probe", "lastk", "lora", "full", "dann"])
    ap.add_argument("--target_frac", type=float, default=0.1); ap.add_argument("--target_upweight", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"]); ap.add_argument("--dann_w", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--n_src", type=int, default=0); ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/c2")
    main(ap.parse_args())
