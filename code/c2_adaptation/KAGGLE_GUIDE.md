# C2 on Kaggle GPU (T4/P100, free ~30 h/week)

## Setup (first notebook cell)
```
!pip -q install timm
# Upload as Kaggle Datasets: fer2013 parquet folder, and your target (e.g. cafe/<emotion>/*.jpg)
# Upload code/c2_adaptation/c2_adapt.py as a dataset too, or paste it into a cell.
```

## Experiment grid (≈ 12–16 GPU-hours total)
| Run | Command flags | ~Time (T4) |
|---|---|---|
| Probe baseline | `--strategy probe --lr 1e-3` | 15 min |
| Last-2 blocks | `--strategy lastk --k 2 --lr 3e-4` | 30 min |
| LoRA (ViT) | `--strategy lora --r 8 --lr 1e-3` | 35 min |
| Full FT | `--strategy full --lr 5e-5` (ViT) / `1e-3` (CNN) | 45 min |
| DANN | `--strategy dann --lr 5e-5 --dann_w 0.3` | 60 min |
| Focal loss | add `--loss focal` to the best strategy | +45 min |

Common: `--fer /kaggle/input/fer2013 --target /kaggle/input/cafe --backbone vit_small_patch16_224 --epochs 8 --bs 64 --out /kaggle/working/c2`

## Learning curve (the main C2 figure)
For the 2–3 best strategies, loop over `--target_frac 0 0.05 0.1 0.25 0.5` and seeds `42 1 2`.
Plot target macro-F1 against the target-label fraction.

## Notes
- `--target_frac 0` + `dann` is unsupervised adaptation (no target labels).
- The target split is random-stratified because folders carry no subject IDs. If your target data has subject IDs (e.g., DE-ENIGMA child IDs), split by subject instead, or results will be optimistic.
- Download `/kaggle/working/c2/*.json` and paste them back here for analysis.
