# C1: Pilot Results (REAL RUN, CPU)

- **Setup:** frozen ImageNet backbone (timm) → 224 px embeddings → class-balanced logistic regression.
- **Training data:** a stratified subset of **6,000 FER2013 training images**.
- **Test data:** the full FER2013 **privateTest (3,589 images)**. Seed 42.
- **Script and output:** `code/c1_benchmark/c1_pilot.py` (~32 min total), `results/c1/c1_pilot_metrics.json`.
- **Data source:** FER2013 from the open Hugging Face mirror `Aaryan333/fer2013_train_publicTest_privateTest` (parquet files).

## In-domain benchmark (FER2013 privateTest, 7 classes; chance = 14.3%, majority class = 24.5%)
| Backbone | Params (M) | CPU FPS | Accuracy | Macro-F1 |
|---|---|---|---|---|
| MobileNetV3-L | 4.2 | **29.2** | 0.442 | 0.416 |
| EfficientNet-B0 | 4.0 | 26.0 | 0.429 | 0.405 |
| ResNet-50 | 23.5 | 8.0 | 0.417 | 0.389 |
| **ViT-S/16** | 21.7 | 10.8 | **0.534** | **0.488** |

## Per-class recall
| Backbone | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| MobileNetV3 | .31 | .35 | .33 | .66 | .33 | .53 | .40 |
| EfficientNet-B0 | .32 | .33 | .29 | .62 | .33 | .54 | .39 |
| ResNet-50 | .32 | .29 | .30 | .62 | .31 | .50 | .38 |
| ViT-S | .42 | .46 | .32 | **.77** | .38 | .67 | .54 |

## Findings
1. **ViT-S is clearly the best frozen feature extractor** (+9 pts accuracy over the CNNs). Its self-attention features transfer better to faces than ImageNet CNN features do.
2. **Larger CNNs did not help.** ResNet-50 scored lower than MobileNetV3 when the backbone is frozen, because ImageNet object features are not tuned for facial expressions.
3. **Class difficulty matches the literature.** Happy and surprise are the easiest classes; fear, sad and angry are the hardest. This matters for the tutor, because frustration and confusion look most like the hard classes.
4. **Real-time feasibility:** MobileNetV3 runs at about 29 FPS on this CPU, which is enough for a live tutor webcam at 5–10 FPS.
5. **Absolute accuracy is low** because the backbones are frozen and only 6k training images were used. Full fine-tuning on a GPU is expected to reach about 65–73% (see `expected.md`).

## Not yet done (needs gated data)
The cross-domain tests (adult → child → autistic) need CAFE, DE-ENIGMA and similar datasets in `data/<name>/<emotion>/*.jpg`. Then rerun:
`python code/c1_benchmark/c1_pilot.py --fer data/fer2013 --n_train 6000 --target cafe=data/cafe --target deenigma=data/deenigma`
The cached FER embeddings are reused, so only the new images are processed.
