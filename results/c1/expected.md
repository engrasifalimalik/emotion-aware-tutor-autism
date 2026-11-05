# C1: Expected Full-Scale Results

> **ILLUSTRATIVE – literature-based estimate, not an actual run.** These numbers are extrapolated from the real CPU pilot (frozen backbones, 6k training images) plus commonly reported published ranges. Verify every cited number before using it.

## In-domain (FER2013 test)
| Setting | Expected acc | Basis |
|---|---|---|
| Frozen linear probe, full 28.7k training images | +2–4 pts over the pilot (ViT-S ≈ 0.55–0.58) | Usual data-scaling gain for a linear probe |
| CLIP ViT-B/16 linear probe | ≈ 0.65–0.70 | CLIP paper reports FER2013 linear-probe accuracy in this range (verify) |
| Full fine-tune MobileNetV3 / EfficientNet-B0 | ≈ 0.66–0.70 | Typical GPU fine-tuning reports |
| Full fine-tune ResNet-50 / ViT-B | ≈ 0.70–0.73 | Typical reports; the human-level accuracy on FER2013 is about 65 ± 5% (verify) |
| State of the art (ensembles, VGG variants) | ≈ 0.73–0.76 | Published best results (verify) |

## Cross-domain (once the gated data is available)
| Train → Test | Expected acc drop vs in-domain | Basis |
|---|---|---|
| Adult (FER/RAF) → posed child (CAFE) | −10 to −20 pts | Posed child expressions are fairly prototypical, but faces differ in morphology |
| Adult → spontaneous child (LIRIS-CSE, EmoReact) | −20 to −30 pts | Subtle, spontaneous expressions |
| Adult → autistic children (DE-ENIGMA) | −25 to −40 pts; macro-F1 may fall to ≈ 0.20–0.35 | Atypical or flat expressions, label ambiguity; mostly neutral-dominated |

**What to compare with your run:** the ranking ViT > CNNs should hold for a frozen probe. After full fine-tuning the gap usually narrows to about 2–4 pts. If CNNs beat ViT after fine-tuning on small data, that is also plausible, because ViTs need more data.
