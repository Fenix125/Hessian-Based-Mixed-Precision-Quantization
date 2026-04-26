# HAWQ-v2

Hessian-trace-based mixed-precision quantization for Vision Transformers.

This folder is the v2 counterpart of `../hawq/`. It replaces the
top-eigenvalue (power iteration) sensitivity from HAWQ-v1 with the
**Hessian trace estimated by Hutchinson's algorithm**, which is the
defining contribution of the HAWQ-v2 paper (Dong et al., NeurIPS 2019).

## Why the trace

For a model at a local minimum (`grad L = 0`), the second-order Taylor
expansion of the loss is

```
delta L ~= 0.5 * delta_w^T H delta_w
```

Quantization perturbs the weights, so `delta_w` is non-zero. Layers
sitting in steeper curvature suffer more loss increase per unit of
perturbation. v1 used the largest eigenvalue of `H` as a proxy for
"steepness". v2 uses the **trace** instead, which is the sum of all
eigenvalues - it captures the full curvature, not just the worst
direction.

Computing the full Hessian for a transformer is hopeless (millions of
parameters per layer). Hutchinson's algorithm gets around that:

```
Tr(H) ~= E_{v ~ Rademacher} [ v^T H v ]
```

`v^T H v` is just two autograd passes (`g = dL/dW`, then `Hv =
d(g . v)/dW`), so we never materialise `H`.

## Files

- `quantizer.py` - `FakeQuantizer`, `QLinear`, `replace_linear_with_qlinear`. Same fake-quant primitives as v1.
- `analyzer_v2.py` - `ViTHAWQv2Analyzer`. Runs Hutchinson trace per layer and returns sensitivity scores.
- `bit_allocator.py` - `HAWQv2BitAllocator`. Three modes: `allocate_uniform`, `allocate_by_rank`, `allocate_by_budget` (the slider engine).
- `ptq.py` - applies bit assignments, measures effective model size, evaluates top-1.
- `data.py` - Tiny-ImageNet loader (HuggingFace `zh-plus/tiny-imagenet`).
- `experiment.ipynb` - end-to-end FP32 vs Uniform vs HAWQ-v2 comparison.
- `slider_demo.ipynb` - interactive `ipywidgets` slider over the bit-width budget.

## How to run

```bash
pip install -r ../requirements.txt
pip install datasets ipywidgets matplotlib pandas
```

1. `experiment.ipynb` - runs the full pipeline. Generates `checkpoint.pt`,
   `sensitivities.json`, `results.csv`, `tradeoff.png`, `size_vs_acc.png`.
2. `slider_demo.ipynb` - reads the artefacts above and exposes the slider.

## Setup at a glance

- **Dataset**: Tiny-ImageNet (200 classes, 64x64 -> 224x224 resize).
- **Model**: `facebook/deit-small-patch16-224` (22M params, ImageNet pretrained, head re-init'd to 200 classes and linear-probed for 2 epochs).
- **Quantization**: weight-only PTQ. Activation quantization is left off so the comparison is purely about weight compression (this is the standard PTQ setting in the HAWQ papers).
- **Bit grid**: candidate bit-widths {2, 4, 6, 8}. Classification head stays in FP32.
- **Metrics**: Top-1 accuracy on Tiny-ImageNet val and effective model size (MB), where size counts each weight at its assigned bit-width and everything else at FP32.

## Method comparison reported

| Method | Avg bits | Description |
|---|---|---|
| FP32 | 32 | No quantization |
| Uniform-{2,4,6,8} | 2 / 4 / 6 / 8 | Same bit-width on every layer (the "PyTorch out-of-the-box" baseline) |
| HAWQ-v2 budget-{3..7} | varies | Hutchinson-trace driven, parameter-weighted average ~= budget |

The slider in `slider_demo.ipynb` extends the HAWQ-v2 row across a fine
grid of budgets so the accuracy/size tradeoff curve is visible
interactively.
