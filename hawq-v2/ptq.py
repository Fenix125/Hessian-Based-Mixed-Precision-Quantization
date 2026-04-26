"""
Post-training quantization helpers for HAWQ-v2.

After the analyzer has produced sensitivities and the allocator has handed
out per-layer bit-widths, we still need to (a) push those bit-widths into
the FakeQuantizers, (b) measure how big the resulting model would be in a
true integer deployment, and (c) evaluate top-1 accuracy on the validation
set.

Nothing in this file trains anything - it's pure PTQ. We rely on the
QLinear modules already being in place (call replace_linear_with_qlinear
first).
"""

import torch
import torch.nn as nn

from quantizer import QLinear


# ---------------------------------------------------------------------- #
# Configure quantization                                                 #
# ---------------------------------------------------------------------- #
def apply_bit_assignment(model, bit_assignment, quantize_activations=False):
    """
    Push bit-widths from `bit_assignment` into the QLinear modules.

    Activation quantization is off by default: PTQ without an activation
    calibration step usually loses accuracy because the per-batch dynamic
    range bounces around. Weight-only PTQ is the standard apples-to-apples
    setting for HAWQ comparisons.
    """
    named_modules = dict(model.named_modules())
    for layer_name, bits in bit_assignment.items():
        module = named_modules.get(layer_name)
        if not isinstance(module, QLinear):
            continue
        module.set_weight_quant(True, int(bits))
        if quantize_activations:
            module.set_act_quant(True, int(bits))
        else:
            module.set_act_quant(False, int(bits))


def disable_all_quantization(model):
    """Turn every FakeQuantizer off - used to measure the FP32 baseline."""
    for module in model.modules():
        if isinstance(module, QLinear):
            module.set_weight_quant(False, 32)
            module.set_act_quant(False, 32)


# ---------------------------------------------------------------------- #
# Size accounting                                                        #
# ---------------------------------------------------------------------- #
def compute_effective_size(model, bit_assignment, default_bits=32):
    """
    Effective deployed size in MB.

    Quantized layer weights are counted at their assigned bit-width.
    Everything else (biases, layernorms, embeddings, head, ...) is counted
    at `default_bits` (32 by default - we don't quantize those).
    """
    quantized_param_ids = set()
    total_bits = 0

    named_modules = dict(model.named_modules())
    for layer_name, bits in bit_assignment.items():
        module = named_modules.get(layer_name)
        if isinstance(module, QLinear):
            total_bits += module.weight.numel() * int(bits)
            quantized_param_ids.add(id(module.weight))

    for p in model.parameters():
        if id(p) in quantized_param_ids:
            continue
        total_bits += p.numel() * default_bits

    return total_bits / 8 / (1024 ** 2)


def average_bits(bit_assignment, params_per_layer):
    """Parameter-weighted average bit-width across the assigned layers."""
    total_bits = sum(
        bit_assignment[n] * params_per_layer[n] for n in bit_assignment
    )
    total_params = sum(params_per_layer[n] for n in bit_assignment)
    if total_params == 0:
        return 0.0
    return total_bits / total_params


def collect_qlinear_param_counts(model):
    """Return {qlinear_name: weight.numel()} for every QLinear in the model."""
    return {
        name: module.weight.numel()
        for name, module in model.named_modules()
        if isinstance(module, QLinear)
    }


# ---------------------------------------------------------------------- #
# Evaluation                                                             #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, dataloader, device=None, max_batches=None, verbose=False):
    """Top-1 classification accuracy on `dataloader`."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"    eval batch {batch_idx+1}: running acc = {correct/max(total,1)*100:.2f}%")

    return correct / max(total, 1)
