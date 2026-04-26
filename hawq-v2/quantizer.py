"""
Fake quantization primitives for HAWQ-v2.

Mirrors the v1 fake-quant logic but lives in this folder so the v2 pipeline
is self-contained.

FakeQuantizer
    Symmetric uniform quantization with the straight-through estimator (STE).
    Forward pass returns a tensor that has been mapped onto the integer grid
    and back to float; backward pass behaves like identity. This lets
    autograd keep working through the simulated quantization step, which is
    required for both QAT and the HAWQ-v2 sensitivity analysis (Hutchinson's
    trace needs second-order gradients to flow).

QLinear
    A drop-in replacement for nn.Linear that owns one FakeQuantizer for the
    weight and one for the input activation. The actual linear arithmetic
    still happens in float - we only "feel" the quantization error in the
    forward pass.

replace_linear_with_qlinear
    Walk a model and swap selected nn.Linear modules in-place with QLinear
    wrappers. Used right before running the analyzer / PTQ pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantizer(nn.Module):
    """
    Symmetric uniform fake quantization with a straight-through estimator.

    Parameters
    ----------
    bit_width : int
        Number of bits used for the integer grid. e.g. 8 -> qmax = 127.
    enabled : bool
        If False the module is a no-op (returns the input unchanged).
    per_channel : bool
        If True, computes one scale per channel; otherwise one global scale.
    channel_axis : int
        Which dimension is the "channel" dimension when per_channel=True.
    eps : float
        Floor on the scale to avoid division by zero on near-zero tensors.
    """

    def __init__(self, bit_width=8, enabled=False, per_channel=False, channel_axis=0, eps=1e-8):
        super().__init__()
        self.bit_width = int(bit_width)
        self.enabled = bool(enabled)
        self.per_channel = bool(per_channel)
        self.channel_axis = int(channel_axis)
        self.eps = float(eps)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def set_bit_width(self, bit_width: int):
        self.bit_width = int(bit_width)

    def forward(self, x):
        if not self.enabled:
            return x

        qmax = (2 ** (self.bit_width - 1)) - 1
        qmin = -qmax
        if qmax <= 0:
            return x

        if self.per_channel:
            reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_axis)
            max_val = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_val = x.detach().abs().amax()

        scale = torch.clamp(max_val / max(qmax, 1), min=self.eps)
        q = torch.round(x / scale)
        q = torch.clamp(q, qmin, qmax)
        x_q = q * scale

        # STE: forward sees x_q, backward flows gradient as if identity.
        return x + (x_q - x).detach()


class QLinear(nn.Module):
    """
    nn.Linear with optional weight + activation fake quantization.

    Stores its own trainable float weight (cloned from the wrapped Linear) so
    a quantization-aware model can still be fine-tuned end-to-end. In PTQ
    mode we just enable the FakeQuantizers and skip training.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = (
            nn.Parameter(linear.bias.detach().clone())
            if linear.bias is not None
            else None
        )
        self.weight_quant = FakeQuantizer(bit_width=8, enabled=False, per_channel=False)
        self.act_quant = FakeQuantizer(bit_width=8, enabled=False, per_channel=False)

    def set_weight_quant(self, enabled: bool, bit_width: int):
        self.weight_quant.set_enabled(enabled)
        self.weight_quant.set_bit_width(bit_width)

    def set_act_quant(self, enabled: bool, bit_width: int):
        self.act_quant.set_enabled(enabled)
        self.act_quant.set_bit_width(bit_width)

    def forward(self, x):
        x = self.act_quant(x)
        w = self.weight_quant(self.weight)
        return F.linear(x, w, self.bias)  # pylint: disable=not-callable


def get_parent_module(root: nn.Module, module_name: str):
    """Resolve the parent module + child attribute name for a dotted path."""
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def replace_linear_with_qlinear(model: nn.Module, target_names):
    """
    Replace selected nn.Linear modules with QLinear in-place.

    `target_names` is an iterable of dotted module names (the same names
    produced by `model.named_modules()`).
    """
    for name in target_names:
        parent, child_name = get_parent_module(model, name)
        old = getattr(parent, child_name)
        if not isinstance(old, nn.Linear):
            continue
        setattr(parent, child_name, QLinear(old))
