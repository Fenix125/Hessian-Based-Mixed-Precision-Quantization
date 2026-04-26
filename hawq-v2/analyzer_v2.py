"""
HAWQ-v2 sensitivity analyzer.

The whole point of v2 is to replace the top-eigenvalue sensitivity from
HAWQ-v1 with the Hessian *trace*, estimated stochastically via Hutchinson's
algorithm:

    Tr(H) ~= E_{v ~ Rademacher}[ v^T H v ]

We compute v^T H v with two autograd passes:
    1. g = dL/dW                          (with create_graph=True)
    2. Hv = d(g . v)/dW                   (second backward)
    3. v^T H v                            (scalar)

For each layer we report:
    trace          - mean of v^T H v over multiple random vectors / batches
    parameters     - number of weights in the layer
    S_i            - |trace| / parameters  (per-parameter sensitivity)

S_i is the number the bit allocator consumes. Layers with high S_i sit in
sharp curvature and need more bits; layers with low S_i are flat and can be
compressed harder.
"""

import contextlib

import torch
import torch.nn as nn


def _math_attention_ctx():
    """
    Context manager that forces PyTorch's math SDPA backend.

    Why: the default (and the flash / mem-efficient) attention kernels do not
    implement double-backward, but Hutchinson's trace needs second-order
    autograd. Forcing the math kernel sidesteps the
    ``derivative for aten::_scaled_dot_product_efficient_attention_backward
    is not implemented`` error on transformers loaded without
    ``attn_implementation='eager'``.
    """
    # Newer torch (>=2.3): torch.nn.attention.sdpa_kernel
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        pass
    # Older torch fallback: backends.cuda.sdp_kernel
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
    except Exception:
        return contextlib.nullcontext()


class ViTHAWQv2Analyzer:
    """
    Hessian-trace sensitivity analyzer for ViT-style models.

    Parameters
    ----------
    model : nn.Module
        The pretrained model to analyze. Must be in eval mode (we put it
        there). Its parameters need requires_grad=True for the layers under
        analysis - the constructor does NOT modify the global requires_grad
        flags, so the caller is responsible for that.
    dataloader : torch.utils.data.DataLoader
        Provides (inputs, targets) batches. Used both to build the loss
        graph (so the Hessian is well-defined) and to average the trace
        estimate over multiple batches.
    criterion : nn.Module
        Loss function. Typically nn.CrossEntropyLoss().
    """

    def __init__(self, model, dataloader, criterion):
        self.dataloader = dataloader
        self.criterion = criterion

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        print(f"[HAWQ-v2] Using device: {device}")
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def isolate_blocks(self, include_qlinear=True):
        """
        Collect modules eligible for analysis.

        We pick up nn.Linear and (if available) QLinear modules. The
        analyzer does not care which class they are - it just needs a
        `weight` Parameter with requires_grad=True.
        """
        target_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                target_layers[name] = module
                continue
            if include_qlinear and module.__class__.__name__ == "QLinear":
                target_layers[name] = module

        print(f"[HAWQ-v2] Isolated {len(target_layers)} layers for trace analysis.")
        return target_layers

    def filter_blocks(
        self,
        target_blocks,
        include_keywords=None,
        exclude_keywords=None,
        max_layers=None,
    ):
        """Keep / drop layers based on substring matching of their dotted names."""
        include_keywords = [k.lower() for k in (include_keywords or [])]
        exclude_keywords = [k.lower() for k in (exclude_keywords or [])]

        filtered = {}
        for name, module in target_blocks.items():
            lname = name.lower()
            if include_keywords and not any(k in lname for k in include_keywords):
                continue
            if exclude_keywords and any(k in lname for k in exclude_keywords):
                continue
            filtered[name] = module
            if max_layers is not None and len(filtered) >= max_layers:
                break
        return filtered

    @staticmethod
    def _sample_rademacher(weight):
        """Draw a ±1 vector with the same shape / dtype / device as `weight`."""
        v = torch.randint(0, 2, weight.shape, dtype=weight.dtype, device=weight.device)
        return v.mul_(2).sub_(1)  # {0, 1} -> {-1, +1}

    def _hutchinson_trace_on_batch(
        self, layer_module, inputs, targets, num_samples=10
    ):
        """
        One batch worth of trace estimation for a single layer.

        Steps:
            1. Compute the loss for this batch.
            2. Compute g = dL/dW with create_graph=True.
            3. For each random Rademacher v:
                 - gv = sum(g * v)
                 - Hv = d(gv)/dW
                 - estimate += v . Hv
            4. Return mean over samples.
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Force math SDPA so the backward graph supports double-backward.
        with _math_attention_ctx():
            outputs = self.model(inputs)
            if hasattr(outputs, "logits"):
                loss = self.criterion(outputs.logits, targets)
            else:
                loss = self.criterion(outputs, targets)

            weight = layer_module.weight
            if weight is None:
                return 0.0

            g = torch.autograd.grad(
                loss, weight, create_graph=True, retain_graph=True
            )[0]

            if g.norm().item() < 1e-12:
                return 0.0

            trace_estimates = []
            for s in range(num_samples):
                v = self._sample_rademacher(weight)
                gv = (g * v).sum()
                Hv = torch.autograd.grad(
                    gv, weight, retain_graph=(s < num_samples - 1)
                )[0]
                vHv = (v * Hv).sum().item()
                trace_estimates.append(vHv)

        return float(sum(trace_estimates) / len(trace_estimates))

    def compute_trace(self, layer_module, num_samples=10, num_batches=1):
        """
        Average Hutchinson trace estimate over `num_batches` data batches.
        """
        traces = []
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            tr = self._hutchinson_trace_on_batch(
                layer_module=layer_module,
                inputs=inputs,
                targets=targets,
                num_samples=num_samples,
            )
            traces.append(tr)
            if batch_idx + 1 >= num_batches:
                break

        if not traces:
            raise RuntimeError("Dataloader produced no batches.")
        return float(sum(traces) / len(traces))

    def compute_layer_sensitivities(
        self,
        target_blocks=None,
        include_keywords=None,
        exclude_keywords=None,
        max_layers=None,
        num_samples=10,
        num_batches=1,
        sort_desc=True,
        verbose=True,
    ):
        """
        Run Hutchinson trace per layer, return a dict keyed by dotted name.

        Each entry has:
            trace      : float    raw Tr(H) estimate (can be slightly < 0)
            parameters : int      number of weight elements in the layer
            S_i        : float    |trace| / parameters  -> sensitivity score

        If sort_desc=True the dict is returned sorted by S_i descending.
        """
        if target_blocks is None:
            target_blocks = self.isolate_blocks()

        target_blocks = self.filter_blocks(
            target_blocks=target_blocks,
            include_keywords=include_keywords,
            exclude_keywords=exclude_keywords,
            max_layers=max_layers,
        )

        if not target_blocks:
            raise ValueError("No layers selected for sensitivity analysis.")

        sensitivities = {}
        if verbose:
            print(
                f"\n[HAWQ-v2] Computing Hutchinson-trace sensitivities "
                f"for {len(target_blocks)} layers "
                f"({num_samples} samples x {num_batches} batches each)..."
            )

        for idx, (layer_name, layer_module) in enumerate(target_blocks.items(), start=1):
            if verbose:
                print(f"  [{idx}/{len(target_blocks)}] {layer_name}")

            trace = self.compute_trace(
                layer_module=layer_module,
                num_samples=num_samples,
                num_batches=num_batches,
            )

            n_i = layer_module.weight.numel()
            s_i = abs(trace) / max(n_i, 1)

            sensitivities[layer_name] = {
                "trace": float(trace),
                "parameters": int(n_i),
                "S_i": float(s_i),
            }

            if verbose:
                print(
                    f"      Tr(H) = {trace:+.4e} | "
                    f"n_i = {n_i} | S_i = {s_i:.4e}"
                )

        if sort_desc:
            sensitivities = dict(
                sorted(
                    sensitivities.items(),
                    key=lambda x: x[1]["S_i"],
                    reverse=True,
                )
            )

        return sensitivities

    @torch.no_grad()
    def simulate_quantization_error(
        self,
        weight_tensor: torch.Tensor,
        bit_width: int,
        per_channel: bool = False,
        channel_axis: int = 0,
        eps: float = 1e-8,
    ) -> float:
        """
        Symmetric-uniform quantization L2 error: ||Q(W) - W||_2^2.
        Used by the Omega_i schedule (sensitivity * perturbation).
        """
        if bit_width < 2:
            raise ValueError(f"bit_width must be >= 2, got {bit_width}")

        w = weight_tensor.detach()
        qmax = (2 ** (bit_width - 1)) - 1
        qmin = -qmax

        if per_channel:
            reduce_dims = tuple(d for d in range(w.ndim) if d != channel_axis)
            max_val = w.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_val = w.abs().amax()

        scale = torch.clamp(max_val / qmax, min=eps)
        q_int = torch.round(w / scale)
        q_int = torch.clamp(q_int, qmin, qmax)
        q_w = q_int * scale

        return float((q_w - w).pow(2).sum().item())
