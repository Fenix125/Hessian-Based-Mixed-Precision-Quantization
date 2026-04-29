import torch
import torch.nn as nn

class ViTHAWQAnalyzer:
    def __init__(self, model, dataloader, criterion):
        self.dataloader = dataloader
        self.criterion = criterion

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        print(f"Using device: {device}")
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def isolate_blocks(self):
        """
        Collect all nn.Linear modules.
        """
        target_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                target_layers[name] = module

        print(f"Isolated {len(target_layers)} linear blocks for Hessian analysis.")
        return target_layers

    def filter_blocks(self, target_blocks, include_keywords=None, exclude_keywords=None, max_layers=None):
        """
        Filter block dict by name.
        """
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

    def _compute_top_eigenvalue_on_batch(self, layer_module, inputs, targets, max_iter=30, tol=1e-3):
        """
        Power iteration on one explicit batch.
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        if hasattr(outputs, "logits"):
            loss = self.criterion(outputs.logits, targets)
        else:
            loss = self.criterion(outputs, targets)

        weight = layer_module.weight

        g_i = torch.autograd.grad(
            loss, weight, create_graph=True, retain_graph=True
        )[0]

        grad_norm = g_i.norm().item()
        if grad_norm < 1e-12:
            return 0.0

        v = torch.randn_like(weight)
        v = v / (torch.norm(v, p=2) + 1e-12)

        prev_eigenvalue = None

        for _ in range(max_iter):
            gv = torch.sum(g_i * v)

            Hv = torch.autograd.grad(
                gv, weight, retain_graph=True
            )[0]

            Hv_norm = torch.norm(Hv, p=2).item()
            if Hv_norm < 1e-12:
                return 0.0

            #rayleigh quotient estimate
            current_eigenvalue = torch.sum(v * Hv).item()

            v = (Hv / (Hv_norm + 1e-12)).detach()

            if prev_eigenvalue is not None and abs(current_eigenvalue - prev_eigenvalue) < tol:
                return float(abs(current_eigenvalue))

            prev_eigenvalue = current_eigenvalue

        return float(abs(prev_eigenvalue) if prev_eigenvalue is not None else 0.0)

    def compute_top_eigenvalue(self, layer_module, max_iter=30, tol=1e-3, num_batches=1):
        """
        Average top-eigenvalue estimate over the first `num_batches` batches.
        """
        eigenvalues = []

        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            eig = self._compute_top_eigenvalue_on_batch(
                layer_module=layer_module,
                inputs=inputs,
                targets=targets,
                max_iter=max_iter,
                tol=tol,
            )
            eigenvalues.append(eig)

            if batch_idx + 1 >= num_batches:
                break

        if not eigenvalues:
            raise RuntimeError("Dataloader produced no batches.")

        avg_eig = sum(eigenvalues) / len(eigenvalues)
        return float(avg_eig)

    def compute_layer_sensitivities(self, target_blocks=None, include_keywords=None, exclude_keywords=None, max_layers=None, max_iter=30, tol=1e-3, num_batches=1, sort_desc=True, verbose=True):
        """
        Compute HAWQ sensitivities for selected layers.

        Returns:
            dict[layer_name] = {
                "lambda_1": float,
                "parameters": int,
                "S_i": float
            }
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

        layer_sensitivities = {}

        if verbose:
            print(f"\nComputing sensitivities for {len(target_blocks)} layers...")

        for idx, (layer_name, layer_module) in enumerate(target_blocks.items(), start=1):
            if verbose:
                print(f"[{idx}/{len(target_blocks)}] Analyzing: {layer_name}")

            lambda_1 = self.compute_top_eigenvalue(
                layer_module=layer_module,
                max_iter=max_iter,
                tol=tol,
                num_batches=num_batches,
            )

            n_i = layer_module.weight.numel()
            s_i = lambda_1 / max(n_i, 1)

            layer_sensitivities[layer_name] = {
                "lambda_1": float(lambda_1),
                "parameters": int(n_i),
                "S_i": float(s_i),
            }

            if verbose:
                print(f"   -> Top Eigenvalue (λ₁): {lambda_1:.8e}")
                print(f"   -> Parameter Count (n_i): {n_i}")
                print(f"   -> Sensitivity Metric (S_i): {s_i:.8e}")

        if sort_desc:
            layer_sensitivities = dict(
                sorted(
                    layer_sensitivities.items(),
                    key=lambda x: x[1]["S_i"],
                    reverse=True,
                )
            )

        return layer_sensitivities
    
    @torch.no_grad()
    def simulate_quantization_error(self, weight_tensor: torch.Tensor, bit_width: int, per_channel: bool = False, channel_axis: int = 0, eps: float = 1e-8) -> float:
        """
        Simulate symmetric uniform quantization and return ||Q(W) - W||_2^2.

        Assumptions:
        - symmetric signed quantization
        - zero-point = 0
        - narrow signed range: [-qmax, qmax], where qmax = 2^(b-1)-1

        Args:
            weight_tensor: tensor to quantize
            bit_width: target bit-width, must be >= 2
            per_channel: whether to use per-channel scales
            channel_axis: channel dimension if per_channel=True
            eps: small constant to avoid divide-by-zero
        """
        if bit_width < 2:
            raise ValueError(f"bit_width must be >= 2, got {bit_width}")

        w = weight_tensor.detach()

        qmax = (2 ** (bit_width - 1)) - 1
        qmin = -qmax

        if qmax <= 0:
            raise ValueError(f"Invalid qmax={qmax} for bit_width={bit_width}")

        if per_channel:
            reduce_dims = tuple(d for d in range(w.ndim) if d != channel_axis)
            max_val = w.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_val = w.abs().amax()

        scale = torch.clamp(max_val / qmax, min=eps)

        q_int = torch.round(w / scale)
        q_int = torch.clamp(q_int, qmin, qmax)

        q_w = q_int * scale

        perturbation_error = (q_w - w).pow(2).sum()

        return float(perturbation_error.item())

    def generate_qat_schedule(self, layer_sensitivities: dict, allocated_bits: dict, per_channel: bool = False, channel_axis: int = 0):
        """
        Compute HAWQ fine-tuning priorities:
            Omega_i = lambda_i * ||Q(W_i) - W_i||_2^2

        Args:
            layer_sensitivities: mapping
                layer_name -> {"lambda_1": ..., ...}
            allocated_bits: mapping
                layer_name -> target bit-width
            per_channel: whether perturbation simulation uses per-channel quantization
            channel_axis: channel axis for per-channel mode
        """
        tuning_priorities = {}
        named_modules = dict(self.model.named_modules())

        print("\nCalculating perturbation impact (Omega_i)...")

        for layer_name, metrics in layer_sensitivities.items():
            if layer_name not in allocated_bits:
                raise KeyError(f"Missing bit allocation for layer: {layer_name}")

            if layer_name not in named_modules:
                raise KeyError(f"Layer not found in model: {layer_name}")

            layer_module = named_modules[layer_name]

            if not hasattr(layer_module, "weight") or layer_module.weight is None:
                raise ValueError(f"Layer {layer_name} has no weight tensor")

            target_bits = int(allocated_bits[layer_name]["weight_bits"])
            lambda_1 = float(metrics["lambda_1"])

            l2_error = self.simulate_quantization_error(
                layer_module.weight,
                bit_width=target_bits,
                per_channel=per_channel,
                channel_axis=channel_axis,
            )

            omega_i = lambda_1 * l2_error

            tuning_priorities[layer_name] = {
                "lambda_1": lambda_1,
                "target_bits": target_bits,
                "L2_error": l2_error,
                "Omega_i": omega_i,
            }

        schedule = sorted(
            tuning_priorities.items(),
            key=lambda x: (x[1]["Omega_i"], x[1]["lambda_1"]),
            reverse=True,
        )
        print("\n=== Deterministic Fine-Tuning Order (Descending Omega_i) ===")
        for order, (name, metrics) in enumerate(schedule, start=1):
            print(f"Phase {order}: Fine-tune {name}")
            print(
                f"   -> Ω_i: {metrics['Omega_i']:.8e} "
                f"(Bits: {metrics['target_bits']}, "
                f"λ₁: {metrics['lambda_1']:.8e}, "
                f"L2 error: {metrics['L2_error']:.8e})"
            )
        return schedule