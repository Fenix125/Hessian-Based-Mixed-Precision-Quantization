import torch
import torch.nn as nn
from quantizer import replace_linear_with_qlinear

class HAWQTrainer:
    """
    Executes staged HAWQ quantization-aware training.

    1. Replaces selected nn.Linear layers with QLinear wrappers.
    2. Builds fine-tuning schedule using Omega_i.
    3. For each phase:
       - enables quantization for current block
       - freeze everything
       - unfreeze current block
       - optionally unfreeze previous quantized blocks
       - optionally unfreeze task head
       - fine-tune
    4. Runs a final recovery stage over quantized blocks.

    Expects an external analyzer object for that.
    """
    def __init__(self, model, analyzer, criterion, train_dataloader, val_dataloader=None, lr=1e-5, weight_decay=1e-4, device=None):
        self.model = model
        self.analyzer = analyzer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        if device is None:
            device = analyzer.device
        self.device = device
        self.model.to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        #blocks already quantized in previous phases.
        self.quantized_blocks = []
        #final fine-tuning order here
        self.block_schedule = []

    def freeze_all(self):
        """
        Freeze all model parameters.
        """
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_module(self, module: nn.Module):
        """
        Unfreeze all parameters inside one module.
        """
        for p in module.parameters():
            p.requires_grad_(True)

    def build_optimizer(self):
        """
        Build AdamW optimizer using only currently trainable parameters.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def forward_loss(self, batch):
        """
        Runs one forward pass and compute loss.
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = self.criterion(logits, targets)
        return loss, logits, targets

    def train_one_epoch(self, optimizer):
        """
        Trains the current model for one epoch over the training dataloader.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_dataloader:
            optimizer.zero_grad()
            loss, _, _ = self.forward_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(self.train_dataloader), 1)

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the model on validation set.
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss, logits, targets = self.forward_loss(batch)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        return {
            "val_loss": total_loss / max(len(self.val_dataloader), 1),
            "val_acc": correct / max(total, 1),
        }

    def enable_block_quantization(self, block_name, weight_bits, act_bits=None):
        """ 
        Enables fake quantization for one selected block.

        Parameters:
        block_name : str
            Dotted module name of the block to quantize.
        weight_bits : int
            Weight precision to activate.
        act_bits : int or None
            Optional activation precision. If None, activation quantization
            is not changed.
        """
        module = dict(self.model.named_modules())[block_name]

        if hasattr(module, "set_weight_quant"):
            module.set_weight_quant(True, weight_bits)

        if act_bits is not None and hasattr(module, "set_act_quant"):
            module.set_act_quant(True, act_bits)

    def prepare_quant_modules(self, target_block_names):
        """
        Replace selected nn.Linear modules with QLinear wrappers.
        """
        replace_linear_with_qlinear(self.model, target_block_names)

    def build_schedule(self, layer_sensitivities, allocated_bits):
        """
        Builds the HAWQ phase order from analyzer-generated Omega_i schedule.
        """
        schedule = self.analyzer.generate_qat_schedule(
            layer_sensitivities=layer_sensitivities,
            allocated_bits=allocated_bits,
        )
        self.block_schedule = [name for name, _ in schedule]
        return schedule

    def staged_qat(self, layer_sensitivities, allocated_bits, phase_epochs=1, train_mode="strict_freeze",   # "strict_freeze" or "progressive"
        unfreeze_head_keywords=("classifier", "head"),
    ):
        """
        Performs HAWQ training.

        Parameters
        ----------
        layer_sensitivities : dict
            Sensitivity output from analyzer.
        allocated_bits : dict
            Mapping layer_name -> target weight bit-width.
        phase_epochs : int
            Number of epochs to train in each HAWQ stage.
        train_mode : str
            One of:
                "progressive"   -> current block + previous quantized blocks stay trainable
                "strict_freeze" -> only current block is trainable
        unfreeze_head_keywords : tuple[str]
            Substrings used to detect task head modules that should stay trainable.

        Returns
        -------
        list
            Final HAWQ schedule.
        """
        if train_mode not in ("strict_freeze", "progressive"):
            raise ValueError("Wrong train mode, should be one of: strict_freeze, progressive")
    
        target_block_names = list(layer_sensitivities.keys())
        self.prepare_quant_modules(target_block_names)
        schedule = self.build_schedule(layer_sensitivities, allocated_bits)

        named_modules = dict(self.model.named_modules())

        for phase_idx, (block_name, block_info) in enumerate(schedule, start=1):

            w_bits = allocated_bits[block_name]["weight_bits"]
            a_bits = allocated_bits[block_name]["act_bits"]

            print(f"\n[Phase {phase_idx}] Quantize + fine-tune: {block_name} (W{w_bits}A{a_bits})")

            self.enable_block_quantization(block_name, weight_bits=w_bits, act_bits=a_bits)

            self.freeze_all()

            self.unfreeze_norm_layers()
            
            #always unfreezes current block
            self.unfreeze_module(named_modules[block_name])

            #optionally keeps earlier quantized blocks trainable
            if train_mode == "progressive":
                for prev_name in self.quantized_blocks:
                    self.unfreeze_module(named_modules[prev_name])

            #unfreezes head if present
            for name, module in named_modules.items():
                if any(k in name.lower() for k in unfreeze_head_keywords):
                    self.unfreeze_module(module)

            optimizer = self.build_optimizer()

            for epoch in range(phase_epochs):
                train_loss = self.train_one_epoch(optimizer)
                metrics = self.evaluate()
                print(
                    f"   epoch {epoch+1}/{phase_epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    + " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                )

            self.quantized_blocks.append(block_name)

        return schedule

    def final_recovery(self, epochs=1, unfreeze_all_quantized=True):
        """
        Final joint recovery phase after all staged blocks were quantized.

        After tuning, it is useful to run a global train recovery
        phase over the quantized part of the model.

        Parameters
        ----------
        epochs : int
            Number of recovery epochs.
        unfreeze_all_quantized : bool
            If True, all previously quantized blocks are unfrozen together.
        """
        print("\n[Final recovery stage]")

        self.freeze_all()
        named_modules = dict(self.model.named_modules())

        if unfreeze_all_quantized:
            for name in self.quantized_blocks:
                if name in named_modules:
                    self.unfreeze_module(named_modules[name])

        optimizer = self.build_optimizer()

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(optimizer)
            metrics = self.evaluate()
            print(
                f"   epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                + " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            )

    def unfreeze_norm_layers(self):
        """
        Unfreeze all normalization layers (LayerNorm in ViTs).
        This is mathematically critical to absorb quantization variance shift.
        """
        for module in self.model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for p in module.parameters():
                    p.requires_grad_(True)