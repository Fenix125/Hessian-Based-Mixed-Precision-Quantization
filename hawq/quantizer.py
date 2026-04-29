import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantizer(nn.Module):
    """
    Simulates uniform symmetric quantization during training.
    Includes Exponential Moving Average (EMA) for dynamic activation ranges.

    Important idea:
    ----------------
    This module does NOT convert the model to real integer inference.
    Instead, it simulates the effect of quantization in the forward pass.

    When enabled=False:
        - the input tensor is returned unchanged

    When enabled=True:
        - the input tensor is quantized onto a discrete grid
        - the output still remains a float tensor
        - this allows autograd / backpropagation to keep working

    This is the standard idea behind Quantization-Aware Training (QAT):
    the model is trained while "feeling" the quantization error.

    Parameters
    ----------
    bit_width : int
        Number of quantization bits. Example: 8, 6, 4.
    enabled : bool
        Whether fake quantization is currently active.
    per_channel : bool
        If False, one global scale is used for the whole tensor.
        If True, separate scales are used per channel.
    channel_axis : int
        Which axis is treated as "channel" when per_channel=True.
        For nn.Linear weights, channel_axis=0 usually means one scale per output row.
    eps : float
        Small positive value to avoid division by zero in scale computation.
    """
    def __init__(self, bit_width=8, enabled=False, per_channel=False, channel_axis=0, eps=1e-8, is_activation=False, momentum=0.1):
        super().__init__()
        self.bit_width = bit_width
        self.enabled = enabled
        self.per_channel = per_channel
        #the dimension considered as the channel dimension in per-channel mode.
        self.channel_axis = channel_axis
        self.eps = eps

        self.is_activation = is_activation
        self.momentum = momentum
        
        self.register_buffer('running_max_val', torch.tensor(0.0))

    def set_enabled(self, enabled: bool):
        """
        Turn fake quantization on or off.
        """
        self.enabled = enabled

    def set_bit_width(self, bit_width: int):
        """
        Change the active quantization precision.
        """
        self.bit_width = int(bit_width)

    def forward(self, x):
        """
        Apply fake quantization to x if enabled.

        Logic
        -----
        1. If quantization is disabled:
           return x unchanged.

        2. Otherwise:
           - compute quantization range
           - compute scale
           - map x to integer grid
           - clamp to valid range
           - map back to float domain

        3. Use an STE-style trick:
           return x + (x_q - x).detach()

            Why? In forward pass:
               value is x_q

            In backward pass:
               gradient behaves approximately like identity through x

            This is the usual straight-through estimator pattern.
        """
        if not self.enabled:
            return x

        qmax = (2 ** (self.bit_width - 1)) - 1
        qmin = -qmax

        #maximum value based on tensor type
        if self.per_channel:
            reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_axis)
            current_max_val = x.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            current_max_val = x.abs().amax()

        #EMA tracking for activations, absolute static for weights
        if self.is_activation:
            if self.training:
                if self.running_max_val.item() == 0.0:
                    self.running_max_val.copy_(current_max_val.detach())
                else:
                    #smoothing out the dynamic range updates
                    self.running_max_val.mul_(1 - self.momentum).add_(current_max_val.detach() * self.momentum)
                
                max_val = self.running_max_val
            else:
                max_val = self.running_max_val
        else:
            max_val = current_max_val

        #quantizing and project
        scale = torch.clamp(max_val / max(qmax, 1), min=self.eps)
        q = torch.round(x / scale)
        q = torch.clamp(q, qmin, qmax)
        x_q = q * scale

        return x + (x_q - x).detach()


class QLinear(nn.Module):
    """
    Quantization-aware replacement for nn.Linear.

    This module stores its own trainable float weights, but can apply:
    - fake quantization to input activations
    - fake quantization to weights

    Design
    ------
    The wrapped module remains trainable in floating point, but during
    forward pass it can simulate quantized behavior.

    
    It gives us ability to train with quantization effects before exporting to a true quantized model.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        #original weight into a new trainable parameter.
        self.weight = nn.Parameter(linear.weight.detach().clone())

        #if original layer has bias
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        self.weight_quant = FakeQuantizer(bit_width=8, enabled=False, per_channel=False, is_activation=False)
        self.act_quant = FakeQuantizer(bit_width=8, enabled=False, per_channel=False, is_activation=True)

        
    def set_weight_quant(self, enabled: bool, bit_width: int):
        """
        Enable/disable weight quantization and set its precision.
        """
        self.weight_quant.set_enabled(enabled)
        self.weight_quant.set_bit_width(bit_width)

    def set_act_quant(self, enabled: bool, bit_width: int):
        """
        Enable/disable activation quantization and set its precision.
        """
        self.act_quant.set_enabled(enabled)
        self.act_quant.set_bit_width(bit_width)

    def forward(self, x):
        """
        Forward pass through quantized linear layer.

        Steps:
        1. Fake-quantize the input activations.
        2. Fake-quantize the weights.
        3. Performs the linear transformation using F.linear
        """
        x = self.act_quant(x)
        w = self.weight_quant(self.weight)
        return F.linear(x, w, self.bias)  # pylint: disable=not-callable
    

def get_parent_module(root: nn.Module, module_name: str):
    """
    Return the parent module and child attribute name for a dotted module path.

    Suppose module_name is:
        "swin.encoder.layers.0.blocks.1.attention.self.query"

    Then:
        - parent will be the module corresponding to everything except "query"
        - child_name will be "query"

    This helper is needed because replacing a nested module requires access
    to its parent object and the name of the child attribute inside that parent.
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def replace_linear_with_qlinear(model: nn.Module, target_names):
    """
    Replace selected nn.Linear modules with QLinear wrappers.

    Parameters
    ----------
    model : nn.Module
        The model whose submodules should be replaced.
    target_names : iterable[str]
        Full dotted names of modules that should be wrapped.
    -----
    This modifies the model in place.
    """
    for name in target_names:
        parent, child_name = get_parent_module(model, name)
        old = getattr(parent, child_name)
        if not isinstance(old, nn.Linear):
            continue
        setattr(parent, child_name, QLinear(old))