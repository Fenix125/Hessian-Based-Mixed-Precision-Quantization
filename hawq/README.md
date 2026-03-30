# HAWQ phase

Let’s make it concrete.

Suppose analyzer says the order is:

1. layer_A
2. layer_B
3. layer_C

and allocator says:

1. layer_A → 8-bit
2. layer_B → 6-bit
3. layer_C → 4-bit

Then staged_qat() will do:

`Phase 1`

- enable quantization for layer_A
- freeze whole model
- unfreeze layer_A
- maybe unfreeze head
- fine-tune

`Phase 2`

- enable quantization for layer_B
- layer_A remains quantized
- freeze whole model
- unfreeze layer_B
- if progressive mode: also unfreeze layer_A
- maybe unfreeze head
- fine-tune

`Phase 3`

- enable quantization for layer_C
- layer_A and layer_B remain quantized
- freeze whole model
- unfreeze layer_C
- in progressive mode also unfreeze layer_A, layer_B
- maybe unfreeze head
- fine-tune

`Final recovery`

- keep all quantized
- jointly fine-tune quantized blocks for a short period
