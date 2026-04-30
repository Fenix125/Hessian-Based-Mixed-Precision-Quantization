"""
Bit-width allocation policies for HAWQ-v2.

Three modes are useful in our experiments:

allocate_uniform
    Same bit-width for every layer. This is the "dumb" baseline that
    everyone compares against - it's what PyTorch / standard PTQ does.

allocate_by_rank
    Sort layers by sensitivity, split into N equal-size buckets, assign
    candidate bits in descending order. Same logic as v1; cheap and
    deterministic.

allocate_by_budget
    Given a target *parameter-weighted average* bit-width, greedily
    promote layers from the minimum bit-width up, picking the most
    sensitive layer at each step. This is the engine behind the slider:
    move the slider to e.g. 5.5 bits, allocator returns a per-layer plan
    whose effective average is ~5.5.
"""


class HAWQv2BitAllocator:
    """
    Parameters
    ----------
    candidate_weight_bits : tuple[int]
        Bit-widths the allocator is allowed to assign. Sorted descending
        internally so index 0 is the highest precision.
    """

    def __init__(self, candidate_weight_bits=(8, 6, 4, 2)):
        self.candidate_weight_bits = tuple(
            sorted(candidate_weight_bits, reverse=True)
        )

    def allocate_uniform(self, layer_sensitivities, bit_width):
        """Same bit-width everywhere - the uniform PTQ baseline."""
        return {name: int(bit_width) for name in layer_sensitivities}

    def allocate_by_rank(self, layer_sensitivities):
        """Bucket allocation: top 1/k -> highest bits, etc."""
        sorted_layers = sorted(
            layer_sensitivities.items(),
            key=lambda x: x[1]["S_i"],
            reverse=True,
        )
        bits_desc = self.candidate_weight_bits
        k = len(bits_desc)
        n = len(sorted_layers)

        allocated = {}
        for rank, (name, _) in enumerate(sorted_layers):
            frac = rank / max(n - 1, 1)
            bin_idx = min(int(frac * k), k - 1)
            allocated[name] = bits_desc[bin_idx]
        return allocated
