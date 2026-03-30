class HAWQBitAllocator:
    """
    Simple bit allocation policy for HAWQ.

    The analyzer measures sensitivity.
    The allocator converts sensitivity ranking into actual bit-width choices.

    This baseline allocator is rank-based:
    - high sensitivity  -> high precision
    - medium sensitivity -> medium precision
    - low sensitivity   -> low precision

    This is simple, deterministic, and good enough for a first prototype.
    """
    def __init__(self, candidate_weight_bits=(8, 6, 4), candidate_act_bits=(8, 6, 4)):
        self.candidate_weight_bits = tuple(candidate_weight_bits)
        self.candidate_act_bits = tuple(candidate_act_bits)

    def allocate_by_rank(self, layer_sensitivities):
        """
        Simple baseline:
        higher S_i -> higher precision
        """
        sorted_layers = sorted(
            layer_sensitivities.items(),
            key=lambda x: x[1]["S_i"],
            reverse=True
        )

        bits_desc = sorted(self.candidate_weight_bits, reverse=True)
        n = len(sorted_layers)

        allocated = {}
        for rank, (name, _) in enumerate(sorted_layers):
            frac = rank / max(n - 1, 1)

            if frac < 1/3:
                bit = bits_desc[0]
            elif frac < 2/3:
                bit = bits_desc[min(1, len(bits_desc)-1)]
            else:
                bit = bits_desc[min(2, len(bits_desc)-1)]

            allocated[name] = bit

        return allocated