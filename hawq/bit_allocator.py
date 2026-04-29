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
        Dynamic baseline allocator
        Mathematically maps the continuous sensitivity rank onto an arbitrary 
        discrete set of candidate bit-widths.
        """
        sorted_layers = sorted(
            layer_sensitivities.items(),
            key=lambda x: x[1]["S_i"],
            reverse=True
        )

        bits_desc = sorted(self.candidate_weight_bits, reverse=True)
        num_candidates = len(bits_desc)
        n = len(sorted_layers)

        allocated = {}
        for rank, (name, _) in enumerate(sorted_layers):
            #calculates the fractional depth of this layer in the sensitivity hierarchy
            frac = rank / max(n - 1, 1)
            
            #the index based on the number of available bit-widths
            bin_idx = int(frac * num_candidates)
            
            bin_idx = min(bin_idx, num_candidates - 1) #prevents index out of bounds for the absolute lowest rank
            
            allocated[name] = bits_desc[bin_idx]

        return allocated