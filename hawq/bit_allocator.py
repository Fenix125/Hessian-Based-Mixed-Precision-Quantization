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
    def __init__(self, 
                 candidate_weight_bits=(8, 6, 4), 
                 candidate_act_bits=(8, 6, 4),
                 protected_keywords=("classifier", "head"),
                 protected_w_bits=8,
                 protected_a_bits=8):
        
        self.candidate_weight_bits = sorted(list(candidate_weight_bits), reverse=True)
        self.candidate_act_bits = sorted(list(candidate_act_bits), reverse=True)
        
        #policy configurations for mathematical bottlenecks
        self.protected_keywords = [k.lower() for k in protected_keywords]
        self.protected_w_bits = protected_w_bits
        self.protected_a_bits = protected_a_bits

    def allocate_by_rank(self, layer_sensitivities):
        """
        Dynamic baseline allocator
        Mathematically maps the continuous sensitivity rank onto an arbitrary 
        discrete set of candidate bit-widths.
        """
        allocated = {}
        dynamic_layers = {}

        for name, metrics in layer_sensitivities.items():
            lname = name.lower()
            if any(k in lname for k in self.protected_keywords):
                allocated[name] = {
                    "weight_bits": self.protected_w_bits,
                    "act_bits": self.protected_a_bits
                }
            else:
                dynamic_layers[name] = metrics

        #sort only the dynamic layers by S_i (Descending)
        sorted_dynamic = sorted(
            dynamic_layers.items(),
            key=lambda x: x[1]["S_i"],
            reverse=True
        )

        num_w_candidates = len(self.candidate_weight_bits)
        n = len(sorted_dynamic)

        for rank, (name, _) in enumerate(sorted_dynamic):
            #weight precision
            frac = rank / max(n - 1, 1)
            w_bin_idx = int(frac * num_w_candidates)
            w_bin_idx = min(w_bin_idx, num_w_candidates - 1)
            w_bits = self.candidate_weight_bits[w_bin_idx]

            #activation precision (A_bits >= W_bits)
            valid_act_bits = [bits for bits in self.candidate_act_bits if bits >= w_bits]
            if not valid_act_bits:
                a_bits = self.candidate_act_bits[0]
            else:
                a_bits = valid_act_bits[-1]

            allocated[name] = {
                "weight_bits": w_bits,
                "act_bits": a_bits
            }

        return allocated