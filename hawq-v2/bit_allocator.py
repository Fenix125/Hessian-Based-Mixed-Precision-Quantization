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

    # ------------------------------------------------------------------ #
    # Baselines                                                          #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Slider / budget mode                                               #
    # ------------------------------------------------------------------ #
    def allocate_by_budget(self, layer_sensitivities, target_avg_bits):
        """
        Pass-based parameter-weighted budget allocation.

        Strategy
        --------
        Walk the candidate bit-widths in *passes*:
            pass 0: promote layers from level 0 (min) to level 1
            pass 1: promote layers from level 1 to level 2
            ...
        Within each pass, promote in sensitivity order (highest first).

        Why passes (instead of "promote one layer all the way up")
        ----------------------------------------------------------
        Depth-first promotion (one layer all the way up before touching the
        next) blows up at intermediate budgets. e.g. with candidates
        {6, 8, 10} and budget 8 the depth-first version ends up with the
        top half of layers at 10 and the bottom half at 6 - the half stuck
        at 6 hurts more than the half at 10 helps, and the result loses
        to Uniform-8 (everyone at 8). The breadth-first pass-based version
        gracefully reduces to "everyone at 8" at budget 8, and only starts
        promoting to 10 once budget exceeds 8.

        Round-to-nearest at the boundary: if the next promotion overshoots
        the target by less than the current undershoot, accept it.

        Returns a dict layer_name -> bit_width.
        """
        bits_asc = sorted(self.candidate_weight_bits)
        min_bit, max_bit = bits_asc[0], bits_asc[-1]

        # Trivial edge cases.
        if target_avg_bits >= max_bit:
            return {name: max_bit for name in layer_sensitivities}
        if target_avg_bits <= min_bit:
            return {name: min_bit for name in layer_sensitivities}

        # Order layers by sensitivity (highest first - they get bumped first).
        sorted_layers = sorted(
            layer_sensitivities.items(),
            key=lambda x: x[1]["S_i"],
            reverse=True,
        )
        layer_names = [name for name, _ in sorted_layers]
        params = {name: meta["parameters"] for name, meta in sorted_layers}
        total_params = sum(params.values())

        allocated = {name: min_bit for name in layer_names}

        def weighted_avg(assignment):
            return (
                sum(assignment[n] * params[n] for n in layer_names)
                / total_params
            )

        # Walk one pass per adjacent pair of bit-widths.
        for level_idx in range(len(bits_asc) - 1):
            cur_bit = bits_asc[level_idx]
            next_bit = bits_asc[level_idx + 1]

            for name in layer_names:
                if allocated[name] != cur_bit:
                    continue  # already promoted in an earlier pass

                tentative = dict(allocated)
                tentative[name] = next_bit
                new_avg = weighted_avg(tentative)
                cur_avg = weighted_avg(allocated)

                if new_avg <= target_avg_bits:
                    allocated = tentative
                    continue

                # Crossing the target. Pick whichever side is closer.
                if abs(new_avg - target_avg_bits) < abs(cur_avg - target_avg_bits):
                    allocated = tentative
                return allocated

        return allocated
