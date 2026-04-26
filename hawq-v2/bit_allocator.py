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
        Greedy parameter-weighted budget allocation.

        Strategy
        --------
        1. Start every layer at the minimum bit-width.
        2. While we are below target average, find the most-sensitive
           layer that has room to grow and bump it to the next higher
           bit-width.
        3. Stop when the next promotion would overshoot. If the overshoot
           is closer to the target than staying put, take that promotion
           anyway (round-to-nearest behavior).

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

        # Greedy promotion loop.
        while True:
            # Pick the highest-sensitivity layer that still has room.
            promote = None
            for name in layer_names:
                if allocated[name] < max_bit:
                    promote = name
                    break
            if promote is None:
                break

            cur = allocated[promote]
            next_bit = bits_asc[bits_asc.index(cur) + 1]

            tentative = dict(allocated)
            tentative[promote] = next_bit
            new_avg = weighted_avg(tentative)
            cur_avg = weighted_avg(allocated)

            if new_avg <= target_avg_bits:
                allocated = tentative
                continue

            # Crossing the target. Pick whichever side is closer.
            if abs(new_avg - target_avg_bits) < abs(cur_avg - target_avg_bits):
                allocated = tentative
            break

        return allocated
