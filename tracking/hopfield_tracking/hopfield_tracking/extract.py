"""Turn a relaxed network state into reconstructed tracks, and score them
against ground truth when available.

The paper doesn't give a precise algorithm for this step -- it just says a
valid track "can be read out by choosing an arbitrary point on the chain
and following the sequence in both directions", and that non-convergence
shows up as "missing or illegal neurons" / "incorrect choice of neurons"
(fig. 8b). `chain_tracks` below implements that reading-out procedure
directly, and truncates rather than crashes at a bifurcation, which is
exactly how an "illegal" partial reconstruction shows up here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .network import Segment

ON_THRESHOLD = 0.1  # paper: "only neurons with output values > 0.1 are drawn"


def on_segments(segments: list[Segment], f_v: np.ndarray, threshold: float = ON_THRESHOLD) -> list[Segment]:
    return [seg for seg, value in zip(segments, f_v) if value > threshold]


def chain_tracks(active: list[Segment]) -> list[list[int]]:
    """Chain "on" segments end-to-start into tracks (lists of hit ids).

    One chain per point that starts a track (no "on" segment ends there);
    walking stops (a shorter, "illegal" chain) at any point with more than
    one outgoing "on" segment, or one already claimed by another chain.
    A closed loop of "on" segments with no such start point is silently
    dropped -- a known limitation, worth flagging rather than hiding.
    """
    outgoing: dict[int, list[Segment]] = defaultdict(list)
    incoming_count: dict[int, int] = defaultdict(int)
    for seg in active:
        outgoing[seg.start_id].append(seg)
        incoming_count[seg.end_id] += 1

    starts = [point for point in outgoing if incoming_count.get(point, 0) == 0]

    chains: list[list[int]] = []
    visited: set[int] = set()
    for start in starts:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        current = start
        while len(outgoing.get(current, [])) == 1:
            next_point = outgoing[current][0].end_id
            if next_point in visited:
                break
            chain.append(next_point)
            visited.add(next_point)
            current = next_point
        chains.append(chain)
    return chains


def score_against_truth(chains: list[list[int]], hit_particle_ids: dict[int, int]) -> dict:
    """Compare found chains to the true per-particle hit groups. `perfect`
    is True iff the found chains are exactly the true groups, no more, no
    less -- the quantitative version of "fig. 8a looks perfect / fig. 8b
    looks confused"."""
    true_groups: dict[int, set[int]] = defaultdict(set)
    for hit_id, particle_id in hit_particle_ids.items():
        true_groups[particle_id].add(hit_id)

    true_sets = {frozenset(group) for group in true_groups.values()}
    found_sets = {frozenset(chain) for chain in chains}
    exact_matches = true_sets & found_sets

    return dict(
        n_true_tracks=len(true_sets),
        n_found_chains=len(found_sets),
        n_exact_matches=len(exact_matches),
        perfect=exact_matches == true_sets == found_sets,
    )
