"""Strahler order, runs, Douglas-Peucker, and the two properties everything else rests on."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from konnektion import geometry as g
from konnektion.sources import Network, coerce_network
from tests.conftest import arbor, chain


def perfect_tree() -> Network:
    """A depth-2 binary tree whose Strahler orders are known by hand: [3, 2, 2, 1, 1, 1, 1]."""
    nodes = np.array(
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [2, 2, 0], [2, 0.5, 0], [2, -0.5, 0], [2, -2, 0]],
        dtype=float,
    )
    edges = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
    return Network(nodes=nodes, edges=edges, root=0)


def test_strahler_matches_the_hand_computed_orders():
    """A leaf is 1; a node whose two children tie takes their order plus one."""
    assert g.strahler_orders(perfect_tree()).tolist() == [3, 2, 2, 1, 1, 1, 1]


def test_strahler_rises_only_where_equal_branches_meet():
    """One child of order 2 and one of order 1 leaves the parent at 2, not 3."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [1, 5, 0]], dtype=float)
    edges = np.array([[0, 1], [1, 2], [1, 3], [0, 4]])
    orders = g.strahler_orders(Network(nodes=nodes, edges=edges, root=0))
    assert orders[1] == 2, "two order-1 children tie, so this rises"
    assert orders[0] == 2, "an order-2 and an order-1 child do not tie, so this does not"


def test_strahler_is_non_decreasing_toward_the_root():
    """The property that makes threshold pruning ancestor-closed for free.

    Checked on a real arbor rather than the toy tree: this is the load-bearing claim, and it is
    the one that would quietly stop holding if the post-order walk were ever wrong.
    """
    network = arbor()
    orders = g.strahler_orders(network)
    parent, _ = g.spanning_forest(network)
    checked = 0
    for child, up in enumerate(parent.tolist()):
        if up < 0:
            continue
        checked += 1
        assert orders[up] >= orders[child]
    assert checked > 100, "the property must be exercised, not merely not-contradicted"


def test_pruning_is_ancestor_closed():
    """Every surviving node's whole path to the root survives with it."""
    network = arbor()
    _, kept = g.prune_to_order(network, 3)
    survived = set(kept.tolist())
    parent, _ = g.spanning_forest(network)
    walked = 0
    for node in survived:
        step = node
        while parent[step] >= 0:
            step = int(parent[step])
            walked += 1
            assert step in survived, f"node {node} survived but its ancestor {step} did not"
    assert walked > 100, "the walk has to actually traverse something"


def test_pruning_backs_off_rather_than_deleting_an_object():
    """A threshold past the tree's own height keeps it rather than emptying it."""
    pruned, _ = g.prune_to_order(perfect_tree(), 99, floor_nodes=2)
    assert pruned.node_count >= 2


def test_pruning_keeps_a_connected_graph():
    """Removing whole branches must not strand what is left."""
    pruned, _ = g.prune_to_order(arbor(), 3)
    assert _components(pruned) == 1


def test_douglas_peucker_keeps_the_endpoints_and_the_bend():
    """A straight run collapses to its ends; a kink survives at a tight epsilon."""
    straight = np.zeros((11, 3))
    straight[:, 0] = np.arange(11)
    assert np.flatnonzero(g.douglas_peucker(straight, 0.5)).tolist() == [0, 10]

    kinked = straight.copy()
    kinked[5, 1] = 3.0
    kept = np.flatnonzero(g.douglas_peucker(kinked, 0.5)).tolist()
    assert kept[0] == 0 and kept[-1] == 10
    assert 5 in kept, "the bend is the whole reason this run cannot be one segment"


def test_douglas_peucker_at_zero_epsilon_keeps_everything():
    """Epsilon 0 is 'do not simplify', not 'simplify maximally'."""
    points = np.random.default_rng(0).normal(size=(20, 3))
    assert g.douglas_peucker(points, 0.0).all()


def test_simplify_relinks_survivors_instead_of_inheriting_edges():
    """The bug this catches: an induced subgraph loses every edge along a straightened run.

    ``A-x-y-B`` becomes ``A-B``, and every original edge on that run has a dropped endpoint --
    so inheriting edges would leave the survivors connected by nothing at all.
    """
    network = chain(11)
    simplified, _ = g.simplify(network, 0.5)
    assert simplified.node_count == 2, "a straight chain keeps only its ends"
    assert simplified.edge_count == 1, "and they must still be joined"


def test_simplify_preserves_connectivity_and_topology():
    """A straightened arbor is still one connected tree with the same branch points."""
    network = arbor()
    before = g.degrees(network)
    simplified, _ = g.simplify(network, 1.0)
    after = g.degrees(simplified)
    assert _components(simplified) == 1
    assert int((before != 2).sum()) == int((after != 2).sum()), (
        "simplification drops only degree-2 interiors, so the count of branch points and "
        "terminals is unchanged"
    )


def test_simplify_never_moves_a_node():
    """Every surviving node is at exactly the coordinate it came in at."""
    network = arbor()
    simplified, kept = g.simplify(network, 1.0)
    assert np.array_equal(simplified.nodes, network.nodes[kept])


def test_deviation_is_bounded_by_the_epsilon_that_produced_it():
    """Douglas-Peucker's own metric is what `lod_error` is spent as, so they must agree."""
    network = arbor()
    for epsilon in (0.25, 1.0, 4.0):
        simplified, kept = g.simplify(network, epsilon)
        assert g.polyline_deviation(network, simplified, kept) <= epsilon + 1e-9


def test_unbranched_runs_cover_every_edge_exactly_once():
    """The run decomposition is what simplification rebuilds edges from, so it must be total."""
    network = arbor()
    seen: list[tuple[int, int]] = []
    for run in g.unbranched_runs(network):
        seen.extend((min(a, b), max(a, b)) for a, b in pairwise(run))
    original = {(min(a, b), max(a, b)) for a, b in network.edges.tolist()}
    assert set(seen) == original
    assert len(seen) == len(original), "an edge covered twice would be simplified twice"


def test_a_pure_loop_is_still_found():
    """A cycle of degree-2 nodes has no endpoint to walk from, and must not be dropped."""
    count = 8
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    nodes = np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=1) * 10 + 50
    edges = np.array([[i, (i + 1) % count] for i in range(count)])
    network = coerce_network((nodes, edges))
    runs = g.unbranched_runs(network)
    assert runs, "a closed loop is a run"
    assert {node for run in runs for node in run} == set(range(count))


def _components(network: Network) -> int:
    """How many connected pieces a network is in."""
    adjacency = g.neighbour_lists(network)
    seen: set[int] = set()
    pieces = 0
    for start in range(network.node_count):
        if start in seen:
            continue
        pieces += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return pieces
