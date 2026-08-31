"""The graph operations a level is built out of: Strahler order, runs, and Douglas-Peucker.

Two of these are the format's whole LOD story and the third is what makes the first two safe.

**Nothing here ever moves a node.** Pruning removes nodes; Douglas-Peucker removes nodes. Both
hand back a *subset* of what they were given, at exactly the coordinates it came in at. That is
worth stating up front because it is the property everything downstream leans on: a coarse level
is a sub-graph of the fine one rather than an approximation of it, ``lod_error`` bounds how far
the drawn *polyline* strayed and never how far a node strayed, and a node's identity survives
coarsening for free -- which is what lets a ghost in one cell be matched to its owner in another.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from konnektion.sources import Network

# --------------------------------------------------------------------------- #
# adjacency
# --------------------------------------------------------------------------- #


def neighbour_lists(network: Network) -> list[list[int]]:
    """Undirected adjacency, one list of neighbours per node.

    Plain Python lists rather than a CSR pair: every consumer here walks the structure node by
    node, and the graphs this runs on are thousands to millions of nodes, where the constant
    factor of an index lookup is not what decides anything.
    """
    lists: list[list[int]] = [[] for _ in range(network.node_count)]
    for a, b in network.edges.tolist():
        if a == b:
            continue
        lists[a].append(b)
        lists[b].append(a)
    return lists


def degrees(network: Network) -> npt.NDArray[np.int64]:
    """How many distinct neighbours each node has."""
    counts = np.zeros(network.node_count, dtype=np.int64)
    for node, neighbours in enumerate(neighbour_lists(network)):
        counts[node] = len(set(neighbours))
    return counts


def spanning_forest(network: Network) -> tuple[npt.NDArray[np.int64], list[int]]:
    """A BFS parent array and the roots it grew from.

    ``parent[i] == -1`` marks a root. The network's own ``root`` seeds the component it belongs
    to; every other component is seeded by its lowest-numbered node.

    A forest rather than a tree because **the input need not be one**. A dendritic arbor is a
    tree; vasculature loops; a connectome is not remotely one. Strahler order is defined on
    rooted trees, so it is computed over this forest, and the edges the forest left out --
    chords -- take no part in it. They are kept whenever both their endpoints survive, which
    means a loop coarsens as its two sides do and never dangles.
    """
    adjacency = neighbour_lists(network)
    parent = np.full(network.node_count, -1, dtype=np.int64)
    seen = np.zeros(network.node_count, dtype=bool)
    roots: list[int] = []

    seeds = list(range(network.node_count))
    if network.root is not None:
        seeds = [network.root, *seeds]

    for seed in seeds:
        if seen[seed]:
            continue
        seen[seed] = True
        roots.append(seed)
        queue = [seed]
        while queue:
            node = queue.pop()
            for neighbour in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    parent[neighbour] = node
                    queue.append(neighbour)
    return parent, roots


def _children_of(parent: npt.NDArray[np.int64]) -> list[list[int]]:
    """Invert a parent array into per-node child lists."""
    children: list[list[int]] = [[] for _ in range(len(parent))]
    for node, up in enumerate(parent.tolist()):
        if up >= 0:
            children[up].append(node)
    return children


def _post_order(roots: Sequence[int], children: Sequence[Sequence[int]]) -> list[int]:
    """Nodes with every child before its parent, iteratively.

    Iterative rather than recursive on purpose: a traced vessel tree is routinely tens of
    thousands of nodes deep along one path, and Python's recursion limit is 1000.
    """
    order: list[int] = []
    for root in roots:
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(children[node])
    order.reverse()
    return order


# --------------------------------------------------------------------------- #
# Strahler
# --------------------------------------------------------------------------- #


def strahler_orders(network: Network) -> npt.NDArray[np.int64]:
    """The Horton-Strahler order of every node, over the BFS spanning forest.

    A leaf is order 1. An internal node takes its children's maximum order, plus one when **two
    or more** children share that maximum -- so order rises only where branches of equal weight
    meet, which is what makes it a measure of how much tree hangs below a node rather than of
    depth.

    Invented for river networks, standard in neuron morphometry, and the reason it is the right
    LOD metric here rather than branch length or radius:

    **It is non-decreasing toward the root**, since a parent takes at least its largest child's
    order. So thresholding on it -- keep every node of order >= k -- is **ancestor-closed by
    construction**: a kept node's parent has an order at least as large and is therefore kept
    too. The invariant the format exists to protect falls out of the metric instead of being
    imposed on top of it, and :func:`prune_to_order` needs no repair pass. It is still verified,
    because a builder can have a bug that a proof does not prevent.
    """
    parent, roots = spanning_forest(network)
    children = _children_of(parent)
    orders = np.ones(network.node_count, dtype=np.int64)

    for node in _post_order(roots, children):
        kids = children[node]
        if not kids:
            continue
        child_orders = [int(orders[kid]) for kid in kids]
        best = max(child_orders)
        orders[node] = best + 1 if child_orders.count(best) > 1 else best
    return orders


# --------------------------------------------------------------------------- #
# subsetting
# --------------------------------------------------------------------------- #


def subset(network: Network, keep: npt.NDArray[np.bool_]) -> tuple[Network, npt.NDArray[np.int64]]:
    """The sub-network induced by a node mask, plus the old-index-per-new-index map.

    An edge survives when **both** its endpoints do. That is the only sane rule and it is worth
    naming: keeping an edge with one endpoint gone would need a new node to put the loose end on,
    and inventing a node is precisely what this format refuses to do.
    """
    kept = np.flatnonzero(keep)
    remap = np.full(network.node_count, -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)

    if network.edge_count:
        alive = keep[network.edges[:, 0]] & keep[network.edges[:, 1]]
        edges = remap[network.edges[alive]]
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    root = None
    if network.root is not None and keep[network.root]:
        root = int(remap[network.root])

    return (
        Network(
            nodes=network.nodes[kept],
            edges=edges,
            radii=None if network.radii is None else network.radii[kept],
            node_ids=network.ids()[kept],
            root=root,
        ),
        kept,
    )


def prune_to_order(
    network: Network, threshold: int, *, floor_nodes: int = 2
) -> tuple[Network, npt.NDArray[np.int64]]:
    """Drop every branch below a Strahler order, backing off rather than deleting the object.

    ``threshold`` of 1 or less keeps everything. Above that, the mask is ancestor-closed for
    free (see :func:`strahler_orders`), so this is a subset and nothing else.

    **The back-off is the interesting part.** A threshold high enough to take an object below
    ``floor_nodes`` would not coarsen it, it would *remove* it -- and an object that vanishes at
    level 2 and returns at level 3 is worse than one drawn coarsely, because a viewer panning
    across levels sees it blink. So the threshold is lowered until the object survives, and the
    level simply holds a less-coarsened version of that object than of its neighbours.
    """
    if threshold <= 1 or not network.node_count:
        return network, np.arange(network.node_count, dtype=np.int64)

    orders = strahler_orders(network)
    for level in range(int(threshold), 0, -1):
        keep = orders >= level
        if int(keep.sum()) >= floor_nodes:
            return subset(network, keep)
    return network, np.arange(network.node_count, dtype=np.int64)


# --------------------------------------------------------------------------- #
# runs and Douglas-Peucker
# --------------------------------------------------------------------------- #


def unbranched_runs(network: Network) -> list[list[int]]:
    """Maximal paths whose interior nodes have exactly two neighbours.

    The unit Douglas-Peucker operates on. A run's endpoints are branch points (degree 3+),
    terminals (degree 1) or isolated nodes; everything between them is a chain that can be
    straightened without touching the topology.

    Cycles with no branch point anywhere -- a closed loop of degree-2 nodes -- have no endpoint
    to start from and are returned as a run that begins and ends at the same node, so that at
    least one node of the loop is always pinned and the loop cannot collapse to nothing.
    """
    adjacency = [sorted(set(n)) for n in neighbour_lists(network)]
    degree = np.array([len(n) for n in adjacency], dtype=np.int64)
    runs: list[list[int]] = []
    walked: set[tuple[int, int]] = set()

    def walk(start: int, first: int) -> list[int]:
        run = [start, first]
        previous, node = start, first
        while degree[node] == 2:
            following = [n for n in adjacency[node] if n != previous]
            if not following:
                break
            previous, node = node, following[0]
            run.append(node)
            if node == start:  # closed the loop
                break
        return run

    for node in range(network.node_count):
        if degree[node] == 2:
            continue
        for neighbour in adjacency[node]:
            if (node, neighbour) in walked:
                continue
            run = walk(node, neighbour)
            walked.add((node, neighbour))
            walked.add((run[-1], run[-2]))
            runs.append(run)

    # Loops of pure degree-2 nodes are unreachable from any endpoint, so they are found here.
    covered = {node for run in runs for node in run}
    for node in range(network.node_count):
        if node in covered or degree[node] != 2:
            continue
        run = walk(node, adjacency[node][0])
        covered.update(run)
        runs.append(run)
    return runs


def douglas_peucker(points: npt.NDArray[np.float64], epsilon: float) -> npt.NDArray[np.bool_]:
    """Which points of a polyline survive simplification at ``epsilon`` voxels.

    The classic recursive split, iterative here for the same depth reason as
    :func:`_post_order`. The two endpoints always survive; an interior point survives when the
    polyline could not be drawn without it to within ``epsilon``.

    **Its error metric is perpendicular distance from the chord**, which is exactly what
    ``lod_error`` is spent as -- so the epsilon handed in *is* the bound, and the level does not
    need a second measurement to declare one. That coincidence is why Douglas-Peucker rather
    than, say, dropping every other node.
    """
    count = len(points)
    keep = np.zeros(count, dtype=bool)
    if count == 0:
        return keep
    if epsilon <= 0.0:
        keep[:] = True
        return keep
    keep[0] = keep[-1] = True
    if count <= 2:
        return keep

    stack = [(0, count - 1)]
    while stack:
        start, stop = stack.pop()
        if stop <= start + 1:
            continue
        anchor = points[start]
        chord = points[stop] - anchor
        span = float(np.linalg.norm(chord))
        offsets = points[start + 1 : stop] - anchor
        if span == 0.0:
            # A closed run: distance from the shared endpoint, since there is no chord.
            distances = np.linalg.norm(offsets, axis=1)
        else:
            distances = np.linalg.norm(np.cross(offsets, chord / span), axis=1)
        worst = int(np.argmax(distances))
        if float(distances[worst]) > epsilon:
            split = start + 1 + worst
            keep[split] = True
            stack.append((start, split))
            stack.append((split, stop))
    return keep


def simplify(
    network: Network, epsilon: float
) -> tuple[Network, npt.NDArray[np.int64]]:
    """Straighten every unbranched run, keeping branch points and terminals.

    Topology-preserving by construction: only interior nodes of a degree-2 chain are ever
    dropped, so no branch point moves, no branch is lost, and the graph a coarse level draws has
    the same shape as the fine one -- fewer bends, same tree.

    **The edges are rebuilt, not inherited.** This is the one place where the induced-subgraph
    rule :func:`subset` applies is exactly wrong: straightening ``A-x-y-B`` to ``A-B`` drops both
    ``x`` and ``y``, and every original edge along that run has at least one dropped endpoint, so
    inheriting edges would leave the survivors connected by nothing. What simplification means is
    that consecutive *survivors* along a run are joined, which is a new edge standing for the
    piece of polyline between them. Pruning is the opposite case and does inherit: removing a
    whole branch removes its edges too, and no re-linking is wanted or correct.
    """
    if epsilon <= 0.0 or not network.node_count:
        return network, np.arange(network.node_count, dtype=np.int64)

    keep = np.zeros(network.node_count, dtype=bool)
    degree = degrees(network)
    keep[degree != 2] = True  # branch points, terminals and isolated nodes are pinned

    runs = unbranched_runs(network)
    for run in runs:
        if len(run) <= 2:
            keep[run] = True
            continue
        indices = np.asarray(run, dtype=np.int64)
        survivors = douglas_peucker(network.nodes[indices], epsilon)
        keep[indices[survivors]] = True

    kept = np.flatnonzero(keep)
    remap = np.full(network.node_count, -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)

    rebuilt: set[tuple[int, int]] = set()
    covered: set[tuple[int, int]] = set()
    for run in runs:
        for a, b in pairwise(run):
            covered.add((min(a, b), max(a, b)))
        survivors = [node for node in run if keep[node]]
        for a, b in pairwise(survivors):
            if a != b:
                rebuilt.add((min(int(remap[a]), int(remap[b])), max(int(remap[a]), int(remap[b]))))

    # Anything the run walk did not cover -- it should cover every edge, and this is the guard
    # that says so rather than assuming it. Such an edge survives on the induced-subgraph rule.
    for a, b in network.edges.tolist():
        if (min(a, b), max(a, b)) in covered or a == b:
            continue
        if keep[a] and keep[b]:
            rebuilt.add((min(int(remap[a]), int(remap[b])), max(int(remap[a]), int(remap[b]))))

    edges = (
        np.array(sorted(rebuilt), dtype=np.int64).reshape(-1, 2)
        if rebuilt
        else np.zeros((0, 2), dtype=np.int64)
    )
    root = None
    if network.root is not None and keep[network.root]:
        root = int(remap[network.root])

    return (
        Network(
            nodes=network.nodes[kept],
            edges=edges,
            radii=None if network.radii is None else network.radii[kept],
            node_ids=network.ids()[kept],
            root=root,
        ),
        kept,
    )


# --------------------------------------------------------------------------- #
# error
# --------------------------------------------------------------------------- #


def polyline_deviation(
    fine: Network, coarse: Network, kept: npt.NDArray[np.int64]
) -> float:
    """How far the coarse polyline strays from the fine one, in voxels.

    Measured as the largest distance from a **dropped** node to the segment that replaced it --
    not from a dropped node to the nearest surviving node, which is the tempting version and is
    wrong for the same reason it is wrong for a mesh: a long straight run drops every interior
    node, and each one can sit exactly on the line that replaced it while being far from either
    end of it. What a renderer shows, and what ``lod_error`` is spent against, is deviation from
    the drawn geometry.
    """
    if not fine.node_count or not coarse.node_count:
        return 0.0
    survived = np.zeros(fine.node_count, dtype=bool)
    survived[kept] = True
    dropped = np.flatnonzero(~survived)
    if not len(dropped):
        return 0.0

    old_to_new = np.full(fine.node_count, -1, dtype=np.int64)
    old_to_new[kept] = np.arange(len(kept), dtype=np.int64)

    worst = 0.0
    for run in unbranched_runs(fine):
        indices = np.asarray(run, dtype=np.int64)
        anchors = [i for i in indices.tolist() if survived[i]]
        if len(anchors) < 2:
            continue
        for start, stop in pairwise(anchors):
            lo, hi = indices.tolist().index(start), indices.tolist().index(stop)
            interior = indices[lo + 1 : hi]
            interior = interior[~survived[interior]]
            if not len(interior):
                continue
            a = coarse.nodes[old_to_new[start]]
            b = coarse.nodes[old_to_new[stop]]
            chord = b - a
            span = float(np.linalg.norm(chord))
            offsets = fine.nodes[interior] - a
            if span == 0.0:
                distances = np.linalg.norm(offsets, axis=1)
            else:
                distances = np.linalg.norm(np.cross(offsets, chord / span), axis=1)
            worst = max(worst, float(distances.max()))
    return worst


__all__ = [
    "degrees",
    "douglas_peucker",
    "neighbour_lists",
    "polyline_deviation",
    "prune_to_order",
    "simplify",
    "spanning_forest",
    "strahler_orders",
    "subset",
    "unbranched_runs",
]
