"""What a caller may hand in, and everything that is turned away at the gate."""

from __future__ import annotations

import numpy as np
import pytest

from konnektion.errors import FormatError
from konnektion.sources import coerce_network, coerce_objects

NODES = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
EDGES = np.array([[0, 1], [1, 2]])


def test_a_bare_pair_is_accepted():
    """The least a caller can hand over."""
    network = coerce_network((NODES, EDGES))
    assert network.node_count == 3 and network.edge_count == 2


def test_a_mapping_is_accepted():
    """Including the optional per-node radius."""
    network = coerce_network({"nodes": NODES, "edges": EDGES, "radii": [1.0, 2.0, 3.0]})
    assert network.radii is not None and network.radii.tolist() == [1.0, 2.0, 3.0]


def test_an_object_with_nodes_and_edges_is_accepted():
    """Structural, so a caller's own class works without importing anything from konnektion."""

    class Mine:
        nodes = NODES
        edges = EDGES

    assert coerce_network(Mine()).node_count == 3


def test_an_empty_graph_is_accepted():
    """A traced object with no edges is a point set, not an error."""
    network = coerce_network((NODES, np.zeros((0, 2))))
    assert network.edge_count == 0


def test_nodes_must_be_three_dimensional():
    """The octree is three-dimensional and the rank is not a parameter."""
    with pytest.raises(FormatError, match=r"\(n, 3\)"):
        coerce_network((np.zeros((4, 2)), np.zeros((0, 2))))


def test_a_triangle_list_is_refused_rather_than_reshaped():
    """The mistake this format most expects, and the one it must never absorb quietly.

    ``reshape(-1, 2)`` would turn six triangle indices into three segments without a murmur --
    the exact silent failure ``encoding.edges`` exists to prevent, committed by the gate meant
    to prevent it. It is refused instead, and the message says why.
    """
    with pytest.raises(FormatError, match="triangle list"):
        coerce_network((NODES, np.zeros((2, 3), dtype=int)))


def test_a_flat_edge_list_is_accepted():
    """One unambiguous reading, so it needs no guess."""
    assert coerce_network((NODES, [0, 1, 1, 2])).edge_count == 2


def test_an_odd_flat_edge_list_is_refused():
    """Half an edge is not an edge."""
    with pytest.raises(FormatError, match="even"):
        coerce_network((NODES, [0, 1, 1]))


def test_a_dangling_edge_index_is_refused():
    """It is in range for *some* cell downstream, so nothing later would catch it."""
    with pytest.raises(FormatError, match="dangling edge"):
        coerce_network((NODES, np.array([[0, 7]])))


def test_radii_are_one_per_node():
    """A radius belongs to a node, so the two counts cannot differ."""
    with pytest.raises(FormatError, match="one per node"):
        coerce_network({"nodes": NODES, "edges": EDGES, "radii": [1.0]})


def test_a_negative_radius_is_refused():
    """A radius is a length."""
    with pytest.raises(FormatError, match="cannot be negative"):
        coerce_network({"nodes": NODES, "edges": EDGES, "radii": [1.0, -2.0, 3.0]})


def test_node_ids_must_be_unique_within_an_object():
    """A repeated id would make a ghost ambiguous -- two nodes answer to the same name."""
    with pytest.raises(FormatError, match="unique"):
        coerce_network({"nodes": NODES, "edges": EDGES, "node_ids": [1, 1, 2]})


def test_node_ids_need_not_be_unique_across_objects():
    """A tracer numbers each neuron from one, and rewriting that would break a join key."""
    objects = coerce_objects(
        {
            1: {"nodes": NODES, "edges": EDGES, "node_ids": [0, 1, 2]},
            2: {"nodes": NODES, "edges": EDGES, "node_ids": [0, 1, 2]},
        }
    )
    assert objects[1].ids().tolist() == objects[2].ids().tolist() == [0, 1, 2]


def test_ids_default_to_dense_from_zero():
    """A caller with no ids of its own gets the obvious ones."""
    assert coerce_network((NODES, EDGES)).ids().tolist() == [0, 1, 2]


def test_a_root_must_index_a_node():
    """The ancestor-closed invariant is stated relative to it, so it has to exist."""
    with pytest.raises(FormatError, match="root"):
        coerce_network({"nodes": NODES, "edges": EDGES, "root": 9})


def test_a_failing_object_is_named():
    """A collection is a mapping, so the error has to say which entry was wrong."""
    with pytest.raises(FormatError, match="Object 7"):
        coerce_objects({7: (np.zeros((2, 2)), np.zeros((0, 2)))})


def test_bounds_of_an_empty_network_are_zero():
    """An empty object has no extent, and asking for one must not raise."""
    assert coerce_network((np.zeros((0, 3)), np.zeros((0, 2)))).bounds.shape == (2, 3)
