"""networkx graphs in and out, and everything a graph may hold that a network may not."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import konnektion
from konnektion.contrib.networkx import from_networkx, to_networkx
from konnektion.errors import FormatError
from konnektion.sources import Network
from tests.conftest import arbor, chain


def path(count: int = 4) -> nx.Graph:
    """A path whose nodes sit along x, well inside the positive octant."""
    graph = nx.path_graph(count)
    for node in graph.nodes:
        graph.nodes[node]["pos"] = (10.0 + node, 20.0, 30.0)
    return graph


def undirected(edges: np.ndarray) -> set[frozenset[int]]:
    """An edge list as a set of unordered pairs, which is all an undirected graph promises."""
    return {frozenset(edge) for edge in edges.tolist()}


def test_a_network_survives_the_round_trip():
    """Positions, radii, ids, the root and the edges all come back as they went."""
    source = arbor(depth=4)
    graph = to_networkx(source, directed=True)
    network, ids = from_networkx(graph, radius="radius", root=graph.graph["root"])

    np.testing.assert_array_equal(network.nodes, source.nodes)
    np.testing.assert_array_equal(network.radii, source.radii)
    np.testing.assert_array_equal(network.ids(), source.ids())
    assert network.root == source.root
    assert sorted(network.edges.tolist()) == sorted(source.edges.tolist())
    assert ids == {node: node for node in range(source.node_count)}


def test_an_undirected_round_trip_keeps_the_edges_as_pairs():
    """An undirected graph has no start node to keep, so the pairs are what is compared."""
    source = arbor(depth=3)
    network, _ = from_networkx(to_networkx(source))
    assert undirected(network.edges) == undirected(source.edges)


def test_attributes_ride_out_and_back():
    """A `NaN` leaves the graph as an absent key and returns as a `NaN`."""
    source = chain(4)
    measured = np.array([1.0, np.nan, 3.0, 4.0])
    source = Network(nodes=source.nodes, edges=source.edges, attributes={"tortuosity": measured})

    graph = to_networkx(source)
    assert "tortuosity" not in graph.nodes[1] and graph.nodes[2]["tortuosity"] == 3.0

    network, _ = from_networkx(graph, attributes=["tortuosity"])
    assert network.attributes is not None
    np.testing.assert_array_equal(network.attributes["tortuosity"], measured)


def test_a_converted_graph_builds_and_verifies():
    """The point of the adapter: a networkx graph becomes a collection that verifies."""
    graph = to_networkx(arbor(), directed=True)
    for node in graph.nodes:
        graph.nodes[node]["load"] = float(node % 7)
    network, _ = from_networkx(
        graph, radius="radius", attributes=["load"], root=graph.graph["root"]
    )

    built = konnektion.build_collection({1: network}, cell_size=(128, 128, 128), axes=("x", "y", "z"))
    store = konnektion.MemoryStore()
    built.write(store, "p")
    report = konnektion.verify(konnektion.open_collection(store, "p"), tier="topology")

    assert report.ok, str(report)
    assert "load" in {attribute.name for attribute in built.manifest().attributes}


def test_axes_say_how_the_position_is_stored():
    """Array order is reversed into ``(x, y, z)`` by naming it, not by slicing."""
    graph = nx.Graph()
    graph.add_node(0, pos=(3.0, 2.0, 1.0))
    network, _ = from_networkx(graph, axes="zyx")
    assert network.nodes.tolist() == [[1.0, 2.0, 3.0]]


def test_a_planar_position_leaves_the_missing_axis_at_zero():
    """A layout is two-dimensional by default, and which two is the caller's to say."""
    graph = nx.Graph()
    graph.add_node(0, where=(5.0, 7.0))
    network, _ = from_networkx(graph, pos="where", axes="yx")
    assert network.nodes.tolist() == [[7.0, 5.0, 0.0]]


def test_positions_may_come_from_a_layout():
    """A layout placed in the positive octant by its caller builds like any other graph."""
    graph = nx.balanced_tree(2, 4)
    layout = nx.spring_layout(graph, dim=3, center=(100.0, 100.0, 100.0), scale=50.0, seed=1)
    network, _ = from_networkx(graph, pos=layout)
    built = konnektion.build_collection({1: network}, cell_size=(128, 128, 128), axes=("x", "y", "z"))
    assert network.nodes.min() >= 50.0 and built.manifest().counts["objects"] == 1


def test_the_node_keys_of_a_graph_may_not_collide():
    """An attribute called `pos` would overwrite every position on the way out."""
    source = chain(3)
    source = Network(nodes=source.nodes, edges=source.edges, attributes={"pos": np.ones(3)})
    with pytest.raises(FormatError, match="must differ"):
        to_networkx(source)
    assert to_networkx(source, pos="position").nodes[0]["pos"] == 1.0


def test_integer_keys_are_the_node_ids():
    """A tracer's numbering is data, so it is kept rather than renumbered."""
    graph = nx.Graph()
    graph.add_node(40, pos=(1.0, 1.0, 1.0))
    graph.add_node(7, pos=(2.0, 1.0, 1.0))
    graph.add_edge(40, 7)

    network, ids = from_networkx(graph, root=7)
    assert network.ids().tolist() == [40, 7] and ids == {40: 40, 7: 7}
    assert network.edges.tolist() == [[0, 1]] and network.root == 1


def test_other_keys_are_numbered_and_the_numbering_is_returned():
    """A string cannot be an id, so it gets one -- and the caller is told which."""
    graph = nx.Graph()
    graph.add_node("soma", pos=(1.0, 1.0, 1.0))
    graph.add_node(("tip", 2), pos=(2.0, 1.0, 1.0))
    graph.add_edge("soma", ("tip", 2))

    network, ids = from_networkx(graph, root="soma")
    assert ids == {"soma": 0, ("tip", 2): 1}
    assert network.ids().tolist() == [0, 1] and network.root == 0


def test_mixed_keys_are_all_numbered():
    """Keeping the integer among strings would let it collide with a numbered one."""
    graph = nx.Graph()
    graph.add_node("a", pos=(1.0, 1.0, 1.0))
    graph.add_node(0, pos=(2.0, 1.0, 1.0))
    _, ids = from_networkx(graph)
    assert ids == {"a": 0, 0: 1}


def test_a_directed_graph_keeps_its_direction():
    """A segment is coloured by its start node, so which end is the start is data."""
    graph = nx.DiGraph()
    graph.add_node(0, pos=(1.0, 1.0, 1.0))
    graph.add_node(1, pos=(2.0, 1.0, 1.0))
    graph.add_edge(1, 0)
    network, _ = from_networkx(graph)
    assert network.edges.tolist() == [[1, 0]]


def test_an_empty_graph_is_an_empty_network():
    """No nodes is a legal object, and its arrays still have their shapes."""
    network, ids = from_networkx(nx.Graph())
    assert network.nodes.shape == (0, 3) and network.edges.shape == (0, 2) and ids == {}


def test_nothing_is_carried_that_was_not_named():
    """A graph's dictionaries hold whatever a pipeline left there, most of it not for the wire."""
    graph = path()
    nx.set_node_attributes(graph, "axon", "label")
    nx.set_node_attributes(graph, 2.0, "radius")
    network, _ = from_networkx(graph)
    assert network.attributes is None and network.radii is None


def test_a_node_without_a_position_is_named():
    """One unplaced node is one too many, and the error says which."""
    graph = path()
    del graph.nodes[2]["pos"]
    with pytest.raises(FormatError, match="Node 2 has no 'pos'"):
        from_networkx(graph)


def test_a_position_must_match_the_axes():
    """Three components read as two would silently drop one."""
    with pytest.raises(FormatError, match="3 components"):
        from_networkx(path(), axes="xy")


@pytest.mark.parametrize("axes", ["x", "xyzx", "xx", "xyw"])
def test_axes_are_two_or_three_distinct_letters(axes: str):
    """Anything else has no reading."""
    with pytest.raises(FormatError, match="`axes`"):
        from_networkx(path(), axes=axes)


def test_a_radius_is_on_every_node_or_on_none():
    """The format has no value for a missing radius."""
    graph = path()
    graph.nodes[0]["radius"] = 1.5
    with pytest.raises(FormatError, match="Node 1 has no 'radius'"):
        from_networkx(graph, radius="radius")


def test_an_attribute_that_is_not_a_number_is_refused():
    """Including a string that would parse as one."""
    graph = path()
    graph.nodes[1]["kind"] = "3.5"
    with pytest.raises(FormatError, match="not a number"):
        from_networkx(graph, attributes=["kind"])


def test_an_attribute_on_no_node_is_refused():
    """A column of nothing is a misspelt name."""
    with pytest.raises(FormatError, match="on no node"):
        from_networkx(path(), attributes=["tortuosty"])


def test_core_still_guards_the_attribute_names():
    """The adapter repeats none of the gate's checks; it goes through the gate."""
    graph = path()
    nx.set_node_attributes(graph, 1.0, "Bad Name")
    with pytest.raises(FormatError, match="lowercase"):
        from_networkx(graph, attributes=["Bad Name"])


def test_a_root_must_be_a_node_of_the_graph():
    """It is a key here, not an index."""
    with pytest.raises(FormatError, match="`root`"):
        from_networkx(path(), root="soma")


def test_a_multigraph_is_refused():
    """Parallel edges would be stored and then counted as one."""
    with pytest.raises(FormatError, match="multigraph"):
        from_networkx(nx.MultiGraph(path()))


def test_a_self_loop_is_refused():
    """A segment of no length draws nothing and counts for nothing."""
    graph = path()
    graph.add_edge(1, 1)
    with pytest.raises(FormatError, match="edge to itself"):
        from_networkx(graph)


class Skeleton(nx.Graph):
    """A caller's own graph class, which lives in the caller's module."""


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, Skeleton])
def test_a_bare_graph_is_refused_at_the_gate_with_the_adapter_named(kind: type):
    """It has `nodes` and `edges`, so without this it is turned away for its *shape*."""
    with pytest.raises(FormatError, match=r"konnektion\.contrib\.networkx\.from_networkx"):
        konnektion.build_collection({1: kind(path())}, axes=("x", "y", "z"))
