"""networkx graphs in, networkx graphs out.

A networkx graph and a :class:`~konnektion.sources.Network` disagree about what a node is: a
hashable key with a dictionary on one side, a row of three coordinates on the other. So the
conversion is a set of statements the caller makes -- where the positions are, in what order,
which attributes come along -- and :func:`from_networkx` refuses to make any of them itself.
The rules it follows are the ones in :mod:`konnektion.contrib`.

    graph = nx.random_geometric_graph(200, 0.2, dim=3)
    network, ids = from_networkx(graph, pos="pos")
    collection = konnektion.build_collection({1: network}, axes=("x", "y", "z"))

**Edge order is networkx's, not the caller's.** ``graph.edges`` walks the adjacency, so an
undirected graph reports each edge once, from whichever endpoint it reaches first. A caller for
whom the direction of an edge is data -- a viewer colours a segment by its start node -- hands
over a ``DiGraph``, whose ``(u, v)`` is kept as given.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt

from konnektion.errors import FormatError, MissingExtraError
from konnektion.sources import Network, NetworkSource, coerce_network

try:
    import networkx as nx
except ImportError as _error:
    raise MissingExtraError(
        "networkx is required to convert a networkx graph. Install it with "
        "`pip install konnektion[networkx]`."
    ) from _error

#: The slot each axis letter names, which is the order konnektion stores positions in.
_SLOTS = "xyz"


def from_networkx(
    graph: nx.Graph[Any],
    *,
    pos: str | Mapping[Any, Sequence[float]] = "pos",
    axes: str = "xyz",
    radius: str | None = None,
    attributes: Sequence[str] = (),
    root: Hashable | None = None,
) -> tuple[Network, dict[Any, int]]:
    """Turn a networkx graph into a validated network, and say which id each node got.

    ``pos`` is the node attribute holding a position, or a mapping of node to position such as
    a layout function returns. ``axes`` names the components *as they are stored*: ``"zyx"``
    for positions in array order, ``"xy"`` for a planar layout, whose missing axis is 0.
    Positions are voxels and are taken as they are: a layout centred on the origin has to be
    placed in the positive octant by its caller (``nx.spring_layout(graph, dim=3, center=...,
    scale=...)``), because a collection cannot be built from negative coordinates.

    ``radius`` names the node attribute that becomes the network's radii, and ``attributes``
    the ones carried as per-node values; a node without a listed attribute gets ``NaN``.
    Nothing is collected that is not named. ``root`` is a node of the graph.

    Node keys that are all non-negative integers are the node ids. Any other keys are numbered
    from 0 in ``graph.nodes`` order. Either way the second return value maps key to id.
    """
    if graph.is_multigraph():
        raise FormatError(
            "A multigraph has parallel edges, which the format would store and then count as "
            "one: its degree and its unbranched runs are over distinct neighbours. Collapse "
            "it first with `nx.Graph(graph)` or `nx.DiGraph(graph)`."
        )

    keys: list[Any] = list(graph.nodes)
    index = {key: position for position, key in enumerate(keys)}
    ids = _node_ids(keys)

    edges: list[tuple[int, int]] = []
    for start, end in graph.edges:
        if start == end:
            raise FormatError(
                f"Node {start!r} has an edge to itself, which is a segment of no length. "
                f"Remove them with `graph.remove_edges_from(nx.selfloop_edges(graph))`."
            )
        edges.append((index[start], index[end]))

    radii = None
    if radius is not None:
        radii = np.asarray(
            [_number(graph, key, radius, required=True) for key in keys], dtype=np.float64
        )

    if isinstance(attributes, str):
        attributes = (attributes,)
    columns: dict[str, npt.NDArray[np.float64]] = {}
    for name in attributes:
        values = np.asarray(
            [_number(graph, key, name, required=False) for key in keys], dtype=np.float64
        )
        if len(values) and np.isnan(values).all():
            raise FormatError(
                f"Attribute {name!r} is on no node of the graph, so there is nothing to carry. "
                f"A column of nothing but `NaN` is more likely a misspelt name than data."
            )
        columns[name] = values

    root_index = None
    if root is not None:
        if root not in index:
            raise FormatError(f"`root` is a node of the graph, and {root!r} is not one.")
        root_index = index[root]

    network = coerce_network(
        Network(
            nodes=_positions(graph, keys, pos, axes),
            edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
            radii=radii,
            node_ids=np.asarray([ids[key] for key in keys], dtype=np.int64),
            root=root_index,
            attributes=columns or None,
        )
    )
    return network, ids


def to_networkx(
    network: NetworkSource,
    *,
    directed: bool = False,
    pos: str = "pos",
    radius: str = "radius",
) -> nx.Graph[int]:
    """Turn a network into a networkx graph keyed by node id.

    Each node carries its ``(x, y, z)`` position under ``pos``, its radius under ``radius``
    when the network has radii, and one entry per attribute column -- left out where the value
    is ``NaN``, which is the format's "no answer". The root's node id is ``graph.graph["root"]``.

    ``directed`` keeps each edge's ``(start, end)`` as stored. Without it the result is an
    undirected ``Graph``, in which an edge stored in both directions is one edge.
    """
    source = coerce_network(network)
    taken = {pos, radius} & set(source.attributes or {})
    if pos == radius or taken:
        raise FormatError(
            f"The node keys for the position ({pos!r}) and the radius ({radius!r}) must differ "
            f"from each other and from the network's attributes, or one would overwrite the "
            f"other on every node. Pass `pos=` or `radius=` to rename them."
        )
    graph: nx.Graph[int] = nx.DiGraph() if directed else nx.Graph()

    ids: list[int] = source.ids().tolist()
    positions: list[list[float]] = source.nodes.tolist()
    radii: list[float] | None = None if source.radii is None else source.radii.tolist()
    columns: dict[str, list[float]] = {
        name: values.tolist() for name, values in (source.attributes or {}).items()
    }
    for position, node_id in enumerate(ids):
        data: dict[str, Any] = {pos: tuple(positions[position])}
        if radii is not None:
            data[radius] = radii[position]
        for name, values in columns.items():
            if not math.isnan(values[position]):
                data[name] = values[position]
        graph.add_node(node_id, **data)

    graph.add_edges_from((ids[start], ids[end]) for start, end in source.edges.tolist())
    if source.root is not None:
        graph.graph["root"] = ids[source.root]
    return graph


def _node_ids(keys: Sequence[Any]) -> dict[Any, int]:
    """The id of every node key: the key itself where that is possible, its position if not.

    All or nothing. Keeping the integers of a graph whose other keys are strings would hand
    out ids that collide with the positions the strings are numbered by.
    """
    if all(_is_node_id(key) for key in keys):
        return {key: int(key) for key in keys}
    return {key: position for position, key in enumerate(keys)}


def _is_node_id(key: object) -> bool:
    """Whether a key can be a node id as it stands: a non-negative integer, and not a bool."""
    return isinstance(key, (int, np.integer)) and not isinstance(key, bool) and int(key) >= 0


def _number(graph: nx.Graph[Any], key: object, name: str, *, required: bool) -> float:
    """One node's value for a named attribute, as a float, or ``NaN`` where it has none."""
    value = graph.nodes[key].get(name)
    if value is None:
        if required:
            raise FormatError(
                f"Node {key!r} has no {name!r}. A radius is one per node, and the format has "
                f"no value for a missing one."
            )
        return math.nan
    if not isinstance(value, Real):
        raise FormatError(
            f"Node {key!r} has {name!r} = {value!r}, which is not a number. A per-node value "
            f"is a float; a label or a category belongs in a table keyed by node id."
        )
    return float(value)


def _positions(
    graph: nx.Graph[Any],
    keys: Sequence[Any],
    pos: str | Mapping[Any, Sequence[float]],
    axes: str,
) -> npt.NDArray[np.float64]:
    """The ``(n, 3)`` positions in ``(x, y, z)`` order, read as the caller said they are stored."""
    if len(axes) not in (2, 3) or len(set(axes)) != len(axes) or not set(axes) <= set(_SLOTS):
        raise FormatError(
            f"`axes` names the stored components with two or three of 'x', 'y' and 'z', each "
            f"once -- 'zyx' for array order, 'xy' for a planar layout -- got {axes!r}."
        )

    nodes = np.zeros((len(keys), 3), dtype=np.float64)
    for row, key in enumerate(keys):
        stored = graph.nodes[key].get(pos) if isinstance(pos, str) else pos.get(key)
        if stored is None:
            where = f"no {pos!r} attribute" if isinstance(pos, str) else "no entry in `pos`"
            raise FormatError(
                f"Node {key!r} has {where}. Every node of a network is a position in voxels; "
                f"pass the attribute that holds it, or a mapping of node to position."
            )
        try:
            components = np.asarray(stored, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise FormatError(f"Node {key!r} has a position that is not numbers: {error}") from error
        if len(components) != len(axes):
            raise FormatError(
                f"Node {key!r} has a position of {len(components)} components and `axes` "
                f"names {len(axes)} ({axes!r})."
            )
        for component, axis in zip(components, axes, strict=True):
            nodes[row, _SLOTS.index(axis)] = component
    return nodes


__all__ = ["from_networkx", "to_networkx"]
