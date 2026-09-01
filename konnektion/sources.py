"""What a caller hands in, and the shape it is coerced to before anything else looks at it.

One object of a collection is a :class:`Network`: node positions, an edge list, and optionally a
radius per node and a root. Everything downstream -- the octree assignment, the pruning, the
codecs -- reads that and nothing else, so this module is where a malformed input is turned away
while the error can still name the argument that was wrong.

**Positions are in voxels, ordered ``(x, y, z)``** -- slots 0, 1 and 2, which the format never
interprets. A collection extracted from a ``(z, y, x)`` image reverses its components before it
gets here and declares its ``axes`` accordingly; nothing below this line can detect a
permutation, so it is the caller's to get right and the manifest's to record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from konnektion.errors import FormatError
from konnektion.manifest import ATTRIBUTE_NAME_PATTERN


@runtime_checkable
class HasNodesAndEdges(Protocol):
    """Anything carrying the two arrays a network is.

    Structural, so a caller's own class works without importing anything from konnektion --
    the same reason fabriks accepts a ``trimesh.Trimesh`` without depending on trimesh.
    """

    @property
    def nodes(self) -> Any:  # noqa: ANN401
        """The ``(n, 3)`` node positions, in voxels."""
        ...

    @property
    def edges(self) -> Any:  # noqa: ANN401
        """The ``(m, 2)`` edge endpoints, as indices into ``nodes``."""
        ...


@dataclass(frozen=True)
class Network:
    """One object of a collection: nodes, edges, and what hangs off them.

    ``node_ids`` are **global** and are what a ghost copy is matched by, so they have to be
    unique within the object. Left unset they are assigned densely from 0 at coercion time,
    which is right for a caller with no ids of its own and wrong for one that has them -- a
    tracer's node numbering is data, and re-deriving it would silently break a join to an
    attributes table keyed on it.

    ``root`` is the node the ancestor-closed invariant is stated relative to: the soma of an
    arbor, the inlet of a vessel tree. ``None`` says this object has no distinguished root --
    a connectome component rather than a rooted tree -- and connectivity is then checked per
    connected component instead, which is weaker only in that it cannot say which way is *up*.

    ``attributes`` are extra per-node floats the caller measured -- a tortuosity, a distance to
    a landmark -- carried through coarsening by the same subset rule as ``radii`` and declared
    in the manifest so a reader knows to look. ``NaN`` is the value for "this node has no
    answer"; the builder adds its own computed columns (Strahler order, degree, depth,
    component) beside them.
    """

    nodes: npt.NDArray[np.float64]
    edges: npt.NDArray[np.int64]
    radii: npt.NDArray[np.float64] | None = None
    node_ids: npt.NDArray[np.int64] | None = None
    root: int | None = None
    attributes: dict[str, npt.NDArray[np.float64]] | None = None

    @property
    def node_count(self) -> int:
        """How many nodes this object has."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """How many edges this object has."""
        return len(self.edges)

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """The ``(2, 3)`` min/max box of the node positions."""
        if not len(self.nodes):
            return np.zeros((2, 3), dtype=np.float64)
        return np.stack([self.nodes.min(axis=0), self.nodes.max(axis=0)])

    def ids(self) -> npt.NDArray[np.int64]:
        """The global node ids, dense from 0 when the caller supplied none."""
        if self.node_ids is not None:
            return self.node_ids
        return np.arange(self.node_count, dtype=np.int64)


#: What :func:`coerce_network` accepts: a :class:`Network`, anything with ``nodes``/``edges``, or
#: a plain ``(nodes, edges)`` pair.
NetworkSource = Any


def coerce_network(source: NetworkSource) -> Network:
    """Turn whatever a caller handed over into a validated :class:`Network`.

    The gate every object passes through. It is deliberately strict about shape and about the
    one thing that is cheap here and expensive later: an edge naming a node that does not exist.
    A dangling edge index survives cell assignment (it is just an integer), survives the codecs
    (it is in range for *some* cell), and surfaces as a segment drawn to the wrong place.
    """
    radii = None
    node_ids = None
    root = None
    attributes = None

    if isinstance(source, Network):
        nodes, edges, radii, node_ids, root, attributes = (
            source.nodes,
            source.edges,
            source.radii,
            source.node_ids,
            source.root,
            source.attributes,
        )
    elif isinstance(source, HasNodesAndEdges):
        nodes, edges = source.nodes, source.edges
        radii = getattr(source, "radii", None)
        node_ids = getattr(source, "node_ids", None)
        root = getattr(source, "root", None)
        attributes = getattr(source, "attributes", None)
    elif isinstance(source, Mapping):
        try:
            nodes, edges = source["nodes"], source["edges"]
        except KeyError as error:
            raise FormatError(
                f"A network given as a mapping carries `nodes` and `edges`, got "
                f"{sorted(source)}."
            ) from error
        radii = source.get("radii")
        node_ids = source.get("node_ids")
        root = source.get("root")
        attributes = source.get("attributes")
    else:
        try:
            nodes, edges = source
        except (TypeError, ValueError) as error:
            raise FormatError(
                f"A network is a `Network`, an object with `nodes` and `edges`, or a "
                f"`(nodes, edges)` pair, got {type(source).__name__}."
            ) from error

    node_array = np.asarray(nodes, dtype=np.float64)
    if node_array.ndim != 2 or node_array.shape[1] != 3:
        raise FormatError(
            f"Nodes come as an (n, 3) array of voxel coordinates, got {node_array.shape}."
        )

    # **Never reshaped into place.** A `reshape(-1, 2)` here would turn an (m, 3) triangle array
    # into an (m * 3 / 2, 2) segment list without a murmur -- which is precisely the silent
    # failure `encoding.edges` exists to prevent, committed by the gate that is supposed to
    # prevent it. An empty array is the one case with no shape to check, and a flat 1-D array is
    # accepted because it has an unambiguous reading; everything else must already be (m, 2).
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.zeros((0, 2), dtype=np.int64)
    elif edge_array.ndim == 1:
        if edge_array.size % 2:
            raise FormatError(
                f"A flat edge list holds two indices per edge, so its length is even, got "
                f"{edge_array.size}."
            )
        edge_array = edge_array.reshape(-1, 2)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise FormatError(
            f"Edges come as an (m, 2) array of node indices -- a segment list, not a strip and "
            f"not triangles, got {edge_array.shape}. An (m, 3) array is a triangle list, which "
            f"belongs in a mesh format; konnektion will not reinterpret it as pairs, because "
            f"doing so silently produces a plausible, wrong graph."
        )
    if edge_array.size and (
        edge_array.max() >= len(node_array) or edge_array.min() < 0
    ):
        raise FormatError(
            f"An edge names node {int(edge_array.max())} of a network with {len(node_array)} "
            f"nodes. A dangling edge index is not caught anywhere downstream -- it is in range "
            f"for some cell and draws a segment to the wrong place -- so it is refused here."
        )

    radius_array = None
    if radii is not None:
        radius_array = np.asarray(radii, dtype=np.float64).reshape(-1)
        if len(radius_array) != len(node_array):
            raise FormatError(
                f"Radii are one per node, got {len(radius_array)} for {len(node_array)} nodes."
            )
        if radius_array.size and radius_array.min() < 0.0:
            raise FormatError(
                f"A radius is a length and cannot be negative, got {radius_array.min():g}."
            )

    id_array = None
    if node_ids is not None:
        id_array = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if len(id_array) != len(node_array):
            raise FormatError(
                f"Node ids are one per node, got {len(id_array)} for {len(node_array)} nodes."
            )
        if len(np.unique(id_array)) != len(id_array):
            raise FormatError(
                "Node ids identify a node across cells and levels, so they are unique within an "
                "object. This one repeats an id, which would make a ghost ambiguous."
            )

    if root is not None:
        root = int(root)
        if not 0 <= root < len(node_array):
            raise FormatError(
                f"`root` is an index into this object's nodes, got {root} for "
                f"{len(node_array)} nodes."
            )

    attribute_arrays = None
    if attributes is not None:
        attribute_arrays = _coerce_attributes(attributes, len(node_array))

    return Network(
        nodes=node_array,
        edges=edge_array,
        radii=radius_array,
        node_ids=id_array,
        root=root,
        attributes=attribute_arrays,
    )


def _coerce_attributes(
    attributes: Any, node_count: int  # noqa: ANN401
) -> dict[str, npt.NDArray[np.float64]]:
    """Validate caller-supplied per-node attributes, naming the one that is wrong.

    ``NaN`` is legal -- it is the format's "this node has no answer" -- but ``inf`` is not: an
    infinity survives the float32 round trip, and downstream it silently swallows every window
    and range a value is compared against.
    """
    if not isinstance(attributes, Mapping):
        raise FormatError(
            f"`attributes` is a mapping of name to one value per node, got "
            f"{type(attributes).__name__}."
        )
    coerced: dict[str, npt.NDArray[np.float64]] = {}
    for raw_name, raw_values in attributes.items():
        name = str(raw_name)
        if not ATTRIBUTE_NAME_PATTERN.match(name):
            raise FormatError(
                f"Attribute {name!r}: a name is lowercase letters, digits and underscores, "
                f"starting with a letter or underscore, at most 64 characters -- it becomes a "
                f"Parquet column and a picker value."
            )
        if name == "radius":
            raise FormatError(
                "Attribute 'radius': a per-node radius travels in the format's own radii "
                "encoding, so pass it as `radii`, not as an attribute of that name."
            )
        try:
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise FormatError(
                f"Attribute {name!r} does not coerce to floats: {error}"
            ) from error
        if len(values) != node_count:
            raise FormatError(
                f"Attribute {name!r} is one value per node, got {len(values)} for "
                f"{node_count} nodes."
            )
        if values.size and np.isinf(values).any():
            raise FormatError(
                f"Attribute {name!r} holds an infinity. `NaN` says a node has no value; an "
                f"infinity is unanswerable to every window a value is compared against, so it "
                f"is refused."
            )
        coerced[name] = values
    return coerced


def coerce_objects(objects: Mapping[int, NetworkSource]) -> dict[int, Network]:
    """Coerce a whole ``{object_id: network}`` mapping, naming the object that failed."""
    coerced: dict[int, Network] = {}
    for object_id, source in objects.items():
        try:
            coerced[int(object_id)] = coerce_network(source)
        except FormatError as error:
            raise FormatError(f"Object {object_id!r}: {error}") from error
    return coerced


__all__ = ["HasNodesAndEdges", "Network", "NetworkSource", "coerce_network", "coerce_objects"]
