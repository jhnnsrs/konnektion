"""Turning ``{object_id: network}`` into an octree of levels, ready to write.

The order of operations is the whole design, and it is not the obvious one:

1. **Coarsen first, globally, per object.** Each level's graph is derived from the level below it
   by pruning and then straightening the *whole* object, before any cell is considered.
2. **Partition second.** The coarsened graph is assigned to cells, and edges that cross a cell
   plane pick up a ghost copy of their foreign endpoint.

Doing it the other way round -- partition once, coarsen each cell -- is what a mesh format does,
and for a graph it is wrong. Strahler order is a property of the whole tree: a twig's order says
how much hangs below it, which a cell holding only that twig cannot know. Coarsening per cell
would prune by a different threshold on each side of a plane and leave the two disagreeing about
which nodes exist.

Coarsening globally buys the format its central guarantee for free: because the decision is made
once per object per level, **every cell of a level agrees about which nodes survive**, so a ghost
is always a copy of a node that really is there. That is what :func:`konnektion.verify.verify`
checks at the ``topology`` tier, and what makes each level independently correct.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from konnektion import geometry
from konnektion.codecs.blobs import (
    encode_attribute_values,
    encode_edges,
    encode_ghost_cells,
    encode_ghost_positions,
    encode_node_ids,
    encode_positions,
    encode_radii,
)
from konnektion.errors import FormatError
from konnektion.frames import (
    arrow_schemas,
    attribute_column,
    build_table,
    ghost_attribute_column,
)
from konnektion.manifest import (
    ATTRIBUTE_COMPONENT,
    ATTRIBUTE_DEGREE,
    ATTRIBUTE_DEPTH,
    ATTRIBUTE_FLOAT32,
    ATTRIBUTE_STRAHLER,
    CODEC_NONE,
    COMPRESSION_NONE,
    MAX_ORDINAL,
    NODE_IDS_UINT64,
    RADII_FLOAT32,
    RADII_NONE,
    Attribute,
    Coarsening,
    Encoding,
    Grid,
    Manifest,
)
from konnektion.octree import cell_box, cell_of, morton_decode
from konnektion.sources import Network, NetworkSource, coerce_objects

#: How many bytes of geometry the coarsest level should fit inside before the ladder stops.
#:
#: A level small enough to arrive in one request is small enough: past that point another level
#: saves a fraction of an already-trivial fetch and costs a whole set of claims a verifier has to
#: check. 256 KiB is half a row group, so the top of a finished ladder is one read.
OVERVIEW_TARGET_BYTES = 256 * 1024

#: The most levels the adaptive ladder will build, however large the input.
MAX_LEVELS = 12

#: The per-node metrics every build computes and declares, in declaration order. All four are
#: O(nodes + edges) on adjacency the build walks anyway, so an opt-in flag would only create
#: collections that lack the vocabulary for no saving. ``strahler`` and ``depth`` are ``NaN``
#: throughout an object with no distinguished root -- both are statements relative to a root,
#: and electing one arbitrarily would bake the caller's node numbering into the data.
INTRINSIC_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("strahler", ATTRIBUTE_STRAHLER),
    ("degree", ATTRIBUTE_DEGREE),
    ("depth", ATTRIBUTE_DEPTH),
    ("component", ATTRIBUTE_COMPONENT),
)


@dataclass
class NetworkCollection:
    """A built collection, in memory: three frames and the manifest that describes them.

    Nothing has been written yet. :meth:`write` serializes it into a store, and that is also
    what the mikro upload path calls -- so a caller can inspect or verify a collection before
    spending an upload on it.
    """

    grid: Grid
    encoding: Encoding
    cell_catalog: Any
    object_catalog: Any
    shards: list[tuple[int, Any]]
    axes: list[str] | None = None
    shape: tuple[int, int, int] | None = None
    attributes: list[Attribute] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def manifest(self) -> Manifest:
        """The manifest for this collection, before a writer fills in ``files``."""
        return Manifest(
            grid=self.grid,
            encoding=self.encoding,
            axes=self.axes,
            shape=self.shape,
            attributes=tuple(self.attributes),
            counts={
                "objects": int(self.object_catalog.num_rows),
                "cells": int(self.cell_catalog.num_rows),
                "levels": int(self.grid.levels),
            },
        )

    def write(self, store: Any, prefix: str = "") -> Manifest:  # noqa: ANN401
        """Write the whole tree into ``store``, landing the manifest last."""
        from konnektion.writer import write_collection

        return write_collection(self, store, prefix)


def choose_cell_size(
    objects: Mapping[int, Network], *, levels: int = 3
) -> tuple[int, int, int]:
    """A level-0 cell that the objects mostly fit inside, rounded to a power of two.

    **Pass the source array's chunk shape instead when you know it.** That is the value worth
    matching -- it makes a cell fetch align with a chunk fetch for whatever the graph was traced
    out of -- and nothing about a set of node positions can reveal it.

    Failing that: twice the 90th-percentile object extent per axis. Cells much smaller than the
    features in them scatter one object across many cells, which costs a ghost at every crossing;
    cells much larger defeat the point of the octree.
    """
    if not objects:
        return (64, 64, 64)
    extents = np.array(
        [network.bounds[1] - network.bounds[0] for network in objects.values()], dtype=np.float64
    )
    target = 2.0 * np.percentile(extents, 90, axis=0)
    chosen: list[int] = []
    for axis in range(3):
        size = max(16.0, float(target[axis]))
        power = int(np.ceil(np.log2(size)))
        chosen.append(int(2 ** min(power, 16)))
    del levels
    return (chosen[0], chosen[1], chosen[2])


def _with_attributes(
    objects: dict[int, Network],
) -> tuple[dict[int, Network], list[Attribute]]:
    """Compute the intrinsic metrics and merge the caller's attributes, once, on the full graphs.

    Every declared name is present on every object -- a caller attribute one object lacks is
    ``NaN``-filled rather than refused, because a partial measurement is a normal state of real
    data and a per-object column set would make the manifest a lie for somebody. What *is*
    refused is a caller attribute wearing an intrinsic name: the manifest's ``semantics`` would
    then claim konnektion computed values it never saw.
    """
    intrinsic_names = {name for name, _ in INTRINSIC_ATTRIBUTES}
    user_names = sorted(
        {name for network in objects.values() for name in (network.attributes or {})}
    )
    collisions = sorted(intrinsic_names & set(user_names))
    if collisions:
        raise FormatError(
            f"Attribute(s) {', '.join(collisions)} collide with the metrics konnektion computes "
            f"itself ({', '.join(sorted(intrinsic_names))}). Rename yours -- the manifest's "
            f"semantics field would otherwise claim these values were computed here."
        )

    declared = [
        *(Attribute(name=name, semantics=semantics) for name, semantics in INTRINSIC_ATTRIBUTES),
        *(Attribute(name=name) for name in user_names),
    ]

    enriched: dict[int, Network] = {}
    for object_id, network in objects.items():
        count = network.node_count
        rooted = network.root is not None
        computed: dict[str, npt.NDArray[np.float64]] = {
            "strahler": (
                geometry.strahler_orders(network).astype(np.float64)
                if rooted
                else np.full(count, np.nan)
            ),
            "degree": geometry.degrees(network).astype(np.float64),
            "depth": geometry.depth_from_root(network),
            "component": geometry.component_labels(network).astype(np.float64),
        }
        supplied = network.attributes or {}
        for name in user_names:
            computed[name] = supplied.get(name, np.full(count, np.nan))
        enriched[object_id] = replace(network, attributes=computed)
    return enriched, declared


def _coarsened_levels(
    network: Network, levels: int, coarsening: Coarsening
) -> list[tuple[Network, float]]:
    """One graph per level, each a subset of the one below it, with its deviation bound.

    Built **cumulatively** -- level ``L`` is coarsened from level ``L-1`` rather than from the
    original -- which is what makes ``coarse is a subset of fine`` true by construction rather
    than by argument. Strahler monotonicity falls out of the same fact.

    Level 0 is the input, untouched. A collection always carries the data it was given.
    """
    built: list[tuple[Network, float]] = [(network, 0.0)]
    current = network
    for level in range(1, levels):
        previous = current
        pruned, kept_prune = geometry.prune_to_order(
            current, coarsening.strahler_threshold(level), floor_nodes=coarsening.floor_nodes
        )
        simplified, kept_simplify = geometry.simplify(pruned, coarsening.epsilon_at(level))
        # Compose the two index maps so the deviation is measured against the level below.
        kept = kept_prune[kept_simplify] if len(kept_prune) else kept_simplify
        error = geometry.polyline_deviation(previous, simplified, kept)
        built.append((simplified, error))
        current = simplified
    return built


def _partition(
    network: Network, level: int, cell_size: Sequence[int]
) -> dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Assign one coarsened graph to cells, resolving ghosts.

    Returns, per cell: the object-local indices of the nodes it **owns**, the object-local
    indices of the **ghosts** it needs, the owning cell of each of those ghosts, and the edges it
    owns as indices into the concatenation ``owned + ghosts``.

    **An edge belongs to exactly one cell** -- the one with the lower Morton code among its two
    endpoints' cells -- and that cell holds a ghost copy of the endpoint it does not own. The
    alternative, putting a crossing edge in both cells, also keeps every cell self-contained but
    draws the segment twice; under any blending but opaque that is visible as a brighter line
    exactly along cell planes. One owner, one draw, and a cell fetched on its own still shows its
    edges reaching out to their real endpoints.
    """
    if not network.node_count:
        return {}

    cells = cell_of(network.nodes, level, cell_size)
    owned: dict[int, list[int]] = {}
    for node, cell in enumerate(cells.tolist()):
        owned.setdefault(cell, []).append(node)

    edge_owner: dict[int, list[int]] = {}
    needed: dict[int, set[int]] = {}
    for index, (a, b) in enumerate(network.edges.tolist()):
        cell_a, cell_b = int(cells[a]), int(cells[b])
        owner = min(cell_a, cell_b)
        edge_owner.setdefault(owner, []).append(index)
        if cell_a != cell_b:
            foreign = b if cell_a == owner else a
            needed.setdefault(owner, set()).add(foreign)

    partition: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]] = {}
    for cell in sorted(set(owned) | set(edge_owner)):
        residents = list(owned.get(cell, []))
        ghosts = sorted(needed.get(cell, set()) - set(residents))
        local = {node: position for position, node in enumerate(residents + ghosts)}
        edges = np.array(
            [[local[a], local[b]] for a, b in network.edges[edge_owner.get(cell, [])].tolist()],
            dtype=np.int64,
        ).reshape(-1, 2)
        partition[cell] = (
            np.asarray(residents, dtype=np.int64),
            np.asarray(ghosts, dtype=np.int64),
            cells[np.asarray(ghosts, dtype=np.int64)] if ghosts else np.zeros(0, dtype=np.int64),
            edges,
        )
    return partition


def _child_mask(cell: int, level: int, present: Mapping[int, set[int]]) -> int:
    """Which of a cell's eight children exist at the level below, as a bitmask."""
    if level == 0:
        return 0
    i, j, k = morton_decode(cell)
    from konnektion.octree import morton_encode_one

    mask = 0
    below = present.get(level - 1, set())
    for bit, (di, dj, dk) in enumerate(
        [(a, b, c) for c in (0, 1) for b in (0, 1) for a in (0, 1)]
    ):
        if morton_encode_one((2 * i + di, 2 * j + dj, 2 * k + dk)) in below:
            mask |= 1 << bit
    return mask


def build_collection(
    objects: Mapping[int, NetworkSource],
    *,
    cell_size: Sequence[int] | None = None,
    levels: int | None = None,
    axes: Sequence[str] | None = ("x", "y", "z"),
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    coarsening: Coarsening | None = None,
    radii: str | None = None,
) -> NetworkCollection:
    """Build a network collection from ``{object_id: network}``.

    ``levels`` left unset is **chosen from the data**: the ladder grows only while the coarsest
    level is still bigger than :data:`OVERVIEW_TARGET_BYTES`, and a small collection therefore
    gets ``levels=1``. That is not a degenerate case to be tolerated but the expected one for a
    traced arbor -- at one level nothing is pruned, nothing is straightened, every node is
    exactly where the tracer put it, and the manifest says so with ``pruning: NONE`` and
    ``simplification: NONE``. A format that forced a ladder would spend build time producing
    levels that are claims somebody then has to verify.

    ``radii`` left unset carries a radius exactly when the objects have one, as ``FLOAT32``.
    Pass ``RADII_NONE`` to drop it, or the quantized encoding to trade precision for bytes.

    Every build also computes and stores the :data:`INTRINSIC_ATTRIBUTES` (Strahler order,
    degree, depth from root, component), on the full level-0 graph, beside whatever per-node
    ``attributes`` the objects carry themselves. Coarser levels subset those values and never
    recompute them, so a node's metric is the same number at every level it survives to.
    """
    coerced = coerce_objects(objects)
    if len(coerced) > MAX_ORDINAL:
        raise FormatError(
            f"A collection holds at most {MAX_ORDINAL} objects, got {len(coerced)}."
        )
    for object_id, network in coerced.items():
        if network.nodes.size and network.nodes.min() < 0.0:
            raise FormatError(
                f"Object {object_id} has a node at {network.nodes.min():g}: the octree addresses "
                f"the positive octant only, so shift the graph before building."
            )

    coerced, declared_attributes = _with_attributes(coerced)
    attribute_names = [attribute.name for attribute in declared_attributes]
    schemas = arrow_schemas(attribute_names=attribute_names)
    grid_cell_size = tuple(int(c) for c in (cell_size or choose_cell_size(coerced)))

    with_radii = any(network.radii is not None for network in coerced.values())
    radii_declaration = radii if radii is not None else (RADII_FLOAT32 if with_radii else RADII_NONE)
    if radii_declaration != RADII_NONE and not with_radii:
        raise FormatError(
            f"`radii` is {radii_declaration!r} but no object carries one. A declared radius "
            f"column that holds nothing is a place for a reader to find zeros and believe them."
        )

    schedule = coarsening or Coarsening()
    depth = int(levels) if levels is not None else 0
    if depth < 0:
        raise FormatError(f"`levels` is at least 1, got {levels}.")

    notes: list[str] = []
    if depth == 0:
        depth, note = _choose_depth(coerced, grid_cell_size, schedule)
        notes.append(note)
    if depth == 1:
        schedule = Coarsening.none()
        notes.append(
            "One level, so nothing is pruned or straightened and the manifest declares "
            "pruning: NONE / simplification: NONE."
        )

    per_object = {
        object_id: _coarsened_levels(network, depth, schedule)
        for object_id, network in coerced.items()
    }
    ordinals = {object_id: index for index, object_id in enumerate(sorted(coerced))}

    encoding = Encoding(
        node_ids=NODE_IDS_UINT64,
        radii=radii_declaration,
        codec=codec,
        compression=compression,
        pruning=schedule.pruning,
        simplification=schedule.simplification,
    )

    cell_rows: list[dict[str, Any]] = []
    shard_rows: dict[int, list[dict[str, Any]]] = {level: [] for level in range(depth)}
    object_cells: dict[int, list[tuple[int, int]]] = {object_id: [] for object_id in coerced}
    object_counts: dict[int, tuple[int, int]] = {}
    present: dict[int, set[int]] = {level: set() for level in range(depth)}

    for level in range(depth):
        # cell -> per-object contributions, so one row can hold several objects.
        buckets: dict[int, list[tuple[int, Network, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]]] = {}
        errors: dict[int, float] = {}
        for object_id, built in per_object.items():
            network, error = built[level]
            for cell, (residents, ghosts, owners, edges) in _partition(
                network, level, grid_cell_size
            ).items():
                buckets.setdefault(cell, []).append(
                    (object_id, network, residents, ghosts, owners, edges)
                )
                errors[cell] = max(errors.get(cell, 0.0), error)
                object_cells[object_id].append((level, cell))
                present[level].add(cell)
            if level == 0:
                object_counts[object_id] = (network.node_count, network.edge_count)

        for cell in sorted(buckets):
            row, catalog = _emit_cell(
                cell,
                level,
                buckets[cell],
                grid_cell_size,
                encoding,
                errors.get(cell, 0.0),
                ordinals,
                attribute_names,
            )
            shard_rows[level].append(row)
            cell_rows.append(catalog)

    for row in cell_rows:
        row["child_mask"] = _child_mask(int(row["cell"]), int(row["level"]), present)

    object_rows = _object_catalog(coerced, object_cells, ordinals, object_counts)

    shape = _shape_of(coerced)
    return NetworkCollection(
        grid=Grid(cell_size=grid_cell_size, levels=depth),  # type: ignore[arg-type]
        encoding=encoding,
        cell_catalog=build_table(cell_rows, schemas["cell_catalog"]),
        object_catalog=build_table(object_rows, schemas["object_catalog"]),
        shards=[(level, build_table(shard_rows[level], schemas["geometry"])) for level in range(depth)],
        axes=None if axes is None else [str(name) for name in axes],
        shape=shape,
        attributes=declared_attributes,
        notes=notes,
    )


def _choose_depth(
    objects: Mapping[int, Network], cell_size: Sequence[int], schedule: Coarsening
) -> tuple[int, str]:
    """How many levels this data earns, and a sentence saying why.

    Grow the ladder only while the coarsest level is still too big to arrive in one request.
    """
    total_nodes = sum(network.node_count for network in objects.values())
    # 6 bytes of position, 8 of node id, and roughly one edge per node at 8 bytes.
    approximate = total_nodes * 22
    if approximate <= OVERVIEW_TARGET_BYTES:
        return 1, (
            f"{total_nodes} nodes is about {approximate / 1024:.0f} KiB, already under the "
            f"{OVERVIEW_TARGET_BYTES / 1024:.0f} KiB a level should fit in, so one level."
        )
    depth = 1
    remaining = approximate
    current: dict[int, Network] = dict(objects)
    while remaining > OVERVIEW_TARGET_BYTES and depth < MAX_LEVELS:
        coarser: dict[int, Network] = {}
        for object_id, network in current.items():
            pruned, _ = geometry.prune_to_order(
                network, schedule.strahler_threshold(depth), floor_nodes=schedule.floor_nodes
            )
            simplified, _ = geometry.simplify(pruned, schedule.epsilon_at(depth))
            coarser[object_id] = simplified
        nodes = sum(network.node_count for network in coarser.values())
        if nodes >= sum(network.node_count for network in current.values()):
            break  # coarsening has stopped paying; another level would be all cost
        current = coarser
        remaining = nodes * 22
        depth += 1
    del cell_size
    return depth, (
        f"{total_nodes} nodes is about {approximate / 1024:.0f} KiB, so the ladder grew to "
        f"{depth} level(s) until the top fit {OVERVIEW_TARGET_BYTES / 1024:.0f} KiB."
    )


def _emit_cell(
    cell: int,
    level: int,
    contributions: Sequence[tuple[int, Network, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    cell_size: Sequence[int],
    encoding: Encoding,
    error: float,
    ordinals: Mapping[int, int],
    attribute_names: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pack every object's share of one cell into a geometry row and its catalog entry.

    Owned nodes and ghosts are concatenated **separately** and then joined, so a cell holding two
    objects lays out as ``[owned of A, owned of B, ghosts of A, ghosts of B]``. That is why the
    edge indices are rebased twice below: once by the owned cursor, and again -- for indices that
    landed in an object's ghost range -- by where that object's ghosts ended up in the tail.
    """
    owned_positions: list[npt.NDArray[np.float64]] = []
    owned_ids: list[npt.NDArray[np.int64]] = []
    owned_radii: list[npt.NDArray[np.float64]] = []
    ghost_positions: list[npt.NDArray[np.float64]] = []
    ghost_ids: list[npt.NDArray[np.int64]] = []
    ghost_owners: list[npt.NDArray[np.int64]] = []
    ghost_radii: list[npt.NDArray[np.float64]] = []
    owned_attribute_values: dict[str, list[npt.NDArray[np.float64]]] = {
        name: [] for name in attribute_names
    }
    ghost_attribute_values: dict[str, list[npt.NDArray[np.float64]]] = {
        name: [] for name in attribute_names
    }
    edges: list[npt.NDArray[np.int64]] = []
    object_ids: list[int] = []
    object_ordinals: list[int] = []
    node_offsets: list[int] = []
    ghost_offsets: list[int] = []
    edge_offsets: list[int] = []

    ordered = sorted(contributions, key=lambda c: c[0])
    owned_total = sum(len(item[2]) for item in ordered)

    owned_cursor = 0
    ghost_cursor = 0
    edge_cursor = 0
    for object_id, network, residents, ghosts, owners, local_edges in ordered:
        owned_positions.append(network.nodes[residents])
        owned_ids.append(network.ids()[residents])
        ghost_positions.append(network.nodes[ghosts])
        ghost_ids.append(network.ids()[ghosts])
        ghost_owners.append(owners)
        if encoding.has_radii:
            source = network.radii
            owned_radii.append(
                np.zeros(len(residents)) if source is None else source[residents]
            )
            ghost_radii.append(np.zeros(len(ghosts)) if source is None else source[ghosts])
        for name in attribute_names:
            values = (network.attributes or {})[name]
            owned_attribute_values[name].append(values[residents])
            ghost_attribute_values[name].append(values[ghosts])

        # An index below len(residents) is an owned node; at or above it, a ghost. The two go to
        # different places in the row, so they are rebased differently.
        rebased = local_edges.copy()
        if rebased.size:
            is_ghost = rebased >= len(residents)
            rebased = np.where(
                is_ghost,
                owned_total + ghost_cursor + (rebased - len(residents)),
                owned_cursor + rebased,
            )
        edges.append(rebased.reshape(-1, 2))

        object_ids.append(int(object_id))
        object_ordinals.append(int(ordinals[object_id]))
        node_offsets.append(owned_cursor)
        ghost_offsets.append(ghost_cursor)
        edge_offsets.append(edge_cursor)
        owned_cursor += len(residents)
        ghost_cursor += len(ghosts)
        edge_cursor += len(local_edges)

    def stack(parts: list[npt.NDArray[np.float64]], width: int) -> npt.NDArray[np.float64]:
        return np.concatenate(parts) if parts else np.zeros((0, width) if width else 0)

    all_owned = stack(owned_positions, 3)
    all_ghosts = stack(ghost_positions, 3)
    all_owned_ids = np.concatenate(owned_ids) if owned_ids else np.zeros(0, dtype=np.int64)
    all_ghost_ids = np.concatenate(ghost_ids) if ghost_ids else np.zeros(0, dtype=np.int64)
    all_owners = np.concatenate(ghost_owners) if ghost_owners else np.zeros(0, dtype=np.int64)
    all_edges = np.concatenate(edges) if edges else np.zeros((0, 2), dtype=np.int64)

    position_blob = encode_positions(
        all_owned, cell=cell, level=level, cell_size=cell_size,
        codec=encoding.codec, compression=encoding.compression,
    )
    id_blob = encode_node_ids(
        all_owned_ids, declaration=encoding.node_ids,
        codec=encoding.codec, compression=encoding.compression,
    )
    edge_blob = encode_edges(all_edges, codec=encoding.codec, compression=encoding.compression)
    ghost_position_blob = encode_ghost_positions(
        all_ghosts, all_owners, level=level, cell_size=cell_size,
        codec=encoding.codec, compression=encoding.compression,
    )
    ghost_cell_blob = encode_ghost_cells(
        all_owners, codec=encoding.codec, compression=encoding.compression
    )
    ghost_id_blob = encode_node_ids(
        all_ghost_ids, declaration=encoding.node_ids,
        codec=encoding.codec, compression=encoding.compression,
    )

    radius_blob = ghost_radius_blob = None
    if encoding.has_radii:
        radius_blob = encode_radii(
            np.concatenate(owned_radii) if owned_radii else np.zeros(0),
            declaration=encoding.radii, cell=cell, level=level, cell_size=cell_size,
            codec=encoding.codec, compression=encoding.compression,
        )
        # A ghost's radius belongs to its owning cell, so it is quantized there too. With
        # FLOAT32 this is moot; with the quantized encoding it is the same trap as the position.
        ghost_radius_blob = _encode_ghost_radii(
            np.concatenate(ghost_radii) if ghost_radii else np.zeros(0),
            all_owners, level, cell_size, encoding,
        )

    # Ghost values ride in their own blob like ghost radii do, but need no owner-cell dance:
    # a float32 is not quantized against any box.
    attribute_blobs: dict[str, bytes] = {}
    for name in attribute_names:
        attribute_blobs[attribute_column(name)] = encode_attribute_values(
            np.concatenate(owned_attribute_values[name]) if owned_attribute_values[name] else np.zeros(0),
            declaration=ATTRIBUTE_FLOAT32,
            codec=encoding.codec,
            compression=encoding.compression,
        )
        attribute_blobs[ghost_attribute_column(name)] = encode_attribute_values(
            np.concatenate(ghost_attribute_values[name]) if ghost_attribute_values[name] else np.zeros(0),
            declaration=ATTRIBUTE_FLOAT32,
            codec=encoding.codec,
            compression=encoding.compression,
        )

    blobs = [position_blob, id_blob, edge_blob, ghost_position_blob, ghost_cell_blob, ghost_id_blob]
    blobs += [blob for blob in (radius_blob, ghost_radius_blob) if blob is not None]
    blobs += list(attribute_blobs.values())

    row = {
        "level": level,
        "cell": cell,
        "positions": position_blob,
        "node_ids": id_blob,
        "edges": edge_blob,
        "ghost_positions": ghost_position_blob,
        "ghost_cells": ghost_cell_blob,
        "ghost_ids": ghost_id_blob,
        "radii": radius_blob,
        "ghost_radii": ghost_radius_blob,
        "node_count": len(all_owned),
        "edge_count": len(all_edges),
        "ghost_count": len(all_ghosts),
        "object_ids": object_ids,
        "object_ordinals": object_ordinals,
        "object_node_offsets": node_offsets,
        "object_ghost_offsets": ghost_offsets,
        "object_edge_offsets": edge_offsets,
        **attribute_blobs,
    }

    # The box covers the ghosts too: it is what a viewer culls against, and an edge reaching into
    # a neighbour is drawn from this cell, so a box that stopped at the cell face would cull away
    # geometry this row is responsible for.
    drawn = np.concatenate([all_owned, all_ghosts]) if len(all_ghosts) else all_owned
    low = drawn.min(axis=0) if len(drawn) else np.zeros(3)
    high = drawn.max(axis=0) if len(drawn) else np.zeros(3)
    # Grown by one quantization step, because the box is what a viewer culls against and what it
    # will be culling is the *decoded* geometry, not these floats. Quantization can move a node
    # by up to half a step, so a box measured before it can exclude the very node it describes --
    # and a cell culled away by its own bounding box is a hole in the drawing with no error
    # anywhere. The step is the same at both ends: a ghost is quantized against a cell of the
    # same level, so it has the same extent.
    _, cell_extent = cell_box(cell, level, cell_size)
    step = float(np.max(cell_extent)) / 65535.0
    low = low - step
    high = high + step
    catalog = {
        "level": level,
        "cell": cell,
        "node_count": len(all_owned),
        "edge_count": len(all_edges),
        "ghost_count": len(all_ghosts),
        **{f"bbox_min_{axis}": float(low[index]) for index, axis in enumerate("xyz")},
        **{f"bbox_max_{axis}": float(high[index]) for index, axis in enumerate("xyz")},
        "lod_error": float(error),
        "object_count": len(object_ids),
        "child_mask": 0,
        "part": None,
        "row_group": None,
        "blob_bytes": sum(len(blob) for blob in blobs),
    }
    return row, catalog


def _encode_ghost_radii(
    radii: npt.NDArray[np.float64],
    owners: npt.NDArray[np.int64],
    level: int,
    cell_size: Sequence[int],
    encoding: Encoding,
) -> bytes:
    """Encode ghost radii against each ghost's owning cell, one cell at a time."""
    if encoding.radii == RADII_FLOAT32 or not len(radii):
        return encode_radii(
            radii, declaration=encoding.radii, cell=0, level=level, cell_size=cell_size,
            codec=encoding.codec, compression=encoding.compression,
        )
    pieces = [
        encode_radii(
            radii[index : index + 1], declaration=encoding.radii, cell=int(owners[index]),
            level=level, cell_size=cell_size, codec=encoding.codec,
            compression=COMPRESSION_NONE,
        )
        for index in range(len(radii))
    ]
    from konnektion.codecs.compression import compress

    return compress(b"".join(pieces), encoding.compression)


def _object_catalog(
    objects: Mapping[int, Network],
    object_cells: Mapping[int, Sequence[tuple[int, int]]],
    ordinals: Mapping[int, int],
    counts: Mapping[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    """One row per object: where it is, how big it is, and which cells hold it."""
    rows: list[dict[str, Any]] = []
    for object_id in sorted(objects):
        network = objects[object_id]
        low, high = network.bounds
        nodes, edges = counts.get(object_id, (network.node_count, network.edge_count))
        root_id = None
        if network.root is not None:
            root_id = int(network.ids()[network.root])
        rows.append({
            "object_id": int(object_id),
            "ordinal": int(ordinals[object_id]),
            "root_node_id": root_id,
            "component_count": _component_count(network),
            **{f"bbox_min_{axis}": float(low[index]) for index, axis in enumerate("xyz")},
            **{f"bbox_max_{axis}": float(high[index]) for index, axis in enumerate("xyz")},
            "node_count": int(nodes),
            "edge_count": int(edges),
            "cells": [int(cell) for _, cell in sorted(set(object_cells[object_id]))],
        })
    return rows


def _component_count(network: Network) -> int:
    """How many connected pieces an object is in, counted once at build time.

    Written into the object catalog so the connectivity claim can be *checked* rather than only
    compared between levels. A disconnected input is legitimate -- two unlinked vessel segments
    are two components and always were -- so the format records the number rather than insisting
    on one, and the verifier then catches a level that has more.
    """
    _, roots = geometry.spanning_forest(network)
    return len(roots)


def _shape_of(objects: Mapping[int, Network]) -> tuple[int, int, int] | None:
    """The smallest whole-voxel box holding every object, or None for an empty collection."""
    if not objects:
        return None
    highs = np.stack([network.bounds[1] for network in objects.values()])
    return tuple(int(np.ceil(value)) for value in highs.max(axis=0))  # type: ignore[return-value]


__all__ = [
    "INTRINSIC_ATTRIBUTES",
    "MAX_LEVELS",
    "OVERVIEW_TARGET_BYTES",
    "NetworkCollection",
    "build_collection",
    "choose_cell_size",
]
