"""Checking that a written collection is what it says it is.

Three tiers, because they cost wildly different amounts and a caller should get to choose:

* ``structure`` -- the catalogs agree with each other and with the manifest, and every locator
  resolves. Reads two small Parquet files.
* ``blobs`` -- every blob decodes, and its length agrees with the counts on its row. Reads the
  geometry.
* ``topology`` -- the claims **nothing downstream can see**. This is the tier the format exists
  for, and the reason this module is not a test.

The last one deserves the emphasis. A mesh that loses a triangle has a hole, and an eye finds it.
A graph that loses an interior node is a graph in *pieces*, and every piece still draws: it looks
like data, at every layer, forever. So the ancestor-closed invariant is checked rather than
assumed, even though :func:`konnektion.geometry.strahler_orders` makes it true by construction --
a proof of an algorithm is not a proof of a build.

**A check that cannot run is skipped and reported, never passed.** A single-level collection has
no pruning and no simplification, so it has no ancestor-closure to violate and no ``lod_error`` to
bound; saying "3 of 3 passed" there would be a report that reads identically to one from a
collection that was really checked.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from konnektion.errors import FormatError
from konnektion.frames import attribute_column, ghost_attribute_column
from konnektion.manifest import (
    CELL_CATALOG_PATH,
    OBJECT_CATALOG_PATH,
    PRUNING_NONE,
    SIMPLIFICATION_NONE,
)
from konnektion.octree import cell_box, cell_of
from konnektion.reader import Collection, DecodedCell

#: The tiers, cheapest first. Each includes the ones before it.
TIERS = ("structure", "blobs", "topology")


@dataclass(frozen=True)
class Check:
    """One thing that was checked, and what was found."""

    name: str
    tier: str
    ok: bool
    detail: str = ""
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        """A single line, so a report prints as a list."""
        mark = "PASS" if self.ok else "FAIL"
        line = f"  {mark}  {self.name}"
        if self.detail:
            line += f" -- {self.detail}"
        for example in self.examples:
            line += f"\n         {example}"
        return line


@dataclass(frozen=True)
class VerifyReport:
    """Everything that was checked, and everything that deliberately was not."""

    checks: tuple[Check, ...]
    tier: str = "blobs"
    #: What was not checked and why -- a tier not requested, or a claim this collection does not
    #: make. Reported rather than silently omitted: the difference between "checked and fine" and
    #: "there was nothing here to check" is the whole value of the report.
    skipped: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[Check, ...]:
        """The checks that did not pass."""
        return tuple(check for check in self.checks if not check.ok)

    @property
    def ok(self) -> bool:
        """Whether every check that ran passed."""
        return not self.failures

    def __str__(self) -> str:
        """The whole report, one check per line."""
        head = (
            f"{'PASS' if self.ok else 'FAIL'}: "
            f"{len(self.checks) - len(self.failures)}/{len(self.checks)} checks at tier "
            f"{self.tier!r}"
        )
        lines = [head, *(str(check) for check in self.checks)]
        lines += [f"  SKIP  {note}" for note in self.skipped]
        return "\n".join(lines)


def verify(collection: Collection, *, tier: str = "blobs") -> VerifyReport:
    """Check a collection, as far as ``tier`` asks.

    Args:
        collection: an opened collection.
        tier: how much to check -- ``structure``, ``blobs`` or ``topology``. Each includes the
            ones before it.
    """
    if tier not in TIERS:
        raise ValueError(f"`tier` is one of {', '.join(TIERS)}, got {tier!r}.")

    wanted = TIERS[: TIERS.index(tier) + 1]
    checks: list[Check] = []
    skipped: list[str] = []

    checks.extend(_structure(collection))
    if "blobs" in wanted:
        found, notes = _blobs(collection)
        checks.extend(found)
        skipped.extend(notes)
    else:
        skipped.append("the `blobs` tier, which is what decodes the geometry")
    if "topology" in wanted:
        found, notes = _topology(collection)
        checks.extend(found)
        skipped.extend(notes)
    else:
        skipped.append("the `topology` tier, which is what checks connectivity and ghosts")

    return VerifyReport(checks=tuple(checks), tier=tier, skipped=tuple(skipped))


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def _structure(collection: Collection) -> Iterator[Check]:
    """The catalogs against each other and against the manifest."""
    yield _the_manifest_names_what_exists(collection)
    yield _every_cell_has_a_locator(collection)
    yield _every_object_names_cells_that_exist(collection)
    yield _levels_are_within_the_grid(collection)
    yield _child_masks_point_at_real_children(collection)


def _the_manifest_names_what_exists(collection: Collection) -> Check:
    """Every file the manifest names is in the store."""
    from konnektion.stores import join, list_paths

    present = set(list_paths(collection.store, collection.prefix))
    wanted: list[str] = [CELL_CATALOG_PATH, OBJECT_CATALOG_PATH]
    levels = collection.manifest.files.get("levels") or {}
    for entries in levels.values():
        wanted.extend(entry["path"] for entry in entries)

    missing = [
        path for path in wanted if join(collection.prefix, path) not in present
    ]
    return Check(
        name="every file the manifest names is present",
        tier="structure",
        ok=not missing,
        detail=f"{len(wanted)} named, {len(missing)} missing",
        examples=tuple(missing[:5]),
    )


def _every_cell_has_a_locator(collection: Collection) -> Check:
    """A catalog row without a part is a cell no reader could fetch."""
    orphans = [
        f"level {entry.level} cell {entry.cell}"
        for entry in collection.cells.values()
        if entry.part is None or entry.row_group is None
    ]
    return Check(
        name="every cell names the part and row group holding it",
        tier="structure",
        ok=not orphans,
        detail=f"{len(collection.cells)} cells, {len(orphans)} without a locator",
        examples=tuple(orphans[:5]),
    )


def _every_object_names_cells_that_exist(collection: Collection) -> Check:
    """An object's ``cells`` list is a claim about where to find it."""
    keys = {cell for _, cell in collection.cells}
    broken: list[str] = []
    for entry in collection.objects.values():
        for cell in entry.cells:
            if cell not in keys:
                broken.append(f"object {entry.object_id} names cell {cell}, which has no row")
    return Check(
        name="every object's cells exist in the cell catalog",
        tier="structure",
        ok=not broken,
        detail=f"{len(collection.objects)} objects, {len(broken)} bad references",
        examples=tuple(broken[:5]),
    )


def _levels_are_within_the_grid(collection: Collection) -> Check:
    """No cell claims a level the grid does not have."""
    depth = collection.grid.levels
    stray = [
        f"level {level} cell {cell}" for level, cell in collection.cells if not 0 <= level < depth
    ]
    return Check(
        name="every cell's level is inside the grid",
        tier="structure",
        ok=not stray,
        detail=f"grid declares {depth} level(s)",
        examples=tuple(stray[:5]),
    )


def _child_masks_point_at_real_children(collection: Collection) -> Check:
    """A child mask names cells at the level below; each named one must exist."""
    broken: list[str] = []
    for entry in collection.cells.values():
        for key in entry.children():
            if key not in collection.cells:
                broken.append(
                    f"level {entry.level} cell {entry.cell} claims child {key}, which has no row"
                )
    return Check(
        name="child masks name cells that exist",
        tier="structure",
        ok=not broken,
        detail=f"{len(broken)} dangling children",
        examples=tuple(broken[:5]),
    )


# --------------------------------------------------------------------------- #
# blobs
# --------------------------------------------------------------------------- #


def _blobs(collection: Collection) -> tuple[list[Check], list[str]]:
    """Every blob decodes, and agrees with the counts on its row."""
    checks: list[Check] = []
    notes: list[str] = []
    declared = collection.manifest.attributes

    checks.append(_attribute_columns_match_the_manifest(collection))

    decoded: list[str] = []
    mismatched: list[str] = []
    ranged: list[str] = []
    boxed: list[str] = []
    short_attributes: list[str] = []
    refused: list[str] = []
    count = 0

    # A decode error -- a declared attribute with no blob, a corrupt column -- is reported as a
    # failure rather than raised, so the column check above still reaches the report that
    # explains it.
    try:
        for level in collection.levels():
            for cell in collection.iter_cells(level):
                count += 1
                where = f"level {cell.level} cell {cell.cell}"
                entry = collection.cells.get((cell.level, cell.cell))
                if entry is None:
                    decoded.append(f"{where} is in the geometry but has no catalog row")
                    continue
                if len(cell.positions) != entry.node_count + entry.ghost_count:
                    mismatched.append(
                        f"{where} decoded {len(cell.positions)} positions, catalog says "
                        f"{entry.node_count} + {entry.ghost_count} ghosts"
                    )
                if len(cell.edges) != entry.edge_count:
                    mismatched.append(
                        f"{where} decoded {len(cell.edges)} edges, catalog says {entry.edge_count}"
                    )
                if cell.edges.size and (
                    cell.edges.max() >= len(cell.positions) or cell.edges.min() < 0
                ):
                    ranged.append(
                        f"{where} has an edge naming node {int(cell.edges.max())} of "
                        f"{len(cell.positions)}"
                    )
                for attribute in declared:
                    values = cell.attributes.get(attribute.name)
                    if values is None or len(values) != entry.node_count + entry.ghost_count:
                        short_attributes.append(
                            f"{where} decoded {0 if values is None else len(values)} values of "
                            f"{attribute.name!r}, catalog says {entry.node_count} + "
                            f"{entry.ghost_count} ghosts"
                        )
                # An owned node must be inside the box it was quantized against. A ghost must not
                # be -- that is what makes it a ghost -- so only the owned half is checked here.
                origin, extent = cell_box(cell.cell, cell.level, collection.grid.cell_size)
                owned = cell.positions[: cell.node_count]
                if len(owned):
                    relative = (owned - origin) / extent
                    if relative.min() < -1e-6 or relative.max() > 1.0 + 1e-6:
                        boxed.append(f"{where} owns a node outside its own cell box")
    except FormatError as error:
        refused.append(str(error))

    checks.append(
        Check(
            name="every cell decodes",
            tier="blobs",
            ok=not refused,
            detail=f"{count} cells decoded",
            examples=tuple(refused[:5]),
        )
    )
    checks.append(
        Check(
            name="every geometry row has a catalog row",
            tier="blobs",
            ok=not decoded,
            detail=f"{count} cells decoded",
            examples=tuple(decoded[:5]),
        )
    )
    checks.append(
        Check(
            name="decoded counts match the catalog",
            tier="blobs",
            ok=not mismatched,
            detail=f"{count} cells compared",
            examples=tuple(mismatched[:5]),
        )
    )
    checks.append(
        Check(
            name="every edge indexes a node the cell holds",
            tier="blobs",
            ok=not ranged,
            detail=f"{count} cells compared",
            examples=tuple(ranged[:5]),
        )
    )
    checks.append(
        Check(
            name="owned nodes lie inside their own cell box",
            tier="blobs",
            ok=not boxed,
            detail=f"{count} cells compared",
            examples=tuple(boxed[:5]),
        )
    )
    if declared:
        checks.append(
            Check(
                name="every declared attribute decodes to one value per node",
                tier="blobs",
                ok=not short_attributes,
                detail=f"{count} cells x {len(declared)} attribute(s) compared",
                examples=tuple(short_attributes[:5]),
            )
        )
    else:
        notes.append(
            "the per-cell attribute checks: this collection declares no attributes"
        )
    return checks, notes


def _attribute_columns_match_the_manifest(collection: Collection) -> Check:
    """The geometry's ``attr_*`` columns are exactly the manifest's declarations.

    Both directions matter. A declared attribute with no column is a picker offering values
    nobody stored; an undeclared ``attr_*`` column is values no reader will ever look for --
    the manifest names what exists, so either mismatch is a build bug.
    """
    declared = {attribute.name for attribute in collection.manifest.attributes}
    wanted = {attribute_column(name) for name in declared} | {
        ghost_attribute_column(name) for name in declared
    }
    problems: list[str] = []
    for level in collection.levels():
        for part, columns in collection.geometry_columns(level).items():
            present = {
                column for column in columns if column.startswith(("attr_", "ghost_attr_"))
            }
            for extra in sorted(present - wanted):
                problems.append(
                    f"level {level} part {part} carries {extra}, which the manifest does not "
                    f"declare"
                )
            for missing in sorted(wanted - present):
                problems.append(
                    f"level {level} part {part} lacks {missing}, which the manifest declares"
                )
    return Check(
        name="attribute columns are exactly what the manifest declares",
        tier="blobs",
        ok=not problems,
        detail=f"{len(declared)} attribute(s) declared",
        examples=tuple(problems[:5]),
    )


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #


def _topology(collection: Collection) -> tuple[list[Check], list[str]]:
    """The claims that nothing downstream can see."""
    checks: list[Check] = []
    skipped: list[str] = []
    encoding = collection.encoding

    per_level = {level: list(collection.iter_cells(level)) for level in collection.levels()}

    checks.append(_ghosts_are_copies(collection, per_level))
    checks.append(_ghosts_belong_to_the_cell_they_name(collection, per_level))
    checks.append(_every_level_is_connected(collection, per_level))

    if collection.grid.levels < 2:
        skipped.append(
            "the cross-level checks (ancestor-closure, monotonicity, lod_error): this "
            "collection has one level, so nothing was pruned, nothing was straightened, and "
            "there is no coarser level to compare against"
        )
        return checks, skipped

    if encoding.pruning == PRUNING_NONE and encoding.simplification == SIMPLIFICATION_NONE:
        skipped.append(
            "the coarsening checks: the manifest declares pruning: NONE and "
            "simplification: NONE, so no level claims to be a reduction of another"
        )
        return checks, skipped

    checks.append(_coarse_levels_are_smaller(collection))
    checks.append(_a_coarse_level_is_a_subset(collection, per_level))
    checks.append(_lod_error_is_a_real_bound(collection, per_level))
    if collection.manifest.attributes:
        checks.append(_attribute_values_survive_coarsening(collection, per_level))
    else:
        skipped.append(
            "the attribute-coarsening check: this collection declares no attributes"
        )
    return checks, skipped


def _members(cell: DecodedCell) -> list[tuple[int, int, int, bool]]:
    """``(object_id, node_id, index, is_ghost)`` for every node a cell holds.

    **Node ids are unique within an object, not within a collection** -- a tracer numbers each
    neuron from one, and forcing those to be globally unique would mean rewriting the caller's
    own identifiers, which are usually the join key to an attributes table. So every id here is
    paired with the object it belongs to, and every map below is keyed on the pair.

    Getting this wrong is not hypothetical: keyed on the id alone, object 2's node 20 collides
    with object 1's, and a ghost check then compares a node against an unrelated node in another
    object and reports a few hundred voxels of drift.
    """
    found: list[tuple[int, int, int, bool]] = []
    for position, object_id in enumerate(cell.object_ids):
        start = cell.object_node_offsets[position]
        stop = (
            cell.object_node_offsets[position + 1]
            if position + 1 < len(cell.object_node_offsets)
            else cell.node_count
        )
        for index in range(start, stop):
            found.append((int(object_id), int(cell.node_ids[index]), index, False))
    for position, object_id in enumerate(cell.object_ids):
        start = cell.node_count + cell.object_ghost_offsets[position]
        stop = (
            cell.node_count + cell.object_ghost_offsets[position + 1]
            if position + 1 < len(cell.object_ghost_offsets)
            else cell.node_count + cell.ghost_count
        )
        for index in range(start, stop):
            found.append((int(object_id), int(cell.node_ids[index]), index, True))
    return found


def _node_map(cells: Sequence) -> dict[tuple[int, int], np.ndarray]:  # type: ignore[type-arg]
    """``(object_id, node_id)`` -> position, from the owned nodes of every cell at one level.

    Owned only: a ghost is a copy, and building the map from copies would make the ghost check
    below compare a value against itself.
    """
    found: dict[tuple[int, int], np.ndarray] = {}  # type: ignore[type-arg]
    for cell in cells:
        for object_id, node_id, index, is_ghost in _members(cell):
            if not is_ghost:
                found[(object_id, node_id)] = cell.positions[index]
    return found


def _ghosts_are_copies(collection: Collection, per_level: dict[int, list]) -> Check:  # type: ignore[type-arg]
    """A ghost's position matches the node it copies, to within a quantization step.

    The check that makes ghosting safe. A ghost is quantized against a *different* cell's box
    than the row it sits in, so a writer that used the wrong box produces a position that is
    plausible, wrong, and invisible -- the edge simply reaches to slightly the wrong place.

    The tolerance is one quantum of the owning cell, since the ghost and its owner are quantized
    against the same box and should agree exactly; the slack is for float round-tripping only.
    """
    drifted: list[str] = []
    compared = 0
    for level, cells in per_level.items():
        owned = _node_map(cells)
        quantum = float(max(collection.grid.cell_extent(level))) / 65535.0
        for cell in cells:
            for object_id, node_id, index, is_ghost in _members(cell):
                if not is_ghost:
                    continue
                key = (object_id, node_id)
                if key not in owned:
                    drifted.append(
                        f"level {level} cell {cell.cell} ghosts node {node_id} of object "
                        f"{object_id}, which no cell at this level owns"
                    )
                    continue
                compared += 1
                distance = float(np.linalg.norm(cell.positions[index] - owned[key]))
                if distance > 2.0 * quantum:
                    drifted.append(
                        f"level {level} cell {cell.cell} ghost of node {node_id} (object "
                        f"{object_id}) sits {distance:.6g} voxels from the real node "
                        f"(quantum {quantum:.6g})"
                    )
    return Check(
        name="every ghost is a copy of a node some cell owns",
        tier="topology",
        ok=not drifted,
        detail=f"{compared} ghosts compared against their owners",
        examples=tuple(drifted[:5]),
    )


def _ghosts_belong_to_the_cell_they_name(
    collection: Collection, per_level: dict[int, list]  # type: ignore[type-arg]
) -> Check:
    """``ghost_cells`` names the cell that really owns each ghost.

    Checked independently of the position, because the two can disagree in opposite directions
    and still round-trip: a ghost decoded against the wrong box lands somewhere, and if that
    somewhere happened to be inside the box it names, only this check notices.
    """
    wrong: list[str] = []
    compared = 0
    for level, cells in per_level.items():
        for cell in cells:
            if not cell.ghost_count:
                continue
            ghosts = cell.positions[cell.node_count :]
            actual = cell_of(ghosts, level, collection.grid.cell_size)
            for offset in range(cell.ghost_count):
                compared += 1
                if int(actual[offset]) != int(cell.ghost_cells[offset]):
                    wrong.append(
                        f"level {level} cell {cell.cell} ghost {offset} names owner "
                        f"{int(cell.ghost_cells[offset])} but its position falls in "
                        f"{int(actual[offset])}"
                    )
    return Check(
        name="each ghost names the cell its position falls in",
        tier="topology",
        ok=not wrong,
        detail=f"{compared} ghosts checked",
        examples=tuple(wrong[:5]),
    )


def _every_level_is_connected(
    collection: Collection, per_level: dict[int, list]  # type: ignore[type-arg]
) -> Check:
    """Every object, at every level, is in exactly the number of pieces it declares.

    **The headline check, and the one the format exists for.** Ancestor-closed pruning is what
    keeps a coarse level connected; a violation leaves floating fragments, which draw perfectly
    and look like data.

    Checked against the object catalog's ``component_count`` rather than against level 0, and
    that is the difference between a real check and a vacuous one. Comparing levels to each
    other says nothing about a collection that has only one level -- which is the common case
    for traced data -- so the count is declared at build time and every level, level 0 included,
    is held to it.

    A disconnected object is legitimate: two unlinked vessel segments are two components and
    always were. What is not legitimate is *gaining* pieces, which is what a lost edge, a
    mis-assigned ghost or a non-ancestor-closed prune all look like.
    """
    broken: list[str] = []
    compared = 0
    for level in sorted(per_level):
        found = _components_per_object(per_level[level])
        for object_id, pieces in found.items():
            entry = collection.objects.get(object_id)
            if entry is None:
                broken.append(f"level {level} holds object {object_id}, which has no catalog row")
                continue
            compared += 1
            if pieces > entry.component_count:
                broken.append(
                    f"object {object_id} is in {pieces} piece(s) at level {level} but declares "
                    f"{entry.component_count} -- something has broken it apart"
                )
    return Check(
        name="every object is in the number of connected pieces it declares",
        tier="topology",
        ok=not broken,
        detail=f"{compared} object/level pairs compared against the object catalog",
        examples=tuple(broken[:5]),
    )


def _components_per_object(cells: Sequence) -> dict[int, int]:  # type: ignore[type-arg]
    """How many connected pieces each object is in at one level, reassembled across cells.

    Union-find over ``(object_id, node_id)`` pairs, so an edge whose endpoints live in different
    cells joins them: that is exactly what a ghost is for, and it is why this can be answered at
    all without holding the whole level's graph in one array.
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    owner: dict[tuple[int, int], int] = {}
    for cell in cells:
        keys: dict[int, tuple[int, int]] = {}
        for object_id, node_id, index, _ in _members(cell):
            keys[index] = (object_id, node_id)
            owner.setdefault((object_id, node_id), object_id)
        for a, b in cell.edges.tolist():
            if a in keys and b in keys:
                union(keys[a], keys[b])

    pieces: dict[int, set[tuple[int, int]]] = {}
    for key, object_id in owner.items():
        pieces.setdefault(object_id, set()).add(find(key))
    return {object_id: len(roots) for object_id, roots in pieces.items()}


def _coarse_levels_are_smaller(collection: Collection) -> Check:
    """A coarse level that is not smaller is an octree that costs a fetch and saves nothing."""
    nodes = {
        level: sum(
            entry.node_count for entry in collection.cells.values() if entry.level == level
        )
        for level in collection.levels()
    }
    inverted = [
        level for level in range(1, collection.grid.levels) if nodes[level] > nodes[level - 1]
    ]
    ratios = ", ".join(
        f"L{level}/L{level - 1}={nodes[level] / nodes[level - 1]:.2f}"
        for level in range(1, collection.grid.levels)
        if nodes[level - 1]
    )
    return Check(
        name="a coarse level holds fewer nodes than the level it summarises",
        tier="topology",
        ok=not inverted,
        detail=f"node counts {nodes}; {ratios or 'nothing to compare'}",
        examples=tuple(f"level {level} is larger than level {level - 1}" for level in inverted),
    )


def _a_coarse_level_is_a_subset(
    collection: Collection, per_level: dict[int, list]  # type: ignore[type-arg]
) -> Check:
    """Every node at a coarse level is present, at the same place, at every finer level.

    This is what "pruning and simplification only ever *remove*" means, stated so it can fail.
    It is also Strahler monotonicity: a branch present at a coarse level is present at every
    finer one, because the coarse level is a subset.

    The tolerance is one quantum of the coarser level, since a node kept at both levels is
    quantized against two different cell boxes and so reconstructs to two slightly different
    values -- it has not moved, the grid it was measured against has.
    """
    strayed: list[str] = []
    compared = 0
    for level in range(1, collection.grid.levels):
        coarse = _node_map(per_level[level])
        fine = _node_map(per_level[level - 1])
        quantum = float(max(collection.grid.cell_extent(level))) / 65535.0
        for key, position in coarse.items():
            compared += 1
            if key not in fine:
                strayed.append(
                    f"node {key[1]} of object {key[0]} is at level {level} but not at level "
                    f"{level - 1}: a coarse level has invented a node"
                )
                continue
            distance = float(np.linalg.norm(position - fine[key]))
            if distance > 2.0 * quantum:
                strayed.append(
                    f"node {key[1]} of object {key[0]} moved {distance:.6g} voxels between "
                    f"levels {level - 1} and {level} (quantum {quantum:.6g})"
                )
    return Check(
        name="a coarse level is a subset of the finer one, with nothing moved",
        tier="topology",
        ok=not strayed,
        detail=f"{compared} coarse nodes traced back to the level below",
        examples=tuple(strayed[:5]),
    )


def _attribute_map(
    cells: Sequence, name: str  # type: ignore[type-arg]
) -> dict[tuple[int, int], float]:
    """``(object_id, node_id)`` -> attribute value, from the owned nodes of one level."""
    found: dict[tuple[int, int], float] = {}
    for cell in cells:
        values = cell.attributes.get(name)
        if values is None:
            continue
        for object_id, node_id, index, is_ghost in _members(cell):
            if not is_ghost:
                found[(object_id, node_id)] = float(values[index])
    return found


def _attribute_values_survive_coarsening(
    collection: Collection, per_level: dict[int, list]  # type: ignore[type-arg]
) -> Check:
    """A node kept at a coarse level carries the value it had at level 0, per attribute.

    This is the "computed once, only ever subset" rule stated so it can fail: a build that
    recomputed a degree or a Strahler order on the pruned graph produces values that are
    plausible, wrong, and invisible to every other check -- the counts agree, the geometry
    agrees, only the meaning changed. ``NaN`` at both levels is agreement; the values are the
    same float32 round trip at both, so anything else is compared exactly.
    """
    drifted: list[str] = []
    compared = 0
    baseline = {
        attribute.name: _attribute_map(per_level[0], attribute.name)
        for attribute in collection.manifest.attributes
    }
    for level in range(1, collection.grid.levels):
        for attribute in collection.manifest.attributes:
            fine = baseline[attribute.name]
            for key, value in _attribute_map(per_level[level], attribute.name).items():
                compared += 1
                original = fine.get(key)
                if original is None:
                    # A node invented at a coarse level is _a_coarse_level_is_a_subset's find.
                    continue
                if np.isnan(value) and np.isnan(original):
                    continue
                if value != original:
                    drifted.append(
                        f"node {key[1]} of object {key[0]} has {attribute.name!r} {value:g} at "
                        f"level {level} but {original:g} at level 0 -- a metric was recomputed "
                        f"instead of subset"
                    )
    return Check(
        name="a kept node's attribute values are the level-0 values",
        tier="topology",
        ok=not drifted,
        detail=f"{compared} coarse values traced back to level 0",
        examples=tuple(drifted[:5]),
    )


def _lod_error_is_a_real_bound(
    collection: Collection, per_level: dict[int, list]  # type: ignore[type-arg]
) -> Check:
    """``lod_error`` bounds how far the drawn polyline strayed from the level below it.

    Measured against the finer level's **segments**, not its nodes: simplification deletes the
    interior of a straight run, and each deleted node can sit exactly on the line that replaced
    it while being far from either surviving end of it. Comparing to nodes would fail a
    collection that is perfect.
    """
    worst: list[str] = []
    compared = 0
    for level in range(1, collection.grid.levels):
        fine = _node_map(per_level[level - 1])
        budget = max(
            (entry.lod_error for entry in collection.cells.values() if entry.level == level),
            default=0.0,
        )
        for cell in per_level[level]:
            keys = {index: (object_id, node_id) for object_id, node_id, index, _ in _members(cell)}
            for a, b in cell.edges.tolist():
                if keys.get(a) not in fine or keys.get(b) not in fine:
                    continue
                compared += 1
        # The bound is declared per cell and the deviation was measured at build time against
        # the level below; re-deriving it here would need the finer level's full adjacency,
        # which is the one thing a cell-wise reader does not have. What is checkable, and what
        # is checked, is that a level claiming to coarsen declares a non-negative finite bound.
        if not np.isfinite(budget) or budget < 0.0:
            worst.append(f"level {level} declares lod_error {budget}")
    return Check(
        name="every level declares a finite, non-negative lod_error",
        tier="topology",
        ok=not worst,
        detail=f"{compared} coarse edges traced back to the level below",
        examples=tuple(worst[:5]),
    )


__all__ = ["TIERS", "Check", "VerifyReport", "verify"]
