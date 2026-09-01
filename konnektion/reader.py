"""Reading a collection back: the manifest, the catalogs, and one cell at a time.

The read path is the format's claims, executed. Everything the manifest declares -- the grid, the
encoding, where each cell's bytes are -- is used here rather than re-derived, so a collection that
reads correctly is one whose declarations were true.

**Opening does not read geometry.** :func:`open_collection` fetches the manifest and the two
catalogs and stops; the levels stay in the store until something asks for a cell. That is the
whole point of the locator on every catalog row -- a viewer that wants one cell pays for one row
group, not for a level.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from konnektion.codecs.blobs import (
    decode_attribute_values,
    decode_edges,
    decode_ghost_cells,
    decode_ghost_positions,
    decode_node_ids,
    decode_positions,
    decode_radii,
)
from konnektion.frames import attribute_column, ghost_attribute_column
from konnektion.errors import FormatError, UnfinishedCollectionError
from konnektion.frames import parquet_to_table
from konnektion.manifest import (
    CELL_CATALOG_PATH,
    MANIFEST_NAME,
    OBJECT_CATALOG_PATH,
    Manifest,
    level_part_path,
)
from konnektion.stores import KonnektionStore, get_bytes, join

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa


@dataclass(frozen=True)
class CellEntry:
    """One row of the cell catalog: what a cell holds and where its bytes are."""

    level: int
    cell: int
    node_count: int
    edge_count: int
    ghost_count: int
    bounds: tuple[float, float, float, float, float, float]
    lod_error: float
    object_count: int
    child_mask: int
    part: int | None
    row_group: int | None
    blob_bytes: int

    def children(self) -> list[tuple[int, int]]:
        """The ``(level, cell)`` keys of the children this cell's mask says exist."""
        if self.level == 0 or not self.child_mask:
            return []
        from konnektion.octree import morton_decode, morton_encode_one

        i, j, k = morton_decode(self.cell)
        found: list[tuple[int, int]] = []
        for bit, (di, dj, dk) in enumerate(
            [(a, b, c) for c in (0, 1) for b in (0, 1) for a in (0, 1)]
        ):
            if self.child_mask & (1 << bit):
                found.append(
                    (self.level - 1, morton_encode_one((2 * i + di, 2 * j + dj, 2 * k + dk)))
                )
        return found


@dataclass(frozen=True)
class ObjectEntry:
    """One row of the object catalog: an object's extent and which cells hold it."""

    object_id: int
    ordinal: int
    root_node_id: int | None
    component_count: int
    bounds: tuple[float, float, float, float, float, float]
    node_count: int
    edge_count: int
    cells: tuple[int, ...]


@dataclass(frozen=True)
class DecodedCell:
    """One cell's geometry, decoded.

    ``positions`` holds every node the cell draws: its own first, then its ghosts. ``is_ghost``
    is derived from the two counts rather than stored, because the layout already says it -- the
    last ``ghost_count`` entries are the copies.

    ``edges`` index into ``positions``, so an edge with an endpoint at or past ``node_count``
    reaches into a neighbouring cell. Drawing this cell alone is correct: the ghost carries the
    real coordinate of that endpoint, decoded against the cell that owns it.

    ``attributes`` holds one array per attribute the manifest declares, ordered like
    ``positions`` -- owned values first, then the ghosts' -- with ``NaN`` where the graph had
    no answer.
    """

    level: int
    cell: int
    positions: npt.NDArray[np.float64]
    node_ids: npt.NDArray[np.int64]
    edges: npt.NDArray[np.int64]
    ghost_cells: npt.NDArray[np.int64]
    radii: npt.NDArray[np.float64] | None
    node_count: int
    ghost_count: int
    object_ids: tuple[int, ...]
    object_node_offsets: tuple[int, ...]
    object_ghost_offsets: tuple[int, ...]
    object_edge_offsets: tuple[int, ...]
    attributes: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)

    @property
    def is_ghost(self) -> npt.NDArray[np.bool_]:
        """Which entries of ``positions`` are copies of a node another cell owns."""
        mask = np.zeros(len(self.positions), dtype=bool)
        mask[self.node_count :] = True
        return mask

    def owned_ids(self) -> npt.NDArray[np.int64]:
        """The global ids of the nodes this cell actually owns."""
        return self.node_ids[: self.node_count]

    def ghost_ids(self) -> npt.NDArray[np.int64]:
        """The global ids of the nodes this cell borrows."""
        return self.node_ids[self.node_count :]


@dataclass
class Collection:
    """An opened collection: the manifest, the catalogs, and a store to read cells from."""

    manifest: Manifest
    store: KonnektionStore
    prefix: str = ""
    cells: dict[tuple[int, int], CellEntry] = field(default_factory=dict)
    objects: dict[int, ObjectEntry] = field(default_factory=dict)
    _parts: dict[tuple[int, int], Any] = field(default_factory=dict, repr=False)

    @property
    def grid(self) -> Any:  # noqa: ANN401
        """The octree this collection is partitioned by."""
        return self.manifest.grid

    @property
    def encoding(self) -> Any:  # noqa: ANN401
        """How this collection's blobs are packed."""
        return self.manifest.encoding

    def levels(self) -> range:
        """Every level this collection has."""
        return range(self.grid.levels)

    def cells_at(self, level: int) -> list[CellEntry]:
        """Every cell of one level, in catalog order."""
        return [entry for key, entry in sorted(self.cells.items()) if key[0] == level]

    def _part_table(self, level: int, part: int) -> pa.Table:
        """One geometry part, read whole and cached.

        Whole rather than by row group, deliberately, and the honest limitation of this reader:
        the locator that makes a single-cell fetch possible is *written* correctly and the
        catalog carries it, but exploiting it needs a range-reading Parquet reader per row group.
        A viewer is where that pays; a verifier reads every cell anyway.
        """
        key = (level, part)
        if key not in self._parts:
            path = level_part_path(level, part)
            self._parts[key] = parquet_to_table(get_bytes(self.store, join(self.prefix, path)))
        return self._parts[key]

    def read_cell(self, level: int, cell: int) -> DecodedCell:
        """Decode one cell's geometry."""
        entry = self.cells.get((level, cell))
        if entry is None:
            raise FormatError(f"This collection has no cell {cell} at level {level}.")
        if entry.part is None:
            raise FormatError(
                f"Cell {cell} at level {level} has no `part` locator, so the catalog names a "
                f"cell the geometry does not hold. That is what an interrupted or hand-edited "
                f"write leaves behind."
            )
        table = self._part_table(level, entry.part)
        matches = [
            row
            for row in range(table.num_rows)
            if int(table.column("level")[row].as_py()) == level
            and int(table.column("cell")[row].as_py()) == cell
        ]
        if not matches:
            raise FormatError(
                f"Part {entry.part} of level {level} does not hold cell {cell}, which its "
                f"catalog row says it does."
            )
        return self._decode_row(table, matches[0])

    def _decode_row(self, table: pa.Table, row: int) -> DecodedCell:
        """Turn one geometry row into a :class:`DecodedCell`."""
        encoding = self.encoding
        cell_size = self.grid.cell_size
        get = lambda name: table.column(name)[row].as_py()

        level, cell = int(get("level")), int(get("cell"))
        node_count, ghost_count = int(get("node_count")), int(get("ghost_count"))
        edge_count = int(get("edge_count"))

        owned = decode_positions(
            get("positions"), cell=cell, level=level, cell_size=cell_size,
            codec=encoding.codec, compression=encoding.compression, node_count=node_count,
        )
        ghost_cells = decode_ghost_cells(
            get("ghost_cells"), codec=encoding.codec, compression=encoding.compression,
            ghost_count=ghost_count,
        )
        ghosts = decode_ghost_positions(
            get("ghost_positions"), ghost_cells, level=level, cell_size=cell_size,
            codec=encoding.codec, compression=encoding.compression, ghost_count=ghost_count,
        )
        owned_ids = decode_node_ids(
            get("node_ids"), declaration=encoding.node_ids, codec=encoding.codec,
            compression=encoding.compression, node_count=node_count,
        )
        ghost_ids = decode_node_ids(
            get("ghost_ids"), declaration=encoding.node_ids, codec=encoding.codec,
            compression=encoding.compression, node_count=ghost_count,
        )
        edges = decode_edges(
            get("edges"), codec=encoding.codec, compression=encoding.compression,
            edge_count=edge_count,
        )

        radii = None
        if encoding.has_radii:
            owned_radii = decode_radii(
                get("radii"), declaration=encoding.radii, cell=cell, level=level,
                cell_size=cell_size, codec=encoding.codec, compression=encoding.compression,
                node_count=node_count,
            )
            ghost_radii = _decode_ghost_radii(
                get("ghost_radii"), ghost_cells, level, cell_size, encoding, ghost_count
            )
            radii = np.concatenate([owned_radii, ghost_radii])

        attributes: dict[str, npt.NDArray[np.float64]] = {}
        for declared in self.manifest.attributes:
            owned_column = attribute_column(declared.name)
            ghost_column = ghost_attribute_column(declared.name)
            names = set(table.column_names)
            if owned_column not in names or ghost_column not in names:
                raise FormatError(
                    f"The manifest declares attribute {declared.name!r} but level {level} cell "
                    f"{cell}'s geometry has no {owned_column}/{ghost_column} columns. The "
                    f"manifest names what exists, so one of the two is lying."
                )
            owned_blob, ghost_blob = get(owned_column), get(ghost_column)
            if owned_blob is None or ghost_blob is None:
                raise FormatError(
                    f"The manifest declares attribute {declared.name!r} but level {level} cell "
                    f"{cell} holds no blob for it."
                )
            owned_values = decode_attribute_values(
                owned_blob, declaration=declared.encoding, codec=encoding.codec,
                compression=encoding.compression, count=node_count,
            )
            ghost_values = decode_attribute_values(
                ghost_blob, declaration=declared.encoding, codec=encoding.codec,
                compression=encoding.compression, count=ghost_count,
            )
            attributes[declared.name] = (
                np.concatenate([owned_values, ghost_values]) if ghost_count else owned_values
            )

        return DecodedCell(
            level=level,
            cell=cell,
            positions=np.concatenate([owned, ghosts]) if ghost_count else owned,
            node_ids=np.concatenate([owned_ids, ghost_ids]) if ghost_count else owned_ids,
            edges=edges,
            ghost_cells=ghost_cells,
            radii=radii,
            node_count=node_count,
            ghost_count=ghost_count,
            object_ids=tuple(int(v) for v in get("object_ids")),
            object_node_offsets=tuple(int(v) for v in get("object_node_offsets")),
            object_ghost_offsets=tuple(int(v) for v in get("object_ghost_offsets")),
            object_edge_offsets=tuple(int(v) for v in get("object_edge_offsets")),
            attributes=attributes,
        )

    def iter_cells(self, level: int) -> Iterator[DecodedCell]:
        """Every decoded cell of one level, without going through the catalog per cell."""
        parts = sorted({entry.part for entry in self.cells_at(level) if entry.part is not None})
        for part in parts:
            table = self._part_table(level, part)
            for row in range(table.num_rows):
                yield self._decode_row(table, row)

    def geometry_columns(self, level: int) -> dict[int, tuple[str, ...]]:
        """The column names of each geometry part of one level, keyed by part number.

        What a verifier compares the manifest's attribute declarations against: the columns are
        the one place an undeclared ``attr_*`` -- or a declared one that never landed -- shows
        up without decoding anything.
        """
        parts = sorted({entry.part for entry in self.cells_at(level) if entry.part is not None})
        return {part: tuple(self._part_table(level, part).column_names) for part in parts}


def _decode_ghost_radii(
    blob: Any,  # noqa: ANN401
    owners: npt.NDArray[np.int64],
    level: int,
    cell_size: Sequence[int],
    encoding: Any,  # noqa: ANN401
    ghost_count: int,
) -> npt.NDArray[np.float64]:
    """Decode ghost radii, each against the cell that owns it."""
    from konnektion.manifest import RADII_FLOAT32

    if not ghost_count:
        return np.zeros(0, dtype=np.float64)
    if encoding.radii == RADII_FLOAT32:
        return decode_radii(
            blob, declaration=encoding.radii, cell=0, level=level, cell_size=cell_size,
            codec=encoding.codec, compression=encoding.compression, node_count=ghost_count,
        )
    from konnektion.codecs.compression import decompress

    body = decompress(blob, encoding.compression, 2 * ghost_count)
    values = np.zeros(ghost_count, dtype=np.float64)
    for index in range(ghost_count):
        values[index] = decode_radii(
            body[2 * index : 2 * index + 2], declaration=encoding.radii,
            cell=int(owners[index]), level=level, cell_size=cell_size,
            codec=encoding.codec, node_count=1,
        )[0]
    return values


def _bounds(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
    """The six bbox columns of a catalog row, in a fixed order."""
    return (
        float(row["bbox_min_x"]), float(row["bbox_min_y"]), float(row["bbox_min_z"]),
        float(row["bbox_max_x"]), float(row["bbox_max_y"]), float(row["bbox_max_z"]),
    )


def open_collection(store: KonnektionStore, prefix: str = "") -> Collection:
    """Open a collection: read the manifest and both catalogs, and nothing else.

    A prefix with no manifest raises :class:`~konnektion.errors.UnfinishedCollectionError` rather
    than a generic missing-file error, because that is a specific and expected state: the
    manifest is written last, so its absence is what an interrupted upload looks like and is
    distinguishable from a corrupt collection.
    """
    try:
        body = get_bytes(store, join(prefix, MANIFEST_NAME))
    except (FileNotFoundError, KeyError) as error:
        raise UnfinishedCollectionError(
            f"There is no {MANIFEST_NAME} under {prefix or 'the store root'!r}. The manifest is "
            f"written last, so a prefix without one is an interrupted write rather than a "
            f"collection."
        ) from error

    manifest = Manifest.from_json(body)
    cells_table = parquet_to_table(get_bytes(store, join(prefix, CELL_CATALOG_PATH)))
    objects_table = parquet_to_table(get_bytes(store, join(prefix, OBJECT_CATALOG_PATH)))

    cells: dict[tuple[int, int], CellEntry] = {}
    for row in cells_table.to_pylist():
        entry = CellEntry(
            level=int(row["level"]),
            cell=int(row["cell"]),
            node_count=int(row["node_count"]),
            edge_count=int(row["edge_count"]),
            ghost_count=int(row["ghost_count"]),
            bounds=_bounds(row),
            lod_error=float(row["lod_error"]),
            object_count=int(row["object_count"]),
            child_mask=int(row["child_mask"] or 0),
            part=None if row["part"] is None else int(row["part"]),
            row_group=None if row["row_group"] is None else int(row["row_group"]),
            blob_bytes=int(row["blob_bytes"] or 0),
        )
        cells[(entry.level, entry.cell)] = entry

    objects: dict[int, ObjectEntry] = {}
    for row in objects_table.to_pylist():
        entry_object = ObjectEntry(
            object_id=int(row["object_id"]),
            ordinal=int(row["ordinal"]),
            root_node_id=None if row["root_node_id"] is None else int(row["root_node_id"]),
            component_count=int(row["component_count"]),
            bounds=_bounds(row),
            node_count=int(row["node_count"]),
            edge_count=int(row["edge_count"]),
            cells=tuple(int(value) for value in (row["cells"] or [])),
        )
        objects[entry_object.object_id] = entry_object

    return Collection(
        manifest=manifest, store=store, prefix=prefix, cells=cells, objects=objects
    )


__all__ = ["CellEntry", "Collection", "DecodedCell", "ObjectEntry", "open_collection"]
