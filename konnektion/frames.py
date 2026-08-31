"""The three Parquet schemas, and the columns each role must carry.

Nothing that reads a collection renders these files itself: a server checks the columns and the
declarations and never opens a blob. So the column layer is the contract, and everything below
it is defined by :mod:`konnektion.codecs` and nowhere else.

Extra columns are allowed on purpose -- a writer may carry a denormalized attribute copy
alongside -- so a check tests that the required columns are present, never that no others are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from konnektion.errors import FormatError, MissingExtraError

try:
    import pyarrow as pa
except ImportError as _error:  # pragma: no cover - pyarrow is a hard dependency
    raise MissingExtraError(
        "pyarrow is required: a network collection is Parquet. Install it with "
        "`pip install konnektion`."
    ) from _error

#: What Parquet itself may compress a *file* with -- a separate thing from the format's
#: ``encoding.compression``, which describes each blob inside a row.
ParquetCompression = Literal["none", "snappy", "gzip", "brotli", "lz4", "zstd"]

#: The columns each role must carry, used to check a frame before an upload is spent on it.
#:
#: ``radii`` is **not** required: a collection whose ``encoding.radii`` is ``NONE`` has no radius
#: to store, and a required-but-empty column would be a place for a reader to find zeros and
#: believe them. The manifest says whether to look for it.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "cell_catalog": (
        "level", "cell", "node_count", "edge_count", "ghost_count",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
        "lod_error", "object_count", "child_mask",
        "part", "row_group", "blob_bytes",
    ),
    "object_catalog": (
        "object_id", "ordinal", "root_node_id", "component_count",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
        "node_count", "edge_count", "cells",
    ),
    "geometry": (
        "level", "cell", "positions", "node_ids", "edges",
        "ghost_positions", "ghost_cells", "ghost_ids",
        "node_count", "edge_count", "ghost_count",
        "object_ids", "object_ordinals",
        "object_node_offsets", "object_ghost_offsets", "object_edge_offsets",
    ),
}


def arrow_schemas() -> dict[str, pa.Schema]:
    """The three Arrow schemas, spelled so a DuckDB ``DESCRIBE`` prints what a server accepts."""
    bbox = [
        pa.field(f"bbox_{corner}_{axis}", pa.float64())
        for corner in ("min", "max")
        for axis in ("x", "y", "z")
    ]
    return {
        "cell_catalog": pa.schema([
            pa.field("level", pa.int32()),
            pa.field("cell", pa.int64()),
            pa.field("node_count", pa.int32()),
            pa.field("edge_count", pa.int32()),
            # Ghosts are counted separately from nodes because they are not data: they are
            # copies, and a client summing node_count across cells would otherwise double-count
            # every endpoint of every edge that crosses a cell plane.
            pa.field("ghost_count", pa.int32()),
            *bbox,
            pa.field("lod_error", pa.float64()),
            pa.field("object_count", pa.int32()),
            pa.field("child_mask", pa.uint8()),
            # The locator: which part of the level holds this cell, and which row group of it.
            # Null on a built-but-unwritten collection -- part assignment and row-group
            # boundaries are the writer's decisions and are not known until it makes them.
            pa.field("part", pa.int32(), nullable=True),
            pa.field("row_group", pa.int32(), nullable=True),
            pa.field("blob_bytes", pa.int64()),
        ]),
        "object_catalog": pa.schema([
            pa.field("object_id", pa.int64()),
            pa.field("ordinal", pa.int32()),
            # The node the ancestor-closed invariant is stated relative to. Null for an object
            # with no distinguished root -- a connectome component rather than a rooted tree --
            # in which case connectivity is checked per component instead.
            pa.field("root_node_id", pa.int64(), nullable=True),
            # How many connected pieces this object is in. Declared so that connectivity is
            # *falsifiable at every level, including level 0* -- without it the only checkable
            # statement is "no coarser than the level below", which says nothing at all about a
            # collection that has one level, and one level is the common case.
            pa.field("component_count", pa.int32()),
            *bbox,
            pa.field("node_count", pa.int32()),
            pa.field("edge_count", pa.int32()),
            pa.field("cells", pa.list_(pa.int64())),
        ]),
        "geometry": pa.schema([
            pa.field("level", pa.int32()),
            pa.field("cell", pa.int64()),
            pa.field("positions", pa.large_binary()),
            pa.field("node_ids", pa.large_binary()),
            pa.field("edges", pa.large_binary()),
            # The endpoints this cell does not own, quantized against the cell that does --
            # which `ghost_cells` names, one Morton code per ghost. They are the tail of the
            # node array, so an edge index past `node_count` addresses one of them.
            pa.field("ghost_positions", pa.large_binary()),
            pa.field("ghost_cells", pa.large_binary()),
            pa.field("ghost_ids", pa.large_binary()),
            # Optional, and present exactly when `encoding.radii` is not NONE.
            pa.field("radii", pa.large_binary(), nullable=True),
            pa.field("ghost_radii", pa.large_binary(), nullable=True),
            pa.field("node_count", pa.int32()),
            pa.field("edge_count", pa.int32()),
            pa.field("ghost_count", pa.int32()),
            pa.field("object_ids", pa.list_(pa.int64())),
            pa.field("object_ordinals", pa.list_(pa.int32())),
            pa.field("object_node_offsets", pa.list_(pa.int32())),
            # Ghosts are concatenated after every object's owned nodes, so they need their own
            # offsets: without them a ghost cannot be attributed to an object, and a verifier
            # matching a ghost to the node it copies has only its id to go on -- which is unique
            # within an object and need not be across a collection.
            pa.field("object_ghost_offsets", pa.list_(pa.int32())),
            pa.field("object_edge_offsets", pa.list_(pa.int32())),
        ]),
    }


def build_table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build an Arrow table from row dicts under a fixed schema, empty rows included."""
    columns = {
        field.name: [row.get(field.name) for row in rows] for field in schema
    }
    return pa.table(columns, schema=schema)


def _column_names(table: Any) -> Iterable[str]:  # noqa: ANN401
    """The column names of an Arrow table, a pandas frame, or anything with a schema."""
    names = getattr(table, "column_names", None)
    if names is not None:
        return list(names)
    schema = getattr(table, "schema", None)
    if schema is not None and hasattr(schema, "names"):
        return list(schema.names)
    columns = getattr(table, "columns", None)
    if columns is not None:
        return [str(column) for column in columns]
    raise FormatError(f"konnektion cannot read column names off a {type(table).__name__}.")


def validate_columns(table: Any, role: str) -> None:  # noqa: ANN401
    """Refuse a frame missing a column its role requires, before an upload is spent on it.

    The earliest point the mistake is catchable and the only point it is cheap.
    """
    try:
        required = REQUIRED_COLUMNS[role]
    except KeyError as error:
        raise FormatError(
            f"{role!r} is not a role of a network collection; try {', '.join(REQUIRED_COLUMNS)}."
        ) from error

    names = set(_column_names(table))
    missing = [column for column in required if column not in names]
    if missing:
        raise FormatError(
            f"This frame is being written as the {role.replace('_', ' ')} of a network "
            f"collection, and the format requires it to carry {', '.join(required)}. It is "
            f"missing {', '.join(missing)} (it has {', '.join(sorted(names))}). Build it with "
            f"`konnektion.build_collection`, which produces all three frames with the right "
            f"columns."
        )


def table_to_parquet(table: pa.Table, *, compression: ParquetCompression = "zstd") -> bytes:
    """Serialize an Arrow table to Parquet bytes.

    The Parquet-level compression is the *file's*, and is a separate thing from the format's:
    ``encoding.compression`` describes each blob *inside* a row, and defaults to ``NONE``. So
    this is what actually compresses the raw blobs on disk -- and it does it better than
    per-blob framing would, having a whole column chunk of context rather than one cell's worth.

    **The codec is not a free choice.** The viewer reads these parts with hyparquet, whose
    codecs are UNCOMPRESSED and SNAPPY plus the ZSTD it registers by hand out of ``fzstd``. A
    gzip or brotli default would upload cleanly, verify cleanly, and draw nothing.
    """
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=compression)
    return bytes(sink.getvalue().to_pybytes())


#: How many bytes of geometry a row group aims to hold. **This is the knob the whole
#: range-reading story turns on**, and it is two-sided: a row group is the smallest thing a
#: reader can fetch, so a large one means fetching a cell drags its neighbours along -- but a
#: Parquet footer grows with row-group count, and the footer is read on every part a reader
#: opens.
#:
#: 512 KiB sits where a row group holds a handful of cells rather than one or hundreds. The
#: footer is cached per part for the life of a reader, so its cost is paid once per session
#: while the row-group cost is paid per fetch -- the asymmetry this number is picked against.
DEFAULT_ROW_GROUP_BYTES = 512 * 1024


def blob_sizes(table: pa.Table) -> list[int]:
    """How many bytes of geometry each row of a shard carries.

    Budgeting on the blobs rather than on a row count is what keeps chunks even: a cell holding
    one small object and a cell holding two hundred differ by orders of magnitude in bytes and
    not at all in rows.
    """
    names = set(table.column_names)
    columns = ["positions", "node_ids", "edges", "ghost_positions", "ghost_cells", "ghost_ids"]
    columns += [name for name in ("radii", "ghost_radii") if name in names]
    parts = [table.column(name).to_pylist() for name in columns]
    return [sum(len(blob or b"") for blob in row) for row in zip(*parts)]


def plan_byte_chunks(sizes: Sequence[int], budget: int) -> list[tuple[int, int]]:
    """Group consecutive rows into ``(start, count)`` runs that each fit ``budget`` bytes.

    Used twice, at two scales: to split a level into parts, and to split a part into row groups.
    A single row larger than the budget still goes in a run of its own rather than being split
    -- a cell is the smallest thing a reader fetches.
    """
    if not sizes:
        return [(0, 0)]
    chunks: list[tuple[int, int]] = []
    start = 0
    running = 0
    for row, size in enumerate(sizes):
        if running and running + size > budget:
            chunks.append((start, row - start))
            start, running = row, 0
        running += size
    chunks.append((start, len(sizes) - start))
    return chunks


def table_to_chunked_parquet(
    table: pa.Table,
    *,
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
    compression: ParquetCompression = "zstd",
) -> tuple[bytes, list[tuple[int, int]]]:
    """Serialize a geometry shard with one row group per byte-budgeted run of cells.

    Returns the Parquet bytes and the ``(start_row, row_count)`` of each row group, in order --
    which is what lets the writer record, per cell, the row group a reader must fetch to get it.
    """
    import pyarrow.parquet as pq

    chunks = plan_byte_chunks(blob_sizes(table), row_group_bytes)
    sink = pa.BufferOutputStream()
    with pq.ParquetWriter(sink, table.schema, compression=compression) as writer:
        for start, count in chunks:
            writer.write_table(table.slice(start, count))
    return bytes(sink.getvalue().to_pybytes()), chunks


def parquet_to_table(body: bytes) -> pa.Table:
    """Read Parquet bytes back into an Arrow table."""
    import pyarrow.parquet as pq

    return pq.read_table(pa.BufferReader(body))


__all__ = [
    "DEFAULT_ROW_GROUP_BYTES",
    "REQUIRED_COLUMNS",
    "ParquetCompression",
    "arrow_schemas",
    "blob_sizes",
    "build_table",
    "parquet_to_table",
    "plan_byte_chunks",
    "table_to_chunked_parquet",
    "table_to_parquet",
    "validate_columns",
]
