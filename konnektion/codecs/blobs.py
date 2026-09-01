"""Per-cell quantization, ghost resolution, and the dispatch from ``codec`` to an implementation.

This is the format-level pair a writer and a reader call, and it is where the part that is *not*
the codec's business lives: quantizing a node against its cell's grid box on the way in,
inverting that on the way out, and refusing a node that does not belong to the cell.

positions
---------
``UINT16_QUANTIZED_PER_CELL``. Each node becomes three ``uint16`` quantized against **the cell's
grid box** -- not the data's bounding box, so a decoder needs only ``level`` and ``cell`` to
invert it::

    origin = morton_to_triple(cell) * cell_size * 2**level
    extent = cell_size * 2**level
    p = origin + q / 65535 * extent

edges
-----
``UINT32_PAIRS``. Two indices per edge into the cell's **concatenated** node array -- so an
object's edges are offset-corrected by its own node start, not local to the object.

ghosts
------
``TRAILING_PER_OWNER_CELL``. A cell's node array is its ``node_count`` owned nodes followed by
its ``ghost_count`` ghosts, so an edge index at or past ``node_count`` addresses a ghost and no
mask is needed -- the layout carries what a bitset would have said.

**Each ghost is quantized against the box of the cell that owns it**, named per ghost in
``ghost_cells``. That is forced rather than chosen: a ghost is by definition a node outside this
cell, so this cell's box cannot represent it -- the normalized coordinate lands past 1.0 and the
encoder refuses it. Inverting against the owner's box also makes the reconstruction bit-identical
to the one the owning cell stores, which is what makes a ghost a *copy* rather than a second,
slightly different opinion about where a node is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from konnektion.codecs.protocol import BlobCodec
from konnektion.codecs.raw import RawCodec
from konnektion.errors import FormatError, PartitioningError
from konnektion.manifest import (
    ATTRIBUTE_FLOAT32,
    CODEC_NONE,
    COMPRESSION_NONE,
    NODE_IDS_UINT32,
    NODE_IDS_UINT64,
    RADII_FLOAT32,
    RADII_NONE,
    RADII_UINT16_QUANTIZED_PER_CELL,
)
from konnektion.octree import cell_box

#: The quantization denominator, odd so that 0 and QUANT_MAX both land exactly on a cell face.
QUANT_MAX = 65535

#: Every codec the format defines, keyed by the ``encoding.codec`` value that selects it. A new
#: codec is an entry here and a module beside this one; nothing above the package changes.
_CODECS: dict[str, BlobCodec] = {
    CODEC_NONE: RawCodec(),
}

#: The numpy dtype each ``encoding.nodeIds`` value names.
_NODE_ID_DTYPES: dict[str, str] = {
    NODE_IDS_UINT64: "<u8",
    NODE_IDS_UINT32: "<u4",
}


def codec_for(codec: str) -> BlobCodec:
    """The implementation a manifest's ``codec`` value names, or a refusal naming what exists.

    A codec the format does not define is refused rather than guessed at: nothing in the bytes
    reveals how they were packed, so a guess here is not an error but geometry that decodes to
    garbage.
    """
    try:
        return _CODECS[codec]
    except KeyError:
        raise FormatError(
            f"`codec` is {codec!r}; the format defines {', '.join(sorted(_CODECS))}."
        ) from None


def node_id_dtype(declaration: str) -> str:
    """The numpy dtype an ``encoding.nodeIds`` value names."""
    try:
        return _NODE_ID_DTYPES[declaration]
    except KeyError:
        raise FormatError(
            f"`encoding.nodeIds` is {declaration!r}; the format defines "
            f"{', '.join(sorted(_NODE_ID_DTYPES))}."
        ) from None


# --------------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------------- #


def encode_positions(
    positions: npt.ArrayLike,
    *,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack node positions into the format's per-cell quantized uint16 blob.

    A node outside the cell is an **error, not something to clamp**. Clamping is what makes a
    partitioning bug invisible: the stray node is flattened onto the cell face, the blob is the
    right length, the columns are the right type, a schema check passes, and the only symptom is
    a graph quietly welded to a wall. Nothing downstream can detect it, so it is caught here or
    not at all.
    """
    implementation = codec_for(codec)
    origin, extent = cell_box(cell, level, cell_size)
    normalized = (np.asarray(positions, dtype=np.float64) - origin) / extent

    # A boundary node is pinned to exactly 0.0 or 1.0, so the slack only has to absorb
    # floating-point noise -- a fraction of a quantum, not a quantum.
    tolerance = 0.5 / QUANT_MAX
    if normalized.size and (normalized.min() < -tolerance or normalized.max() > 1.0 + tolerance):
        stray = int(((normalized < -tolerance) | (normalized > 1.0 + tolerance)).any(axis=1).sum())
        raise PartitioningError(
            f"{stray} node(s) fall outside cell {cell} at level {level}, whose box is "
            f"{origin.tolist()} + {extent.tolist()} voxels (worst normalized coordinate "
            f"{normalized.min():.6f} .. {normalized.max():.6f}). Quantization is per cell, so a "
            f"node outside the cell cannot be represented -- this is a partitioning bug, not a "
            f"rounding one."
        )

    quantized = np.rint(np.clip(normalized, 0.0, 1.0) * QUANT_MAX).astype("<u2")
    return implementation.encode_positions(quantized, compression=compression)


def decode_positions(
    blob: bytes,
    *,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    node_count: int | None = None,
) -> npt.NDArray[np.float64]:
    """Unpack a ``positions`` blob back into an ``(n, 3)`` float array of voxel coordinates.

    The executable half of the format documentation, and what the round-trip check asserts
    against.
    """
    implementation = codec_for(codec)
    if compression != COMPRESSION_NONE and node_count is None:
        raise FormatError(
            "`node_count` is required to decode a compressed positions blob: it is how the "
            "uncompressed length is known, since the format's ZSTD framing carries no size of "
            "its own."
        )
    origin, extent = cell_box(cell, level, cell_size)
    quantized = implementation.decode_positions(
        blob, compression=compression, node_count=node_count
    )
    return origin + np.asarray(quantized, dtype=np.float64) / QUANT_MAX * extent


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def encode_edges(
    edges: npt.ArrayLike,
    *,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack edges into the format's uint32 pair blob.

    Edge order is the order handed in, which is what keeps ``object_edge_offsets`` meaningful:
    an object's range into the concatenated arrays has to still be its own after a round trip.
    """
    implementation = codec_for(codec)
    pairs = np.asarray(edges, dtype="<u4").reshape(-1, 2)
    return implementation.encode_edges(pairs, compression=compression)


def decode_edges(
    blob: bytes,
    *,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    edge_count: int | None = None,
) -> npt.NDArray[np.int64]:
    """Unpack an ``edges`` blob back into an ``(m, 2)`` array."""
    implementation = codec_for(codec)
    if compression != COMPRESSION_NONE and edge_count is None:
        raise FormatError(
            "`edge_count` is required to decode a compressed edges blob: it is how the "
            "uncompressed length is known, since the format's ZSTD framing carries no size of "
            "its own."
        )
    return implementation.decode_edges(blob, compression=compression, edge_count=edge_count)


# --------------------------------------------------------------------------- #
# node ids
# --------------------------------------------------------------------------- #


def encode_node_ids(
    node_ids: npt.ArrayLike,
    *,
    declaration: str = NODE_IDS_UINT64,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack the global node ids of a cell, in the cell's own node order."""
    implementation = codec_for(codec)
    dtype = node_id_dtype(declaration)
    values = np.asarray(node_ids, dtype=np.int64)
    ceiling = np.iinfo(np.dtype(dtype)).max
    if values.size and (values.min() < 0 or values.max() > ceiling):
        raise FormatError(
            f"A node id must fit `encoding.nodeIds` = {declaration!r}, which holds 0..{ceiling}; "
            f"this cell has {values.min()}..{values.max()}. Declare {NODE_IDS_UINT64} for ids "
            f"this large."
        )
    return implementation.encode_scalars(values.astype(dtype), compression=compression)


def decode_node_ids(
    blob: bytes,
    *,
    declaration: str = NODE_IDS_UINT64,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    node_count: int | None = None,
) -> npt.NDArray[np.int64]:
    """Unpack a cell's global node ids."""
    implementation = codec_for(codec)
    dtype = node_id_dtype(declaration)
    values = implementation.decode_scalars(
        blob, dtype=dtype, compression=compression, count=node_count
    )
    return np.asarray(values, dtype=np.int64)


# --------------------------------------------------------------------------- #
# radii
# --------------------------------------------------------------------------- #


def encode_radii(
    radii: npt.ArrayLike,
    *,
    declaration: str,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack per-node radii at the width the manifest declares.

    ``UINT16_QUANTIZED_PER_CELL`` quantizes against the cell's **largest extent** rather than
    per component: a radius is one scalar and has no axis, so quantizing it against a
    per-component box would make the same physical radius encode differently depending on which
    way an anisotropic cell was longest.
    """
    if declaration == RADII_NONE:
        raise FormatError(
            "This collection declares `encoding.radii: NONE`, so it has no radius column to "
            "write. Declare FLOAT32 or UINT16_QUANTIZED_PER_CELL to carry one."
        )
    implementation = codec_for(codec)
    values = np.asarray(radii, dtype=np.float64)
    if declaration == RADII_FLOAT32:
        return implementation.encode_scalars(values.astype("<f4"), compression=compression)
    if declaration == RADII_UINT16_QUANTIZED_PER_CELL:
        _, extent = cell_box(cell, level, cell_size)
        span = float(np.max(extent))
        if values.size and (values.min() < 0.0 or values.max() > span):
            raise PartitioningError(
                f"A radius of {values.max():.6g} voxels does not fit cell {cell} at level "
                f"{level}, whose largest extent is {span:g}. A quantized radius is measured "
                f"against that extent, so one larger than the cell cannot be represented -- "
                f"declare `radii: {RADII_FLOAT32}` for a collection with objects thicker than "
                f"their cells."
            )
        quantized = np.rint(np.clip(values, 0.0, span) / span * QUANT_MAX).astype("<u2")
        return implementation.encode_scalars(quantized, compression=compression)
    raise FormatError(f"`encoding.radii` is {declaration!r}, which this writer does not know.")


def decode_radii(
    blob: bytes,
    *,
    declaration: str,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    node_count: int | None = None,
) -> npt.NDArray[np.float64]:
    """Unpack per-node radii back into voxels."""
    implementation = codec_for(codec)
    if declaration == RADII_FLOAT32:
        values = implementation.decode_scalars(
            blob, dtype="<f4", compression=compression, count=node_count
        )
        return np.asarray(values, dtype=np.float64)
    if declaration == RADII_UINT16_QUANTIZED_PER_CELL:
        quantized = implementation.decode_scalars(
            blob, dtype="<u2", compression=compression, count=node_count
        )
        _, extent = cell_box(cell, level, cell_size)
        return np.asarray(quantized, dtype=np.float64) / QUANT_MAX * float(np.max(extent))
    raise FormatError(
        f"`encoding.radii` is {declaration!r}; there is nothing to decode for {RADII_NONE}."
    )


# --------------------------------------------------------------------------- #
# attributes
# --------------------------------------------------------------------------- #


def encode_attribute_values(
    values: npt.ArrayLike,
    *,
    declaration: str,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack one attribute's per-node values at the width the manifest declares.

    ``FLOAT32`` and nothing else in version 1: an attribute is a metric whose "no value" is
    ``NaN``, which the quantized-integer trick radii play has no way to say. Ghost values are
    packed by the same call -- a float needs no cell box, so there is no owner-cell dance.
    """
    if declaration != ATTRIBUTE_FLOAT32:
        raise FormatError(
            f"An attribute's `encoding` is {declaration!r}; the format defines "
            f"{ATTRIBUTE_FLOAT32}."
        )
    implementation = codec_for(codec)
    return implementation.encode_scalars(
        np.asarray(values, dtype=np.float64).astype("<f4"), compression=compression
    )


def decode_attribute_values(
    blob: bytes,
    *,
    declaration: str,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    count: int | None = None,
) -> npt.NDArray[np.float64]:
    """Unpack one attribute's values back into floats, ``NaN`` where the graph had no answer."""
    if declaration != ATTRIBUTE_FLOAT32:
        raise FormatError(
            f"An attribute's `encoding` is {declaration!r}; the format defines "
            f"{ATTRIBUTE_FLOAT32}."
        )
    implementation = codec_for(codec)
    values = implementation.decode_scalars(blob, dtype="<f4", compression=compression, count=count)
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------- #
# ghosts
# --------------------------------------------------------------------------- #


def encode_ghost_cells(
    cells: npt.ArrayLike, *, compression: str = COMPRESSION_NONE, codec: str = CODEC_NONE
) -> bytes:
    """Pack the owning-cell Morton code of each ghost, in the cell's ghost order."""
    implementation = codec_for(codec)
    return implementation.encode_scalars(
        np.asarray(cells, dtype="<u8"), compression=compression
    )


def decode_ghost_cells(
    blob: bytes,
    *,
    compression: str = COMPRESSION_NONE,
    codec: str = CODEC_NONE,
    ghost_count: int | None = None,
) -> npt.NDArray[np.int64]:
    """Unpack the owning-cell code of each ghost."""
    implementation = codec_for(codec)
    values = implementation.decode_scalars(
        blob, dtype="<u8", compression=compression, count=ghost_count
    )
    return np.asarray(values, dtype=np.int64)


def encode_ghost_positions(
    positions: npt.ArrayLike,
    owner_cells: npt.ArrayLike,
    *,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack ghost positions, **each quantized against the cell that owns it**.

    Not against the cell holding the row -- that is the whole point of a ghost and the one thing
    about it that is not obvious. A ghost is by definition a node outside this cell, so this
    cell's box cannot represent it: the normalized coordinate lands past 1.0 and
    :func:`encode_positions` refuses it, correctly. Quantizing against the owner's box keeps
    every coordinate in range and keeps the reconstruction bit-identical to the one the owning
    cell stores, which is what makes the ghost a *copy* rather than an approximation.
    """
    implementation = codec_for(codec)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    owners = np.asarray(owner_cells, dtype=np.int64).reshape(-1)
    if len(points) != len(owners):
        raise FormatError(
            f"Every ghost names the cell that owns it, got {len(points)} positions and "
            f"{len(owners)} cells."
        )
    quantized = np.zeros((len(points), 3), dtype="<u2")
    for index in range(len(points)):
        origin, extent = cell_box(int(owners[index]), level, cell_size)
        normalized = (points[index] - origin) / extent
        tolerance = 0.5 / QUANT_MAX
        if normalized.min() < -tolerance or normalized.max() > 1.0 + tolerance:
            raise PartitioningError(
                f"A ghost of cell {int(owners[index])} at level {level} does not fit that "
                f"cell's box (normalized {normalized.min():.6f} .. {normalized.max():.6f}). The "
                f"ghost names the wrong owner, which is a partitioning bug."
            )
        quantized[index] = np.rint(np.clip(normalized, 0.0, 1.0) * QUANT_MAX).astype("<u2")
    return implementation.encode_positions(quantized, compression=compression)


def decode_ghost_positions(
    blob: bytes,
    owner_cells: npt.ArrayLike,
    *,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    ghost_count: int | None = None,
) -> npt.NDArray[np.float64]:
    """Unpack ghost positions, inverting each against the cell that owns it."""
    implementation = codec_for(codec)
    owners = np.asarray(owner_cells, dtype=np.int64).reshape(-1)
    quantized = implementation.decode_positions(
        blob, compression=compression, node_count=ghost_count
    )
    points = np.zeros((len(quantized), 3), dtype=np.float64)
    for index in range(len(quantized)):
        origin, extent = cell_box(int(owners[index]), level, cell_size)
        points[index] = origin + quantized[index] / QUANT_MAX * extent
    return points


__all__ = [
    "QUANT_MAX",
    "codec_for",
    "decode_attribute_values",
    "decode_edges",
    "decode_ghost_cells",
    "decode_ghost_positions",
    "decode_node_ids",
    "decode_positions",
    "decode_radii",
    "encode_attribute_values",
    "encode_edges",
    "encode_ghost_cells",
    "encode_ghost_positions",
    "encode_node_ids",
    "encode_positions",
    "encode_radii",
    "node_id_dtype",
]
