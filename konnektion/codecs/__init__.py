"""The byte format: Morton codes, per-cell quantization, ghost resolution, and the blob codecs.

**This package is the wire format.** A decoder in any language needs everything stated here and
nothing else, which is why the inverses live next to the encoders rather than in a test.

Where each part is
------------------
:mod:`~konnektion.codecs.blobs` is the quantization either side of a codec, and the dispatch from
a manifest's ``codec`` value to an implementation. :mod:`~konnektion.codecs.protocol` states what
an implementation is; :mod:`~konnektion.codecs.raw` is the one that ships, and
:mod:`~konnektion.codecs.compression` is the blob compression declared alongside it.

How space is divided -- the octree levels, the Morton code a ``cell`` is, and the box it stands
for -- is :mod:`konnektion.octree`, not this package. Quantization needs a cell's box and imports
it from there; nothing in the addressing needs to know how a blob is packed.

Components are referred to as ``x``, ``y`` and ``z`` throughout, and in the ``bbox_*`` column
names, purely as labels for slots 0, 1 and 2 -- the code never asks which physical axis a slot
holds, and a collection whose components are ``(z, y, x)`` encodes and decodes identically. What
a slot *means* is not stated anywhere in the format: it is a claim about the collection's
relation to something else, and it belongs to whatever owns that coordinate system.

positions
---------
``UINT16_QUANTIZED_PER_CELL``. Three ``uint16`` per node, quantized against the cell's own grid
box, written little-endian and interleaved -- so the blob is exactly ``6 * node_count`` bytes and
a reader can hand the column straight to a vertex buffer.

edges
-----
``UINT32_PAIRS``, **two** per edge, indexing the cell's **concatenated** node array. The blob is
a flat little-endian ``uint32`` segment list, in the order the writer emitted -- which is what
keeps ``object_edge_offsets`` meaningful.

The field is required and always stated, and of everything in ``encoding`` it is the one whose
absence would be least survivable. Arity is the only thing distinguishing this blob from a mesh's
index buffer, and reading a segment list at arity three produces no error at any layer: the
length divides whenever the edge count is a multiple of three, every index is in range, and the
picture is a plausible, wrong graph. There is nothing downstream that could notice.

nodeIds
-------
``UINT64`` (or ``UINT32``), one per node, in the cell's own node order. A node's identity is
global and survives being copied into a neighbouring cell as a ghost, which is what lets a client
that cares dedup across cells -- and what lets the verifier check that a ghost really is a copy
of the node it claims to be.

ghosts
------
``TRAILING_PER_OWNER_CELL``. A cell's node array is its ``node_count`` owned nodes followed by its
``ghost_count`` ghosts, each quantized against the box of the cell that owns it -- named per ghost
in ``ghost_cells``. An edge index at or past ``node_count`` therefore addresses a ghost, and no
mask is needed.

Ghosts are how konnektion cuts. A mesh crossing a cell plane is *split*, and the new vertices are
real geometry; a graph crossing a cell plane cannot be split without inventing a node, and an
invented degree-2 node in a morphology is a measurement artefact rather than a rendering detail.
So the edge is kept whole and the foreign endpoint is copied. The cost is one duplicated node per
crossing; the gain is that a cell remains self-contained, which is the entire premise of the
octree -- fetch it and you can draw it, with no walk to an ancestor.

**An edge has exactly one owning cell**, the one with the lower Morton code of its two endpoints'
cells, and that is the cell holding the ghost. Putting a crossing edge in both cells would also be
self-contained, and would draw the segment twice; under any blending but opaque that shows up as a
brighter line running along every cell plane.

radii
-----
``NONE``, ``FLOAT32``, or ``UINT16_QUANTIZED_PER_CELL`` against the cell's largest extent -- one
per node, and absent from the shard columns entirely when ``NONE``.

Stored, where a mesh's normals deliberately are not. fabriks's ``NORMALS.md`` argues that a
normal is recoverable from positions plus winding, so storing it is redundancy rather than
information. A radius is recoverable from nothing: it is the calibre of the dendrite, the bore of
the vessel, a measurement the tracer made. Dropping it would lose data, and a renderer that wants
tapered segments has nowhere else to get it.

codec / compression
-------------------
``codec`` defaults to ``NONE``: a blob is the renderer's buffer verbatim, which is the point
rather than a shortfall -- a consumer reads the column and uploads it, with no decoder in front
of the geometry at all.

The field is **required and always stated**, for the reason it always was: nothing in the bytes
reveals how they were packed, so a reader handed a manifest without it would be guessing, and a
guess here is not an error but geometry that decodes to garbage. A second codec would arrive as a
*value* -- a module beside these and an entry in the table in :mod:`~konnektion.codecs.blobs` --
not as a new field.

``compression`` likewise defaults to ``NONE``. The Parquet file around the blobs is already
zstd-compressed, which is why the raw blobs cost roughly half rather than the multiple their
sizes suggest.
"""

from konnektion.codecs.blobs import (
    QUANT_MAX,
    codec_for,
    decode_attribute_values,
    decode_edges,
    decode_ghost_cells,
    decode_ghost_positions,
    decode_node_ids,
    decode_positions,
    decode_radii,
    encode_attribute_values,
    encode_edges,
    encode_ghost_cells,
    encode_ghost_positions,
    encode_node_ids,
    encode_positions,
    encode_radii,
    node_id_dtype,
)
from konnektion.codecs.compression import compress, decompress, require_known_compression
from konnektion.codecs.protocol import BlobCodec
from konnektion.codecs.raw import RawCodec

__all__ = [
    "QUANT_MAX",
    "BlobCodec",
    "RawCodec",
    "codec_for",
    "compress",
    "decode_attribute_values",
    "decode_edges",
    "decode_ghost_cells",
    "decode_ghost_positions",
    "decode_node_ids",
    "decode_positions",
    "decode_radii",
    "decompress",
    "encode_attribute_values",
    "encode_edges",
    "encode_ghost_cells",
    "encode_ghost_positions",
    "encode_node_ids",
    "encode_positions",
    "encode_radii",
    "node_id_dtype",
    "require_known_compression",
]
