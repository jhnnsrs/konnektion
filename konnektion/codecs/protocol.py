"""What a blob codec is, stated as the calls the format needs from one.

A codec's job starts *after* quantization and ends before dequantization: it turns an ``(n, 3)``
array of ``uint16`` quantized positions, an ``(m, 2)`` array of ``uint32`` edge endpoints, a
node-id array or a radius array into the bytes that go in the column, and back. Everything a
codec would have to guess -- the cell's box, the node count, the declared compression -- is
handed to it, so an implementation is only ever the packing itself.

Which one runs is a *value* in the manifest, never a new field: ``encoding.codec`` names it, and
:func:`konnektion.codecs.blobs.codec_for` resolves that name to an implementation. Adding one
means a module here and an entry in that table, and nothing above the package changes.

Only :mod:`konnektion.codecs.raw` (``NONE``) ships today. There is no meshopt equivalent here
and its absence is worth stating: meshopt's vertex codec wants a stride that is a multiple of
four, and its *index* codec has two entry points -- ``encode_index_buffer``, which assumes a
triangle list, and ``encode_index_sequence``, which does not. A future ``MESHOPT`` codec for
konnektion must use the **sequence** pair. Reaching for the buffer pair because it is the one
fabriks calls would encode a segment list as though it were triangles, which does not fail: it
decodes to a plausible, wrong graph.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class BlobCodec(Protocol):
    """The calls a codec provides, plus the manifest value that selects it.

    Structural, like the store protocol: an implementation satisfies this by having the methods,
    not by inheriting anything.
    """

    #: The ``encoding.codec`` value that selects this implementation.
    name: str

    def encode_positions(self, quantized: npt.NDArray[np.uint16], *, compression: str) -> bytes:
        """Pack an ``(n, 3)`` array of ``uint16`` quantized coordinates into a blob."""
        ...

    def decode_positions(
        self, blob: bytes, *, compression: str, node_count: int | None
    ) -> npt.NDArray[np.uint16]:
        """Unpack a positions blob back into the ``(n, 3)`` ``uint16`` array that went in.

        ``node_count`` comes from the geometry row. Whether it is required or merely checked is
        the codec's own business: a raw blob is self-describing at six bytes a node, an encoded
        one is not.
        """
        ...

    def encode_edges(self, edges: npt.NDArray[np.uint32], *, compression: str) -> bytes:
        """Pack an ``(m, 2)`` array of cell-local endpoint indices, preserving edge order.

        Order is not cosmetic: ``object_edge_offsets`` names ranges into the concatenated array,
        so a codec that reordered edges would break slicing one object out of a shared cell.
        """
        ...

    def decode_edges(
        self, blob: bytes, *, compression: str, edge_count: int | None
    ) -> npt.NDArray[np.int64]:
        """Unpack an edges blob back into an ``(m, 2)`` array."""
        ...

    def encode_scalars(self, values: npt.NDArray[Any], *, compression: str) -> bytes:
        """Pack a flat per-node array -- node ids, or radii -- preserving dtype and order."""
        ...

    def decode_scalars(
        self, blob: bytes, *, dtype: str, compression: str, count: int | None
    ) -> npt.NDArray[Any]:
        """Unpack a flat per-node array, given the dtype the manifest declared for it."""
        ...


__all__ = ["BlobCodec"]
