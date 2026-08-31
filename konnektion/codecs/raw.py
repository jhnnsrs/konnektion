"""``codec: NONE`` -- the blob is the renderer's buffer verbatim.

The format's default, and the point of it rather than a shortfall: a consumer reads the column
and uploads it, with no decoder in front of the geometry at all. Positions are three
little-endian ``uint16`` interleaved, so the blob is exactly ``6 * node_count`` bytes; edges are
a flat little-endian ``uint32`` **pair** list at ``8 * edge_count``; node ids and radii are flat
arrays at their declared width.

Every blob being self-describing at a fixed width is what lets the counts in the geometry row be
*checked* here rather than needed -- a row that disagrees with its blob is a row and a geometry
that came from different writes. Under ``compression: ZSTD`` the check turns into a requirement,
since the compressed frame carries no reliable length of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from konnektion.codecs.compression import compress, decompress
from konnektion.errors import FormatError
from konnektion.manifest import CODEC_NONE


class RawCodec:
    """The identity codec: quantized values, little-endian, in the order they were handed over."""

    name = CODEC_NONE

    def __repr__(self) -> str:
        """Name the manifest value this implements."""
        return f"RawCodec({self.name!r})"

    def encode_positions(self, quantized: npt.NDArray[np.uint16], *, compression: str) -> bytes:
        """Interleave the ``uint16`` triples and hand back the bytes."""
        return compress(
            np.ascontiguousarray(quantized, dtype="<u2").reshape(-1).tobytes(), compression
        )

    def decode_positions(
        self, blob: bytes, *, compression: str, node_count: int | None
    ) -> npt.NDArray[np.uint16]:
        """Read the blob back as ``(n, 3)`` ``uint16``, checking the row's count against it."""
        blob = decompress(blob, compression, 6 * int(node_count or 0))
        quantized = np.frombuffer(blob, dtype="<u2").reshape(-1, 3)
        if node_count is not None and len(quantized) != node_count:
            raise FormatError(
                f"This positions blob holds {len(quantized)} nodes and its row declares "
                f"{node_count}. A blob is 6 bytes a node, so the two cannot disagree unless the "
                f"row and the geometry belong to different writes."
            )
        return quantized

    def encode_edges(self, edges: npt.NDArray[np.uint32], *, compression: str) -> bytes:
        """Flatten the pair list to little-endian ``uint32``."""
        return compress(
            np.ascontiguousarray(edges, dtype="<u4").reshape(-1).tobytes(), compression
        )

    def decode_edges(
        self, blob: bytes, *, compression: str, edge_count: int | None
    ) -> npt.NDArray[np.int64]:
        """Read the blob back as ``(m, 2)`` edges, checking the row's count against it.

        **The reshape is to two, and that is the whole difference from a mesh.** A flat
        ``uint32`` array reshaped to three divides evenly whenever the edge count is a multiple
        of three, indexes in range, and draws a plausible wrong picture. Nothing downstream
        catches it, which is why ``encoding.edges`` is a required key rather than a default.
        """
        blob = decompress(blob, compression, 8 * int(edge_count or 0))
        edges = np.frombuffer(blob, dtype="<u4").reshape(-1, 2).astype(np.int64)
        if edge_count is not None and len(edges) != edge_count:
            raise FormatError(
                f"This edges blob holds {len(edges)} edges and its row declares {edge_count}. A "
                f"blob is 8 bytes an edge, so the two cannot disagree unless the row and the "
                f"geometry belong to different writes."
            )
        return edges

    def encode_scalars(self, values: npt.NDArray[Any], *, compression: str) -> bytes:
        """Write a flat per-node array as-is, little-endian, in the order handed over."""
        return compress(np.ascontiguousarray(values).reshape(-1).tobytes(), compression)

    def decode_scalars(
        self, blob: bytes, *, dtype: str, compression: str, count: int | None
    ) -> npt.NDArray[Any]:
        """Read a flat per-node array back at the width the manifest declared for it."""
        width = np.dtype(dtype).itemsize
        blob = decompress(blob, compression, width * int(count or 0))
        values = np.frombuffer(blob, dtype=dtype)
        if count is not None and len(values) != count:
            raise FormatError(
                f"This blob holds {len(values)} values of {dtype} and its row declares {count}. "
                f"At {width} bytes each the two cannot disagree unless the row and the geometry "
                f"belong to different writes."
            )
        return values


__all__ = ["RawCodec"]
