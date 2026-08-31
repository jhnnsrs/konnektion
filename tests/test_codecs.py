"""The wire format, encoded and decoded. Every claim in `konnektion.codecs` is executable."""

from __future__ import annotations

import numpy as np
import pytest

from konnektion import codecs
from konnektion.codecs.blobs import QUANT_MAX
from konnektion.errors import FormatError, PartitioningError
from konnektion.manifest import (
    COMPRESSION_NONE,
    COMPRESSION_ZSTD,
    NODE_IDS_UINT32,
    NODE_IDS_UINT64,
    RADII_FLOAT32,
    RADII_NONE,
    RADII_UINT16_QUANTIZED_PER_CELL,
)
from konnektion.octree import cell_box, morton_encode_one

CELL_SIZE = (64, 64, 32)


@pytest.mark.parametrize("compression", [COMPRESSION_NONE, COMPRESSION_ZSTD])
def test_positions_round_trip_within_half_a_quantum(compression):
    """The bound is the quantization step, and it is per axis of the cell's own box."""
    cell = morton_encode_one((2, 1, 3))
    origin, extent = cell_box(cell, 0, CELL_SIZE)
    rng = np.random.default_rng(0)
    points = origin + rng.random((50, 3)) * extent

    blob = codecs.encode_positions(
        points, cell=cell, level=0, cell_size=CELL_SIZE, compression=compression
    )
    back = codecs.decode_positions(
        blob, cell=cell, level=0, cell_size=CELL_SIZE, compression=compression, node_count=50
    )
    assert np.all(np.abs(back - points) <= extent / QUANT_MAX / 2 + 1e-9)


def test_positions_are_exactly_six_bytes_a_node_uncompressed():
    """The blob is a vertex buffer; a consumer uploads it without a decoder."""
    blob = codecs.encode_positions(np.zeros((7, 3)), cell=0, level=0, cell_size=CELL_SIZE)
    assert len(blob) == 6 * 7


def test_a_node_outside_its_cell_is_refused_not_clamped():
    """Clamping is what makes a partitioning bug invisible."""
    with pytest.raises(PartitioningError, match="partitioning bug"):
        codecs.encode_positions(
            np.array([[1000.0, 0.0, 0.0]]), cell=0, level=0, cell_size=CELL_SIZE
        )


def test_edges_are_pairs_not_triples():
    """Eight bytes an edge. The arity is the whole difference from a mesh index buffer."""
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    blob = codecs.encode_edges(edges)
    assert len(blob) == 8 * 3
    assert codecs.decode_edges(blob, edge_count=3).tolist() == edges.tolist()


def test_an_edge_blob_read_at_the_wrong_arity_would_not_error():
    """Why `encoding.edges` is a required key: the failure is silent, so it must be declared.

    Six edges is twelve integers, which divides by three as cleanly as by two. Nothing about the
    bytes says which is right.
    """
    edges = np.arange(12).reshape(6, 2)
    blob = codecs.encode_edges(edges)
    as_triples = np.frombuffer(blob, dtype="<u4").reshape(-1, 3)
    assert as_triples.shape == (4, 3), "it reshapes cleanly, which is exactly the danger"


def test_edge_order_survives_the_round_trip():
    """`object_edge_offsets` names ranges, so a codec that reordered would break slicing."""
    edges = np.array([[5, 4], [0, 1], [3, 2]])
    assert codecs.decode_edges(codecs.encode_edges(edges), edge_count=3).tolist() == edges.tolist()


@pytest.mark.parametrize("declaration", [NODE_IDS_UINT64, NODE_IDS_UINT32])
def test_node_ids_round_trip(declaration):
    """A node keeps its identity across cells and levels, at either declared width."""
    ids = np.array([0, 7, 4294967295 if declaration == NODE_IDS_UINT32 else 2**40])
    blob = codecs.encode_node_ids(ids, declaration=declaration)
    assert codecs.decode_node_ids(blob, declaration=declaration, node_count=3).tolist() == ids.tolist()


def test_an_id_too_large_for_its_declaration_is_refused():
    """Truncating an id silently would make two nodes answer to the same name."""
    with pytest.raises(FormatError, match="nodeIds"):
        codecs.encode_node_ids(np.array([2**40]), declaration=NODE_IDS_UINT32)


def test_ghost_cells_round_trip():
    """Each ghost names the cell that owns it, and that name has to survive."""
    owners = np.array([morton_encode_one((1, 0, 0)), morton_encode_one((0, 2, 1))])
    blob = codecs.encode_ghost_cells(owners)
    assert codecs.decode_ghost_cells(blob, ghost_count=2).tolist() == owners.tolist()


def test_a_ghost_is_quantized_against_the_cell_that_owns_it():
    """The point of ghosts: a position this cell's box cannot represent, stored exactly anyway."""
    owner = morton_encode_one((3, 0, 0))
    origin, extent = cell_box(owner, 0, CELL_SIZE)
    point = origin + extent * 0.5

    # The holding cell is (0, 0, 0), which cannot represent this point at all.
    with pytest.raises(PartitioningError):
        codecs.encode_positions(point[None], cell=0, level=0, cell_size=CELL_SIZE)

    blob = codecs.encode_ghost_positions(
        point[None], np.array([owner]), level=0, cell_size=CELL_SIZE
    )
    back = codecs.decode_ghost_positions(
        blob, np.array([owner]), level=0, cell_size=CELL_SIZE, ghost_count=1
    )
    assert np.allclose(back[0], point, atol=float(max(extent)) / QUANT_MAX)


def test_a_ghost_naming_the_wrong_owner_is_refused():
    """The position then falls outside the box it claims, which is a partitioning bug."""
    far = np.array([[500.0, 0.0, 0.0]])
    with pytest.raises(PartitioningError, match="wrong owner"):
        codecs.encode_ghost_positions(far, np.array([0]), level=0, cell_size=CELL_SIZE)


@pytest.mark.parametrize("declaration", [RADII_FLOAT32, RADII_UINT16_QUANTIZED_PER_CELL])
def test_radii_round_trip(declaration):
    """Both encodings land within a quantum of what went in."""
    radii = np.array([0.0, 0.5, 3.25, 31.0])
    blob = codecs.encode_radii(
        radii, declaration=declaration, cell=0, level=0, cell_size=CELL_SIZE
    )
    back = codecs.decode_radii(
        blob, declaration=declaration, cell=0, level=0, cell_size=CELL_SIZE, node_count=4
    )
    assert np.allclose(back, radii, atol=64.0 / QUANT_MAX)


def test_a_collection_without_radii_cannot_write_them():
    """Writing a column the manifest says is absent would leave a reader zeros to believe."""
    with pytest.raises(FormatError, match="NONE"):
        codecs.encode_radii(
            np.zeros(1), declaration=RADII_NONE, cell=0, level=0, cell_size=CELL_SIZE
        )


def test_a_compressed_blob_needs_its_count_to_decode():
    """ZSTD framing carries no content size, so the row is the only statement of the length."""
    blob = codecs.encode_positions(
        np.zeros((3, 3)), cell=0, level=0, cell_size=CELL_SIZE, compression=COMPRESSION_ZSTD
    )
    with pytest.raises(FormatError, match="node_count"):
        codecs.decode_positions(
            blob, cell=0, level=0, cell_size=CELL_SIZE, compression=COMPRESSION_ZSTD
        )


def test_a_count_disagreeing_with_its_blob_is_refused():
    """The row and the geometry would then belong to different writes."""
    blob = codecs.encode_positions(np.zeros((3, 3)), cell=0, level=0, cell_size=CELL_SIZE)
    with pytest.raises(FormatError, match="declares"):
        codecs.decode_positions(blob, cell=0, level=0, cell_size=CELL_SIZE, node_count=5)


def test_an_unknown_codec_is_refused_naming_what_exists():
    """Guessing a codec produces geometry that decodes to garbage, not an error."""
    with pytest.raises(FormatError, match="the format defines"):
        codecs.codec_for("MESHOPT")
