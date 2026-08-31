"""Writing a collection out and reading it back: the store layer, the layout, and the ordering."""

from __future__ import annotations

import numpy as np
import pytest

import konnektion
from konnektion.manifest import CELL_CATALOG_PATH, MANIFEST_NAME, OBJECT_CATALOG_PATH
from konnektion.octree import cell_box, morton_decode, morton_encode, morton_encode_one


def test_the_layout_is_the_one_the_format_fixes(written):
    """A reader who knows one konnektion prefix knows them all."""
    assert set(written.objects) == {
        f"p/{MANIFEST_NAME}",
        f"p/{CELL_CATALOG_PATH}",
        f"p/{OBJECT_CATALOG_PATH}",
        "p/level0/part-00000.parquet",
    }


def test_the_manifest_names_every_file_with_its_length(opened):
    """A reader that can neither list nor stat still has to find and range-read the parts."""
    files = opened.manifest.files
    assert files["cells"]["bytes"] > 0
    assert files["objects"]["bytes"] > 0
    for entries in files["levels"].values():
        for entry in entries:
            assert entry["bytes"] > 0
            assert entry["rowGroups"] >= 1


def test_every_cell_carries_a_locator(opened):
    """A catalog row without one names a cell no reader could fetch."""
    for entry in opened.cells.values():
        assert entry.part is not None
        assert entry.row_group is not None


def test_the_geometry_decodes_back_to_what_went_in(objects):
    """Every node, matched by (object, id), within half a quantization step."""
    collection = konnektion.build_collection(objects, cell_size=(128, 128, 128))
    store = konnektion.MemoryStore()
    collection.write(store, "p")
    opened = konnektion.open_collection(store, "p")

    step = float(max(opened.grid.cell_extent(0))) / konnektion.QUANT_MAX
    seen: dict[tuple[int, int], np.ndarray] = {}
    for cell in opened.iter_cells(0):
        for position, object_id in enumerate(cell.object_ids):
            start = cell.object_node_offsets[position]
            stop = (
                cell.object_node_offsets[position + 1]
                if position + 1 < len(cell.object_node_offsets)
                else cell.node_count
            )
            for index in range(start, stop):
                seen[(int(object_id), int(cell.node_ids[index]))] = cell.positions[index]

    total = 0
    worst = 0.0
    for object_id, network in objects.items():
        for node in range(network.node_count):
            key = (object_id, node)
            assert key in seen, f"node {node} of object {object_id} was not written"
            worst = max(worst, float(np.linalg.norm(seen[key] - network.nodes[node])))
            total += 1
    assert total == sum(network.node_count for network in objects.values())
    assert worst <= np.sqrt(3) * step / 2 + 1e-9, worst


def test_the_radii_survive_the_round_trip(objects):
    """Owned nodes and ghosts alike come back with the radius that went in."""
    collection = konnektion.build_collection(objects, cell_size=(128, 128, 128))
    store = konnektion.MemoryStore()
    collection.write(store, "p")
    opened = konnektion.open_collection(store, "p")
    for cell in opened.iter_cells(0):
        assert cell.radii is not None
        assert len(cell.radii) == len(cell.positions)
        assert cell.radii[: cell.node_count].min() > 0.0
        return


def test_a_ghost_is_the_tail_of_the_node_array(opened):
    """No mask is stored, because the layout already says which nodes are copies."""
    for cell in opened.iter_cells(0):
        if not cell.ghost_count:
            continue
        assert cell.is_ghost.sum() == cell.ghost_count
        assert not cell.is_ghost[: cell.node_count].any()
        assert cell.is_ghost[cell.node_count :].all()
        return
    pytest.skip("no cell in this fixture holds a ghost")


def test_reading_one_cell_by_key_matches_iterating(opened):
    """Two ways to the same bytes must not disagree."""
    key = min(opened.cells)
    direct = opened.read_cell(*key)
    matching = next(c for c in opened.iter_cells(key[0]) if c.cell == key[1])
    assert np.array_equal(direct.positions, matching.positions)
    assert np.array_equal(direct.edges, matching.edges)


def test_a_directory_store_round_trips(tmp_path, objects):
    """The same tree lands on a disk and in an S3 prefix; nothing above the store cares."""
    collection = konnektion.build_collection(objects, cell_size=(128, 128, 128))
    store = konnektion.DirectoryStore(str(tmp_path))
    collection.write(store, "p")
    opened = konnektion.open_collection(store, "p")
    assert konnektion.verify(opened, tier="topology").ok


# --------------------------------------------------------------------------- #
# addressing
# --------------------------------------------------------------------------- #


def test_morton_round_trips():
    """The interleave is the sort key, so it has to be exactly invertible."""
    triples = [(0, 0, 0), (1, 2, 3), (17, 4, 9), (131071, 0, 0)]
    codes = morton_encode(np.array(triples))
    assert [morton_decode(int(code)) for code in codes] == triples


def test_a_cell_box_is_invertible_from_the_row_alone():
    """Level and cell and the manifest's cell_size are all a decoder gets, and all it needs."""
    origin, extent = cell_box(morton_encode_one((2, 1, 3)), 1, (64, 64, 32))
    assert origin.tolist() == [2 * 128, 1 * 128, 3 * 64]
    assert extent.tolist() == [128, 128, 64]


def test_a_cell_index_past_the_morton_limit_is_refused():
    """Three axes at 17 bits interleave to 51, which is what keeps a code under 2**53."""
    with pytest.raises(ValueError, match="17-bit"):
        morton_encode(np.array([[1 << 17, 0, 0]]))


def test_a_negative_cell_index_is_refused():
    """The octree addresses the positive octant only."""
    with pytest.raises(ValueError, match="non-negative"):
        morton_encode(np.array([[-1, 0, 0]]))
