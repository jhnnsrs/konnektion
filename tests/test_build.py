"""Building levels: adaptive depth, ghosts, edge ownership, and what a level declares."""

from __future__ import annotations

import numpy as np
import pytest

import konnektion
from konnektion.errors import FormatError
from konnektion.manifest import (
    PRUNING_NONE,
    PRUNING_STRAHLER,
    RADII_FLOAT32,
    RADII_NONE,
    SIMPLIFICATION_DOUGLAS_PEUCKER,
    SIMPLIFICATION_NONE,
    Coarsening,
)
from konnektion.sources import Network
from tests.conftest import arbor, chain


def test_small_data_gets_one_level_and_says_it_coarsened_nothing():
    """If it does not need downscaling, it is not downscaled -- and the manifest is honest."""
    collection = konnektion.build_collection({1: arbor(depth=4)}, cell_size=(128, 128, 128))
    assert collection.grid.levels == 1
    assert collection.encoding.pruning == PRUNING_NONE
    assert collection.encoding.simplification == SIMPLIFICATION_NONE
    assert any("one level" in note for note in collection.notes)


def test_one_level_keeps_every_node_exactly(collection, objects):
    """Level 0 is the input, untouched -- a collection always carries the data it was given."""
    stored = sum(collection.cell_catalog.column("node_count").to_pylist())
    assert stored == sum(network.node_count for network in objects.values())


def test_large_data_grows_a_ladder(ladder):
    """Past the overview budget the ladder grows, and says which operations it ran."""
    assert ladder.grid.levels > 1
    assert ladder.encoding.pruning == PRUNING_STRAHLER
    assert ladder.encoding.simplification == SIMPLIFICATION_DOUGLAS_PEUCKER


def test_each_level_is_smaller_than_the_one_below(ladder):
    """A coarse level that is not smaller costs a fetch and saves nothing."""
    counts = {
        level: sum(table.column("node_count").to_pylist()) for level, table in ladder.shards
    }
    for level in range(1, ladder.grid.levels):
        assert counts[level] < counts[level - 1], counts


def test_an_explicit_level_count_is_honoured():
    """Adaptive depth is the default, not a policy imposed on a caller who knows better."""
    collection = konnektion.build_collection(
        {1: arbor(depth=7)}, cell_size=(128, 128, 128), levels=3
    )
    assert collection.grid.levels == 3


def test_an_edge_has_exactly_one_owning_cell(collection, objects):
    """Not two: a crossing edge stored twice draws a brighter line along every cell plane."""
    stored = sum(collection.cell_catalog.column("edge_count").to_pylist())
    assert stored == sum(network.edge_count for network in objects.values())


def test_a_crossing_edge_produces_a_ghost(collection):
    """An object spanning cells has to reach across, and reaching across is what a ghost is."""
    assert sum(collection.cell_catalog.column("ghost_count").to_pylist()) > 0


def test_a_graph_inside_one_cell_needs_no_ghosts():
    """The degenerate case is genuinely free -- no crossings, so no copies."""
    collection = konnektion.build_collection({1: chain(5)}, cell_size=(1024, 1024, 1024))
    assert sum(collection.cell_catalog.column("ghost_count").to_pylist()) == 0


def test_radii_are_carried_when_the_objects_have_them(collection):
    """A radius is a measurement, so it is stored when there is one."""
    assert collection.encoding.radii == RADII_FLOAT32


def test_radii_are_absent_when_the_objects_have_none():
    """A column that would hold nothing is not declared at all."""
    collection = konnektion.build_collection({1: chain(5)}, cell_size=(256, 256, 256))
    assert collection.encoding.radii == RADII_NONE


def test_declaring_radii_that_do_not_exist_is_refused():
    """A declared column holding zeros is a place for a reader to find data that is not there."""
    with pytest.raises(FormatError, match="no object carries one"):
        konnektion.build_collection(
            {1: chain(5)}, cell_size=(256, 256, 256), radii=RADII_FLOAT32
        )


def test_a_negative_coordinate_is_refused():
    """The octree addresses the positive octant; shifting is the caller's to do knowingly."""
    network = Network(nodes=np.array([[-1.0, 0, 0], [1.0, 0, 0]]), edges=np.array([[0, 1]]))
    with pytest.raises(FormatError, match="positive octant"):
        konnektion.build_collection({1: network}, cell_size=(64, 64, 64))


def test_the_object_catalog_declares_the_component_count(collection):
    """Declared so connectivity is checkable at every level, one-level collections included."""
    counts = collection.object_catalog.column("component_count").to_pylist()
    assert counts == [1, 1], "both fixtures are single connected arbors"


def test_a_disconnected_object_declares_its_pieces():
    """A disconnected input is legitimate; the format records the number rather than refusing."""
    nodes = np.array([[10.0, 10, 10], [11, 10, 10], [50, 50, 50], [51, 50, 50]])
    edges = np.array([[0, 1], [2, 3]])
    collection = konnektion.build_collection(
        {1: Network(nodes=nodes, edges=edges)}, cell_size=(256, 256, 256)
    )
    assert collection.object_catalog.column("component_count").to_pylist() == [2]


def test_cell_size_is_taken_when_given_and_chosen_when_not(objects):
    """The chunk shape is worth matching, and nothing about the nodes can reveal it."""
    given = konnektion.build_collection(objects, cell_size=(128, 128, 128))
    assert tuple(given.grid.cell_size) == (128, 128, 128)
    chosen = konnektion.build_collection(objects)
    assert all(size >= 16 for size in chosen.grid.cell_size)


def test_the_bounding_box_covers_the_ghosts_too(collection):
    """A viewer culls against it, and this cell draws edges reaching into its neighbours."""
    import konnektion as k

    store = k.MemoryStore()
    collection.write(store, "p")
    opened = k.open_collection(store, "p")
    for cell in opened.iter_cells(0):
        if not cell.ghost_count:
            continue
        entry = opened.cells[(cell.level, cell.cell)]
        low = np.array(entry.bounds[:3])
        high = np.array(entry.bounds[3:])
        assert np.all(cell.positions >= low - 1e-6)
        assert np.all(cell.positions <= high + 1e-6)
        return
    pytest.skip("no cell in this fixture holds a ghost")


def test_a_custom_coarsening_is_declared_verbatim():
    """Whatever the schedule does, the manifest says which of the two operations ran."""
    schedule = Coarsening(strahler_step=2, epsilon=0.0, simplification=SIMPLIFICATION_NONE)
    collection = konnektion.build_collection(
        {1: arbor(depth=7)}, cell_size=(128, 128, 128), levels=2, coarsening=schedule
    )
    assert collection.encoding.pruning == PRUNING_STRAHLER
    assert collection.encoding.simplification == SIMPLIFICATION_NONE
