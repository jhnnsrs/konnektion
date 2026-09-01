"""Per-node attributes: computed once on the full graph, declared, and only ever subset.

The properties under test are the ones a renderer will lean on without being able to check:
that a value decoded at any level is the value computed at level 0, that a ghost carries its
owner's value, and that the manifest's declarations are exactly what the geometry holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import konnektion
from konnektion.frames import attribute_column, ghost_attribute_column, parquet_to_table
from konnektion.manifest import Attribute, Manifest
from konnektion.sources import Network
from konnektion.verify import _members

from .conftest import arbor, chain

INTRINSIC_NAMES = [name for name, _ in konnektion.INTRINSIC_ATTRIBUTES]


def value_map(cells, name: str) -> dict[tuple[int, int], float]:
    """``(object_id, node_id)`` -> attribute value, owned nodes only."""
    found: dict[tuple[int, int], float] = {}
    for cell in cells:
        for object_id, node_id, index, is_ghost in _members(cell):
            if not is_ghost:
                found[(object_id, node_id)] = float(cell.attributes[name][index])
    return found


# --------------------------------------------------------------------------- #
# what a build computes
# --------------------------------------------------------------------------- #


def test_every_build_declares_the_intrinsics(collection: konnektion.NetworkCollection):
    """The four computed metrics are declared, in order, wearing their semantics."""
    declared = collection.manifest().attributes
    assert [attribute.name for attribute in declared] == INTRINSIC_NAMES
    assert all(attribute.semantics is not None for attribute in declared)
    assert all(attribute.encoding == "FLOAT32" for attribute in declared)


def test_intrinsic_values_on_a_known_tree():
    """A Y-shaped tree whose metrics can be checked by hand."""
    #   2   3
    #    \ /
    #     1
    #     |
    #     0  (root)
    tree = Network(
        nodes=np.array([[5, 5, 0], [5, 5, 1], [4, 5, 2], [6, 5, 2]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64),
        root=0,
    )
    built = konnektion.build_collection({7: tree}, cell_size=(16, 16, 16))
    store = konnektion.MemoryStore()
    built.write(store, "p")
    opened = konnektion.open_collection(store, "p")
    cells = list(opened.iter_cells(0))
    by_id = {
        name: {node_id: value for (_, node_id), value in value_map(cells, name).items()}
        for name in INTRINSIC_NAMES
    }
    assert by_id["degree"] == {0: 1.0, 1: 3.0, 2: 1.0, 3: 1.0}
    assert by_id["depth"] == {0: 0.0, 1: 1.0, 2: 2.0, 3: 2.0}
    assert by_id["strahler"] == {0: 2.0, 1: 2.0, 2: 1.0, 3: 1.0}
    assert by_id["component"] == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}


def test_rootless_strahler_and_depth_are_nan():
    """An object with no distinguished root has no 'up', and the format says so with NaN."""
    built = konnektion.build_collection({1: chain(5)}, cell_size=(64, 64, 64))
    store = konnektion.MemoryStore()
    built.write(store, "p")
    opened = konnektion.open_collection(store, "p")
    cells = list(opened.iter_cells(0))
    for name in ("strahler", "depth"):
        assert all(np.isnan(value) for value in value_map(cells, name).values()), name
    degrees = value_map(cells, "degree")
    assert set(degrees.values()) == {1.0, 2.0}


# --------------------------------------------------------------------------- #
# caller-supplied attributes
# --------------------------------------------------------------------------- #


def test_user_attributes_round_trip():
    """A caller's per-node column comes back at every node, declared with null semantics."""
    tree = arbor(depth=3)
    tortuosity = np.linspace(0.0, 2.0, tree.node_count)
    network = Network(
        nodes=tree.nodes, edges=tree.edges, radii=tree.radii, root=tree.root,
        attributes={"tortuosity": tortuosity},
    )
    built = konnektion.build_collection({4: network}, cell_size=(128, 128, 128))
    declared = {attribute.name: attribute for attribute in built.manifest().attributes}
    assert declared["tortuosity"].semantics is None

    store = konnektion.MemoryStore()
    built.write(store, "p")
    opened = konnektion.open_collection(store, "p")
    found = value_map(list(opened.iter_cells(0)), "tortuosity")
    expected = {int(node_id): tortuosity[index] for index, node_id in enumerate(network.ids())}
    assert len(found) == tree.node_count
    for (_, node_id), value in found.items():
        assert value == pytest.approx(expected[node_id], rel=1e-6)


def test_a_missing_user_attribute_is_nan_filled():
    """An attribute one object lacks is NaN there, not refused: partial measurement is normal."""
    with_it = arbor(depth=3)
    without_it = arbor(depth=3, origin=(700.0, 500.0, 400.0), seed=9)
    built = konnektion.build_collection(
        {
            1: Network(
                nodes=with_it.nodes, edges=with_it.edges, root=with_it.root,
                attributes={"tortuosity": np.ones(with_it.node_count)},
            ),
            2: without_it,
        },
        cell_size=(512, 512, 512),
    )
    store = konnektion.MemoryStore()
    built.write(store, "p")
    opened = konnektion.open_collection(store, "p")
    found = value_map(list(opened.iter_cells(0)), "tortuosity")
    assert all(value == 1.0 for (obj, _), value in found.items() if obj == 1)
    assert all(np.isnan(value) for (obj, _), value in found.items() if obj == 2)


def test_an_intrinsic_name_collision_is_refused():
    """A caller column named like a computed metric would forge the manifest's semantics."""
    tree = chain(4)
    network = Network(
        nodes=tree.nodes, edges=tree.edges,
        attributes={"degree": np.zeros(tree.node_count)},
    )
    with pytest.raises(konnektion.FormatError, match="collide"):
        konnektion.build_collection({1: network}, cell_size=(64, 64, 64))


@pytest.mark.parametrize(
    ("attributes", "match"),
    [
        ({"Bad-Name": np.zeros(4)}, "lowercase"),
        ({"radius": np.zeros(4)}, "radii"),
        ({"short": np.zeros(3)}, "one value per node"),
        ({"hot": np.array([1.0, np.inf, 0.0, 0.0])}, "infinity"),
    ],
)
def test_malformed_attributes_are_refused(attributes, match):
    """Each refusal names the attribute and the rule it broke."""
    tree = chain(4)
    with pytest.raises(konnektion.FormatError, match=match):
        konnektion.coerce_network(
            Network(nodes=tree.nodes, edges=tree.edges, attributes=attributes)
        )


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def test_manifest_attributes_round_trip(collection: konnektion.NetworkCollection):
    """Declarations survive to_dict/from_dict, and the key is written even when empty."""
    manifest = collection.manifest()
    raw = manifest.to_dict()
    assert [entry["name"] for entry in raw["attributes"]] == INTRINSIC_NAMES
    again = Manifest.from_dict(raw)
    assert again.attributes == manifest.attributes

    bare = dict(raw)
    bare["attributes"] = []
    assert Manifest.from_dict(bare).attributes == ()


def test_a_manifest_without_the_attributes_key_still_reads(collection):
    """A spec-1 manifest written before attributes existed parses to none, not an error."""
    raw = collection.manifest().to_dict()
    del raw["attributes"]
    assert Manifest.from_dict(raw).attributes == ()


def test_a_duplicate_declaration_is_refused(collection):
    """One name, one column: a repeated declaration is two claims about the same bytes."""
    raw = collection.manifest().to_dict()
    raw["attributes"] = raw["attributes"] + [raw["attributes"][0]]
    with pytest.raises(konnektion.FormatError, match="once"):
        Manifest.from_dict(raw)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"name": "x", "encoding": "UINT8"}, "encoding"),
        ({"name": "x", "semantics": "SHOE_SIZE"}, "semantics"),
        ({"name": "radius"}, "radii"),
    ],
)
def test_a_declaration_outside_the_vocabulary_is_refused(entry, match):
    """The vocabulary refusals, at the dataclass gate."""
    with pytest.raises(konnektion.FormatError, match=match):
        Attribute.from_any(entry)


# --------------------------------------------------------------------------- #
# ghosts and levels
# --------------------------------------------------------------------------- #


def test_ghost_values_are_the_owners_values(opened: konnektion.Collection):
    """A ghost carries the same attribute values as the node it copies."""
    cells = list(opened.iter_cells(0))
    compared = 0
    for name in INTRINSIC_NAMES:
        owned = value_map(cells, name)
        for cell in cells:
            assert len(cell.attributes[name]) == len(cell.positions)
            for object_id, node_id, index, is_ghost in _members(cell):
                if not is_ghost:
                    continue
                compared += 1
                ghost_value = float(cell.attributes[name][index])
                owner_value = owned[(object_id, node_id)]
                assert ghost_value == owner_value or (
                    np.isnan(ghost_value) and np.isnan(owner_value)
                )
    assert compared, "the fixture grew no ghosts, so this test checked nothing"


def test_coarse_levels_carry_level_zero_values(opened_ladder: konnektion.Collection):
    """A node kept at a coarse level keeps its level-0 value: subset, never recomputed."""
    assert opened_ladder.grid.levels > 1, "the ladder fixture must be deep enough to test this"
    for name in INTRINSIC_NAMES:
        baseline = value_map(list(opened_ladder.iter_cells(0)), name)
        for level in range(1, opened_ladder.grid.levels):
            coarse = value_map(list(opened_ladder.iter_cells(level)), name)
            assert coarse, f"level {level} decoded no {name} values"
            for key, value in coarse.items():
                original = baseline[key]
                assert value == original or (np.isnan(value) and np.isnan(original)), (
                    f"{name} at {key} is {value} at level {level} but {original} at level 0"
                )


# --------------------------------------------------------------------------- #
# the verifier
# --------------------------------------------------------------------------- #


def test_verify_passes_a_clean_collection(opened_ladder: konnektion.Collection):
    """The attribute checks run -- not skipped -- and pass on an honest build."""
    report = konnektion.verify(opened_ladder, tier="topology")
    assert report.ok, str(report)
    names = {check.name for check in report.checks}
    assert "attribute columns are exactly what the manifest declares" in names
    assert "every declared attribute decodes to one value per node" in names
    assert "a kept node's attribute values are the level-0 values" in names


def test_an_undeclared_attribute_column_is_caught(written: konnektion.MemoryStore):
    """Values no reader will ever look for: the manifest names what exists."""
    import pyarrow as pa

    path = "p/level0/part-00000.parquet"
    table = parquet_to_table(written.objects[path])
    stray = pa.array([b""] * table.num_rows, type=pa.large_binary())
    written.objects[path] = _reserialize(
        table.append_column(pa.field("attr_stray", pa.large_binary(), nullable=True), stray)
    )
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="blobs")
    failed = {check.name for check in report.failures}
    assert "attribute columns are exactly what the manifest declares" in failed, str(report)


def test_a_recomputed_metric_is_caught(opened_ladder: konnektion.Collection):
    """A coarse level whose values differ from level 0 fails the coarsening check."""
    import pyarrow as pa

    store = konnektion.MemoryStore()
    # Rewrite the ladder with one coarse cell's degree blob replaced wholesale.
    for path, body in opened_ladder.store.objects.items():
        store.objects[path.replace("p/", "q/", 1)] = body
    path = next(p for p in sorted(store.objects) if "level1/part" in p)
    table = parquet_to_table(store.objects[path])
    column = table.column(attribute_column("degree")).to_pylist()
    counts = table.column("node_count").to_pylist()
    column[0] = np.full(counts[0], 99.0, dtype="<f4").tobytes()
    index = table.schema.get_field_index(attribute_column("degree"))
    tampered = table.set_column(
        index, table.schema.field(attribute_column("degree")),
        pa.array(column, type=pa.large_binary()),
    )
    store.objects[path] = _reserialize(tampered)
    report = konnektion.verify(konnektion.open_collection(store, "q"), tier="topology")
    failed = {check.name for check in report.failures}
    assert "a kept node's attribute values are the level-0 values" in failed, str(report)


def _reserialize(table) -> bytes:
    """Parquet bytes for a tampered table, compression matching the writer's."""
    from konnektion.frames import table_to_parquet

    return table_to_parquet(table)
