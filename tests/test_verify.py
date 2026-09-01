"""The verifier, on collections that are right and on collections deliberately made wrong.

**The negative tests are the point of this file.** A verifier that has never been seen to fail is
not evidence of anything: every check below is paired with a corruption that must trip it, so a
check that quietly stopped looking at anything would show up here as a test that no longer fails.
"""

from __future__ import annotations

import numpy as np
import pytest

import konnektion
from konnektion.frames import parquet_to_table, table_to_parquet
from konnektion.manifest import CELL_CATALOG_PATH, MANIFEST_NAME
from konnektion.stores import get_bytes, join, put_bytes


def names(report: konnektion.VerifyReport) -> set[str]:
    """The names of the checks that failed."""
    return {check.name for check in report.failures}


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_a_written_collection_verifies(opened):
    """The baseline every negative test below is measured against."""
    report = konnektion.verify(opened, tier="topology")
    assert report.ok, str(report)


def test_a_ladder_verifies_including_the_cross_level_checks(opened_ladder):
    """The multi-level path, where the coarsening claims are real."""
    report = konnektion.verify(opened_ladder, tier="topology")
    assert report.ok, str(report)
    assert opened_ladder.grid.levels > 1, "the fixture must actually build a ladder"


def test_the_cross_level_checks_are_not_vacuous(opened_ladder):
    """They have to compare a real number of things, or they prove nothing."""
    report = konnektion.verify(opened_ladder, tier="topology")
    subset = next(c for c in report.checks if c.name.startswith("a coarse level is a subset"))
    assert int(subset.detail.split()[0]) > 100, subset.detail
    ghosts = next(c for c in report.checks if c.name.startswith("every ghost is a copy"))
    assert int(ghosts.detail.split()[0]) > 100, ghosts.detail


def test_one_level_skips_the_cross_level_checks_rather_than_passing_them(opened):
    """A pass for want of anything to check reads exactly like a real pass. It must not."""
    report = konnektion.verify(opened, tier="topology")
    assert opened.grid.levels == 1
    assert any("one level" in note for note in report.skipped), report.skipped
    assert not any("subset of the finer" in check.name for check in report.checks)


def test_a_lower_tier_says_what_it_did_not_do(opened):
    """A cheap run must not read like a thorough one."""
    report = konnektion.verify(opened, tier="structure")
    assert report.ok
    assert any("blobs" in note for note in report.skipped)
    assert any("topology" in note for note in report.skipped)


def test_an_unknown_tier_is_refused(opened):
    """Naming what exists beats silently checking less than was asked."""
    with pytest.raises(ValueError, match="tier"):
        konnektion.verify(opened, tier="everything")


# --------------------------------------------------------------------------- #
# the negative tests
# --------------------------------------------------------------------------- #


def test_a_missing_part_is_caught(written):
    """Delete a level's geometry; the manifest still names it."""
    del written.objects["p/level0/part-00000.parquet"]
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="structure")
    assert "every file the manifest names is present" in names(report)


def test_a_prefix_without_a_manifest_is_an_unfinished_write(written):
    """The manifest is written last, so its absence is a specific, expected state."""
    del written.objects[f"p/{MANIFEST_NAME}"]
    with pytest.raises(konnektion.UnfinishedCollectionError):
        konnektion.open_collection(written, "p")


def test_a_cell_without_a_locator_is_caught(written):
    """Blank the part column: the catalog then names a cell no reader could fetch."""
    table = parquet_to_table(get_bytes(written, join("p", CELL_CATALOG_PATH)))
    import pyarrow as pa

    blanked = table.set_column(
        table.schema.get_field_index("part"),
        table.schema.field("part"),
        pa.array([None] * table.num_rows, type=pa.int32()),
    )
    put_bytes(written, join("p", CELL_CATALOG_PATH), table_to_parquet(blanked))
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="structure")
    assert "every cell names the part and row group holding it" in names(report)


def test_a_moved_ghost_is_caught(written):
    """Re-encode one ghost against the wrong cell -- the failure that is otherwise invisible.

    A ghost quantized against the wrong box decodes to a plausible position, and the only
    symptom is an edge reaching slightly the wrong way. Nothing but this check sees it.
    """
    _corrupt_ghost_positions(written)
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="topology")
    assert "every ghost is a copy of a node some cell owns" in names(report), str(report)


def test_a_severed_edge_breaks_connectivity_and_is_caught(written):
    """Sever one cell's edges: the object it held falls into pieces.

    The headline failure. Every piece still draws, and every piece looks like data -- which is
    why counting components per object is a check and not a nicety.
    """
    _sever_the_busiest_cell(written)
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="topology")
    assert names(report) == {"every object is in the number of connected pieces it declares"}, str(report)


def test_an_out_of_range_edge_is_caught(written):
    """An edge naming a node the cell does not hold."""
    _corrupt_edges_out_of_range(written)
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="blobs")
    assert "every edge indexes a node the cell holds" in names(report), str(report)


def test_a_count_that_disagrees_with_its_blob_is_caught(written):
    """The row and the geometry then belong to different writes."""
    import pyarrow as pa

    path = "p/level0/part-00000.parquet"
    table = parquet_to_table(written.objects[path])
    counts = table.column("node_count").to_pylist()
    counts[0] = counts[0] + 3
    bad = table.set_column(
        table.schema.get_field_index("node_count"),
        table.schema.field("node_count"),
        pa.array(counts, type=pa.int32()),
    )
    written.objects[path] = table_to_parquet(bad)
    # The blob decoder refuses a length that cannot hold the declared count; the verifier turns
    # that refusal into a failed check rather than an exception, so the rest of the report --
    # the column scan that may explain it -- still reaches the caller.
    report = konnektion.verify(konnektion.open_collection(written, "p"), tier="blobs")
    assert "every cell decodes" in names(report), str(report)
    failed = {check.name: check for check in report.failures}["every cell decodes"]
    assert any("declares" in example for example in failed.examples), str(report)


# --------------------------------------------------------------------------- #
# helpers that make a collection wrong in one specific way
# --------------------------------------------------------------------------- #


def _rewrite_geometry(store, mutate) -> None:  # type: ignore[no-untyped-def]
    """Read the level-0 part, let ``mutate`` change its columns, write it back."""
    import pyarrow as pa

    path = "p/level0/part-00000.parquet"
    table = parquet_to_table(store.objects[path])
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    mutate(columns)
    store.objects[path] = table_to_parquet(
        pa.table(columns, schema=table.schema)
    )


def _corrupt_ghost_positions(store) -> None:  # type: ignore[no-untyped-def]
    """Shift every ghost blob by one quantum in x, as a wrong-box encoding would."""
    def mutate(columns):  # type: ignore[no-untyped-def]
        for row, blob in enumerate(columns["ghost_positions"]):
            if not blob:
                continue
            values = np.frombuffer(blob, dtype="<u2").copy().reshape(-1, 3)
            values[:, 0] = np.clip(values[:, 0].astype(np.int64) + 20000, 0, 65535)
            columns["ghost_positions"][row] = values.tobytes()
            return
    _rewrite_geometry(store, mutate)


def _corrupt_edges_out_of_range(store) -> None:  # type: ignore[no-untyped-def]
    """Point one edge endpoint past the end of its cell's node array."""
    def mutate(columns):  # type: ignore[no-untyped-def]
        for row, blob in enumerate(columns["edges"]):
            if not blob:
                continue
            values = np.frombuffer(blob, dtype="<u4").copy()
            values[0] = 10_000
            columns["edges"][row] = values.tobytes()
            return
    _rewrite_geometry(store, mutate)


def _sever_the_busiest_cell(store) -> None:  # type: ignore[no-untyped-def]
    """Replace every edge of the busiest cell with a copy of its first one.

    Deliberately count-preserving. The obvious corruption -- emptying the edge blob -- is caught
    one tier earlier by "decoded counts match the catalog", which masks the check this test is
    actually for. Duplicating an edge keeps the count, keeps every index in range, and still
    disconnects everything those edges were holding together, so the only thing left that can
    notice is the connectivity check.
    """
    def mutate(columns):  # type: ignore[no-untyped-def]
        target = int(np.argmax(columns["edge_count"]))
        values = np.frombuffer(columns["edges"][target], dtype="<u4").copy().reshape(-1, 2)
        if len(values) < 2:
            return
        values[1:] = values[0]
        columns["edges"][target] = values.tobytes()
    _rewrite_geometry(store, mutate)
