"""What a collection declares, and every declaration it is not allowed to make."""

from __future__ import annotations

import json

import pytest

from konnektion.errors import FormatError
from konnektion.manifest import (
    PRUNING_NONE,
    PRUNING_STRAHLER,
    SIMPLIFICATION_DOUGLAS_PEUCKER,
    SIMPLIFICATION_NONE,
    SPEC_VERSION,
    Coarsening,
    Encoding,
    Grid,
    Manifest,
)


def a_manifest(**kwargs: object) -> Manifest:
    """A minimal valid manifest."""
    return Manifest(
        grid=Grid(cell_size=(128, 128, 64), levels=3),
        encoding=Encoding(),
        axes=["x", "y", "z"],
        **kwargs,
    )


def test_a_manifest_round_trips_through_json():
    """What is written is what is read back, byte for byte of meaning."""
    manifest = a_manifest(counts={"objects": 2}, files={"cells": {"path": "c.parquet"}})
    assert Manifest.from_json(manifest.to_json()) == manifest


def test_the_encoding_is_written_complete():
    """Never sparse: a reader configures its decoder from what it reads back."""
    written = a_manifest().to_dict()["encoding"]
    assert set(written) == {
        "positions", "edges", "nodeIds", "radii", "ghosts",
        "codec", "compression", "pruning", "simplification",
    }


def test_an_encoding_missing_a_key_is_refused():
    """A decoder cannot infer these, and a wrong guess is not an error but garbage."""
    with pytest.raises(FormatError, match="omits"):
        Encoding.from_dict({"positions": "UINT16_QUANTIZED_PER_CELL"})


def test_an_unknown_spec_version_is_refused():
    """The version is the statement that the rest of the file means what this reader thinks."""
    raw = a_manifest().to_dict()
    raw["specVersion"] = "99"
    with pytest.raises(FormatError, match="specVersion"):
        Manifest.from_dict(raw)


def test_an_unknown_top_level_key_is_ignored():
    """A later version may add one; this reader should not choke on it."""
    raw = a_manifest().to_dict()
    raw["somethingLater"] = {"a": 1}
    assert Manifest.from_dict(raw).spec_version == SPEC_VERSION


def test_a_value_outside_the_vocabulary_is_refused():
    """A vocabulary that accepted anything would declare nothing."""
    with pytest.raises(FormatError, match="edges"):
        Encoding(edges="UINT32_TRIPLES")


def test_shape_and_axes_are_written_even_when_null():
    """So a reader can tell 'considered and unanswerable' from 'predates the question'."""
    raw = json.loads(Manifest(grid=Grid((64, 64, 64), 1), encoding=Encoding()).to_json())
    assert raw["shape"] is None and raw["axes"] is None
    assert "shape" in raw and "axes" in raw


def test_a_coarsening_that_prunes_nothing_must_say_so():
    """The declaration is what a reader is told happened, so it is checked, not trusted."""
    with pytest.raises(FormatError, match="strahler_step"):
        Coarsening(strahler_step=0, epsilon=1.0, pruning=PRUNING_STRAHLER)


def test_a_coarsening_that_moves_nothing_must_say_so():
    """Same rule, the other operation: an epsilon of zero straightens nothing."""
    with pytest.raises(FormatError, match="epsilon"):
        Coarsening(strahler_step=1, epsilon=0.0, simplification=SIMPLIFICATION_DOUGLAS_PEUCKER)


def test_the_empty_coarsening_declares_none_of_both():
    """A schedule that does nothing has to be able to say so."""
    schedule = Coarsening.none()
    assert schedule.pruning == PRUNING_NONE
    assert schedule.simplification == SIMPLIFICATION_NONE
    assert not schedule.coarsens


def test_coarsening_never_moves_a_node():
    """The fact the error bound rests on: both operations remove, neither repositions."""
    assert not Coarsening().moves_nodes()


def test_the_default_coarsening_is_self_consistent():
    """A bare Coarsening() has to be constructible -- it is the default the builder uses."""
    schedule = Coarsening()
    assert schedule.coarsens
    assert schedule.strahler_threshold(0) == 1
    assert schedule.strahler_threshold(2) == 3
    assert schedule.epsilon_at(2) == schedule.epsilon * 4


def test_a_floor_below_one_segment_is_refused():
    """The smallest drawable graph is one segment, so that is the floor."""
    with pytest.raises(FormatError, match="floor_nodes"):
        Coarsening(floor_nodes=1)


def test_a_grid_needs_three_positive_sizes():
    """A grid that cannot address a graph is refused where it is built."""
    with pytest.raises(FormatError, match="cell_size"):
        Grid(cell_size=(128, 0, 64), levels=1)
    with pytest.raises(FormatError, match="at least one level"):
        Grid(cell_size=(128, 128, 64), levels=0)


def test_one_level_is_a_legal_grid():
    """The common case for traced data, not a degenerate one."""
    assert Grid(cell_size=(64, 64, 64), levels=1).levels == 1
