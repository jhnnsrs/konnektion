"""Fixtures: a branching arbor, a written collection, and the store it landed in.

The arbor is generated rather than loaded so the suite needs no data file, and it is generated
with several nodes along every branch on purpose -- a tree whose branches are single edges has no
unbranched run to straighten, so Douglas-Peucker would be exercised by nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import konnektion
from konnektion.sources import Network

#: Big enough that the adaptive ladder builds more than one level, so the cross-level checks are
#: not permanently skipped. Kept in one place because several tests assert against its depth.
LADDER_SEEDS = range(1, 9)
CELL_SIZE = (256, 256, 256)


def arbor(
    depth: int = 6,
    spread: float = 30.0,
    origin: tuple[float, float, float] = (300.0, 300.0, 300.0),
    seed: int = 3,
    per_branch: int = 5,
) -> Network:
    """A binary arbor with ``per_branch`` nodes along each branch, rooted at node 0."""
    rng = np.random.default_rng(seed)
    nodes = [np.array(origin, dtype=float)]
    edges: list[list[int]] = []
    radii = [4.0]
    frontier = [(0, np.array([0.0, 0.0, 1.0]))]
    for level in range(depth):
        following: list[tuple[int, np.ndarray]] = []
        for index, direction in frontier:
            for _ in range(2):
                heading = direction + rng.normal(0, 0.45, 3)
                heading /= np.linalg.norm(heading)
                previous = index
                for _ in range(per_branch):
                    point = nodes[previous] + heading * (spread / per_branch)
                    point = point + rng.normal(0, 0.25, 3)
                    nodes.append(point)
                    radii.append(4.0 / (level + 1))
                    edges.append([previous, len(nodes) - 1])
                    previous = len(nodes) - 1
                following.append((previous, heading))
        frontier = following
    return Network(
        nodes=np.asarray(nodes),
        edges=np.asarray(edges),
        radii=np.asarray(radii),
        root=0,
    )


def chain(count: int = 11, bend: float = 0.0) -> Network:
    """A straight run of ``count`` nodes along x, optionally kinked in the middle."""
    nodes = np.zeros((count, 3), dtype=float)
    nodes[:, 0] = np.arange(count, dtype=float)
    if bend:
        nodes[count // 2, 1] = bend
    edges = np.array([[i, i + 1] for i in range(count - 1)], dtype=np.int64)
    return Network(nodes=nodes + 100.0, edges=edges)


@pytest.fixture(scope="session")
def objects() -> dict[int, Network]:
    """Two arbors, far enough apart to occupy different cells."""
    return {
        1: arbor(),
        2: arbor(depth=5, origin=(700.0, 500.0, 400.0), seed=9),
    }


@pytest.fixture(scope="session")
def ladder_objects() -> dict[int, Network]:
    """Eight large arbors, enough for the adaptive ladder to build a second level."""
    return {
        seed: arbor(depth=9, spread=40.0,
                    origin=(1200.0 + 1800 * (seed % 4), 1200.0 + 1800 * (seed // 4), 1200.0),
                    seed=seed)
        for seed in LADDER_SEEDS
    }


@pytest.fixture(scope="session")
def collection(objects: dict[int, Network]) -> konnektion.NetworkCollection:
    """A built one-level collection."""
    return konnektion.build_collection(objects, cell_size=(128, 128, 128))


@pytest.fixture(scope="session")
def ladder(ladder_objects: dict[int, Network]) -> konnektion.NetworkCollection:
    """A built collection deep enough to have coarser levels."""
    return konnektion.build_collection(ladder_objects, cell_size=CELL_SIZE)


@pytest.fixture()
def written(collection: konnektion.NetworkCollection) -> konnektion.MemoryStore:
    """The one-level collection, written into a fresh store under a prefix."""
    store = konnektion.MemoryStore()
    collection.write(store, "p")
    return store


@pytest.fixture()
def opened(written: konnektion.MemoryStore) -> konnektion.Collection:
    """The one-level collection, opened back off its store."""
    return konnektion.open_collection(written, "p")


@pytest.fixture()
def opened_ladder(ladder: konnektion.NetworkCollection) -> konnektion.Collection:
    """The multi-level collection, written and opened back."""
    store = konnektion.MemoryStore()
    ladder.write(store, "p")
    return konnektion.open_collection(store, "p")
