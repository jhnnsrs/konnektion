"""What a collection declares about itself: the version, the octree, and how the bytes are packed.

``konnektion.json`` sits at the root of the prefix and is **written last**. Everything a reader
needs to interpret the tree is here, and nothing here is defaulted on a reader's behalf: the
manifest is the one file that says how to read every other one, so a value it omits is a value
somebody has to guess, and a guess about an encoding is not an error -- it is geometry that
decodes to garbage.

That principle is inherited from fabriks and it is worth restating because konnektion has one
more way to fall foul of it. A mesh has a single obvious primitive; a graph has two, and the
edge blob is a flat integer array whose *arity* is the only thing separating a segment list from
a triangle list. Reading one as the other produces no error at any layer -- the lengths divide,
the indices are in range, and the picture is nonsense. So the arity is stated, and stated as a
required key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from konnektion.errors import FormatError

# --------------------------------------------------------------------------- #
# The version
# --------------------------------------------------------------------------- #

#: The format version this writer emits and this reader accepts. It selects how every byte in
#: the prefix is read, so it is never defaulted on a reader's behalf and never guessed.
#:
#: **1 is the first published version.** Like fabriks's, it includes the two things a reader
#: needs to fetch *one cell* rather than its whole level: the ``part`` / ``row_group`` locator on
#: every cell-catalog row, and a byte length beside every file named in ``files``. The length is
#: what lets a reader range-read a Parquet part without being able to stat it -- a store is only
#: asked for ``get``/``put``/``list``, and an HTTP one can do neither ``head`` nor ``list``.
#:
#: This is the number a server checks against -- see ``SUPPORTED_VERSIONS`` in mikro's
#: ``datalayer/konnektion.py``, kept in step by the contract rather than by an import. A bump
#: here is a bump there, made in the same change or the store becomes unreadable.
SPEC_VERSION = "1"

#: The manifest's name, at the root of the collection's prefix.
MANIFEST_NAME = "konnektion.json"

#: The two catalogs, at paths the format fixes.
CELL_CATALOG_PATH = "catalog/cells.parquet"
OBJECT_CATALOG_PATH = "catalog/objects.parquet"

#: A dense ordinal is 24 bits in the format, so a collection holds at most this many objects.
MAX_ORDINAL = 1 << 24

#: An octree of networks is three-dimensional, so ``shape`` and ``axes`` are three long. The rank
#: is fixed by the Morton key and the ``bbox_*_{x,y,z}`` columns, not a parameter.
_SHAPE_RANK = 3


def level_prefix(level: int) -> str:
    """The directory holding one octree level's geometry.

    ``level0`` rather than the Hive-style ``level=0``: ``=`` is a sub-delimiter rather than an
    unreserved character, so a SigV4 canonical request carries it as ``%3D`` while a signer that
    treats a path as opaque leaves it bare -- two different strings to sign for one object, and a
    ``SignatureDoesNotMatch`` that looks like a credentials problem.
    """
    return f"level{int(level)}"


def level_part_path(level: int, part: int = 0) -> str:
    """The path of one geometry part inside a level.

    A level is a *directory* rather than a file so a large level can be split across parts
    without the layout changing shape -- a reader globs the level and reads what it finds.
    """
    return f"{level_prefix(level)}/part-{int(part):05d}.parquet"


# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #

#: Node positions are three ``uint16`` quantized against the cell's own grid box.
POSITIONS_UINT16_QUANTIZED_PER_CELL = "UINT16_QUANTIZED_PER_CELL"

#: Edges are pairs of ``uint32`` cell-local node indices -- a **segment list**, not a strip.
#:
#: A strip would cost one integer per segment instead of two, and it cannot express a branch: a
#: strip is a path, and the structures this format exists for are trees and networks. The extra
#: integer per segment buys arbitrary topology, which is the whole point.
EDGES_UINT32_PAIRS = "UINT32_PAIRS"

#: Global node identifiers, so a node keeps its identity across cells and across levels.
NODE_IDS_UINT64 = "UINT64"
NODE_IDS_UINT32 = "UINT32"

#: Per-node radius. ``NONE`` means the collection carries none and a renderer falls back to a
#: flat width.
#:
#: Stored rather than derived, unlike a mesh's normals: fabriks argues at length (``NORMALS.md``)
#: that a normal is recoverable from positions plus winding and so is redundancy rather than
#: information. A radius is not recoverable from anything. It is a **measurement** -- the calibre
#: of a dendrite, the bore of a vessel -- and dropping it would lose data, not just a hint.
RADII_NONE = "NONE"
RADII_FLOAT32 = "FLOAT32"
RADII_UINT16_QUANTIZED_PER_CELL = "UINT16_QUANTIZED_PER_CELL"

#: How a cell carries the endpoints it does not own.
#:
#: ``TRAILING_PER_OWNER_CELL``: ghosts are the **tail** of the cell's node array, after its
#: ``node_count`` owned nodes, and each one is quantized against **the box of the cell that owns
#: it** -- named per ghost in the ``ghost_cells`` column -- rather than against this cell's.
#:
#: That last part is forced, not chosen, and it is the one thing about ghosts that is not
#: obvious. Quantization is per cell precisely so a decoder needs nothing but ``level`` and
#: ``cell`` to invert it; a ghost is by definition a node *outside* this cell, so this cell's box
#: cannot represent it at all. The alternatives are worse: widening the box breaks the property
#: that makes per-cell quantization work, and clamping the ghost onto the cell face draws every
#: crossing edge stopping short at a plane.
#:
#: Storing ghosts separately also removes the need for a mask. A bitset saying which nodes are
#: copies is exactly the information "the last ``ghost_count`` of them are", so the array's own
#: layout carries it and there is nothing to keep in step.
GHOSTS_TRAILING = "TRAILING_PER_OWNER_CELL"

#: The blob codec. ``NONE`` is the default and the one that needs nothing installed: a blob is
#: the raw little-endian layout :mod:`konnektion.codecs` describes, which a consumer hands
#: straight to a vertex buffer. A second codec would arrive as a **value** -- a module beside
#: the others and an entry in the table in :mod:`konnektion.codecs.blobs` -- not as a new field.
CODEC_NONE = "NONE"

COMPRESSION_NONE = "NONE"
COMPRESSION_ZSTD = "ZSTD"

# There is deliberately no `boundary` key here, and its absence is a design decision rather
# than an omission. fabriks declares `boundary: LOCKED` -- vertices on a cell face plane do not
# move -- which is what lets a fine cell be drawn beside a coarse one without a crack. That
# guarantee is **not available to a format that prunes branches**, and pretending otherwise
# would be the worst kind of declaration: one nothing can check and everyone would rely on.
#
# Two reasons it cannot hold. A traced node sits at an arbitrary position, so unlike a clipped
# triangle's new vertices there are typically *no* nodes on a cell plane to pin. And a branch
# present at level 0 may be absent at level 1 entirely, so an edge crossing a plane can exist at
# one level and not the other -- no amount of pinning recovers that.
#
# What konnektion offers instead is that **every level is independently correct**: coarsening is
# decided per object over the whole graph and only then partitioned into cells, so within one
# level every cell agrees, and a ghost is always a copy of a node that really is there. The
# reading advice that follows is in NETWORKS.md: draw a contiguous region at one level. That is
# cheap here in a way it is not for meshes, a graph being far smaller than the surface it runs
# through.

#: How a level drops whole branches. ``STRAHLER`` keeps branches whose Horton-Strahler order is
#: at or above the level's threshold; ``NONE`` drops none, which is what a single-level
#: collection declares and what makes that declaration checkable.
PRUNING_NONE = "NONE"
PRUNING_STRAHLER = "STRAHLER"
PRUNING_CUSTOM = "CUSTOM"

#: How a level straightens the runs that survive pruning. ``DOUGLAS_PEUCKER`` removes interior
#: nodes of an unbranched run whose perpendicular distance to the chord is under the level's
#: epsilon; ``NONE`` removes none.
SIMPLIFICATION_NONE = "NONE"
SIMPLIFICATION_DOUGLAS_PEUCKER = "DOUGLAS_PEUCKER"
SIMPLIFICATION_CUSTOM = "CUSTOM"

SORT_KEY_MORTON = "MORTON"

_ENCODING_VOCABULARY: dict[str, frozenset[str]] = {
    "positions": frozenset({POSITIONS_UINT16_QUANTIZED_PER_CELL}),
    "edges": frozenset({EDGES_UINT32_PAIRS}),
    "nodeIds": frozenset({NODE_IDS_UINT64, NODE_IDS_UINT32}),
    "radii": frozenset({RADII_NONE, RADII_FLOAT32, RADII_UINT16_QUANTIZED_PER_CELL}),
    "ghosts": frozenset({GHOSTS_TRAILING}),
    "codec": frozenset({CODEC_NONE}),
    "compression": frozenset({COMPRESSION_NONE, COMPRESSION_ZSTD}),
    "pruning": frozenset({PRUNING_NONE, PRUNING_STRAHLER, PRUNING_CUSTOM}),
    "simplification": frozenset(
        {SIMPLIFICATION_NONE, SIMPLIFICATION_DOUGLAS_PEUCKER, SIMPLIFICATION_CUSTOM}
    ),
}

#: The keys a reader cannot work without, and which are therefore never defaulted.
#:
#: All of them, and that is deliberate. ``codec`` and ``compression`` are the pair fabriks calls
#: load-bearing, for the reason its own docstring gives: guessing them does not produce an error,
#: it produces geometry that decodes to garbage. ``edges`` is konnektion's addition to that list
#: and it is the sharpest one -- a flat ``uint32`` array read at the wrong arity divides evenly,
#: indexes in range, and draws nonsense. ``pruning`` and ``simplification`` are here because a
#: collection that coarsened nothing must be able to *say so*: declaring a scheme it did not run
#: would be a claim no check could catch.
_REQUIRED_ENCODING_KEYS = (
    "positions",
    "edges",
    "nodeIds",
    "radii",
    "ghosts",
    "codec",
    "compression",
    "pruning",
    "simplification",
)

#: :class:`Encoding` attribute -> the manifest key it is written under. Two of them are not
#: snake_case, and the mapping lives here rather than on the dataclass so that the dataclass has
#: exactly the fields a caller passes -- a field holding this table would become a constructor
#: argument, which is not a thing anyone should be able to override.
_ENCODING_FIELDS: dict[str, str] = {
    "positions": "positions",
    "edges": "edges",
    "node_ids": "nodeIds",
    "radii": "radii",
    "ghosts": "ghosts",
    "codec": "codec",
    "compression": "compression",
    "pruning": "pruning",
    "simplification": "simplification",
}


# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FileEntry:
    """One file the manifest names, with what a reader needs to range-read it.

    ``size`` is here because **nothing else in the tree can tell a reader how long a file is**.
    konnektion asks a store for ``put``/``get``/``list`` and nothing more, so there is no ``head``
    to call, and an HTTP-backed store usually cannot list either. Yet the length is the first
    thing a Parquet reader needs: the footer lives at the end, so a reader that cannot seek to
    the end cannot parse the file at all without downloading it whole -- which is the exact thing
    the locator exists to avoid.
    """

    path: str
    size: int | None = None
    row_groups: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """The entry as it is written, omitting what was never measured."""
        written: dict[str, Any] = {"path": self.path}
        if self.size is not None:
            written["bytes"] = int(self.size)
        if self.row_groups is not None:
            written["rowGroups"] = int(self.row_groups)
        return written

    @classmethod
    def from_any(cls, raw: Any) -> FileEntry:  # noqa: ANN401
        """Read an entry, accepting a bare path string for a hand-written manifest."""
        if isinstance(raw, str):
            return cls(path=raw)
        if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
            size = raw.get("bytes")
            groups = raw.get("rowGroups")
            return cls(
                path=str(raw["path"]),
                size=None if size is None else int(size),
                row_groups=None if groups is None else int(groups),
            )
        raise FormatError(
            f"A file entry in a manifest is a path, or an object carrying one, got {raw!r}."
        )


@dataclass(frozen=True)
class Grid:
    """The octree: how big a level-0 cell is, how many levels there are, how cells are keyed.

    ``cell_size`` is **one size per component, in the same order as the node positions** -- the
    order the ``bbox_*_x/y/z`` columns also use, where ``x``/``y``/``z`` are labels for slots 0,
    1 and 2 rather than claims about physical axes. Nothing here reads a slot as a physical axis,
    so a collection built from ``(z, y, x)`` data states its cell size ``(z, y, x)`` too and is
    entirely consistent. What must not differ is the order *between* the two.

    Units are voxels of the collection's own coordinate system, which is what lets the octree
    align to the grid the graph was traced in.

    **``levels=1`` is legal and is the common case.** A traced arbor of a few thousand nodes has
    nothing to gain from a ladder, and a format that forced one would spend build time producing
    levels that are claims somebody then has to verify. See :class:`Coarsening.none`.
    """

    cell_size: tuple[int, int, int]
    levels: int
    sort_key: str = SORT_KEY_MORTON

    def __post_init__(self) -> None:
        """Refuse a grid that cannot address a graph."""
        if len(self.cell_size) != 3 or any(int(component) < 1 for component in self.cell_size):
            raise FormatError(
                f"`cell_size` is three whole numbers of at least 1 voxel, one per component in "
                f"the same order as the node positions, got {self.cell_size!r}."
            )
        if self.levels < 1:
            raise FormatError(f"An octree has at least one level, got {self.levels}.")
        if self.sort_key != SORT_KEY_MORTON:
            raise FormatError(
                f"`sort_key` is {self.sort_key!r}; the format defines {SORT_KEY_MORTON}."
            )

    def to_dict(self) -> dict[str, Any]:
        """The manifest's ``grid`` object."""
        return {
            "cellSize": [int(component) for component in self.cell_size],
            "levels": int(self.levels),
            "sortKey": self.sort_key,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Grid:
        """Read a manifest's ``grid`` object."""
        try:
            cell_size = tuple(int(component) for component in raw["cellSize"])
        except (KeyError, TypeError, ValueError) as error:
            raise FormatError(
                f"A manifest's `grid` needs a three-component `cellSize`, got {raw!r}."
            ) from error
        if len(cell_size) != _SHAPE_RANK:
            raise FormatError(
                f"A network grid is three-dimensional, so `cellSize` takes 3 values, "
                f"got {raw['cellSize']!r}."
            )
        return cls(
            cell_size=cell_size,  # type: ignore[arg-type]
            levels=int(raw.get("levels", 0)),
            sort_key=str(raw.get("sortKey", SORT_KEY_MORTON)),
        )

    def cell_extent(self, level: int) -> tuple[float, float, float]:
        """How many voxels a cell spans per axis at ``level``."""
        scale = 2 ** int(level)
        return tuple(float(component) * scale for component in self.cell_size)  # type: ignore[return-value]


@dataclass(frozen=True)
class Coarsening:
    """What each level does to the level below it, as two independent operations.

    A mesh coarsens one way -- fewer triangles for the same surface. A graph coarsens two ways,
    and conflating them loses the useful one:

    * **pruning** drops whole branches. Topology-changing, and the operation that actually makes
      a dense arbor legible when zoomed out: you do not want a dendrite drawn with three points
      instead of three hundred, you want the twigs *gone* and the shape of the tree left.
    * **simplification** straightens the runs that survive. Topology-preserving, and the
      operation that makes a single long wiggly vessel cheaper -- the case pruning cannot touch.

    Each level ``L`` prunes at Strahler order ``strahler_step * L`` and straightens at epsilon
    ``epsilon * 2**L`` voxels, so both scale with the cell that holds them.

    The default epsilon is **half a voxel at level 0**: a node moved less than that has moved
    less than the grid the graph was traced in can express, so level 0 is coarsened only in the
    sense of dropping nodes that carried no information. It doubles per level with the cell.

    ``floor_nodes`` is the smallest any one object is reduced to: below a couple of nodes there
    is nothing left to take, and taking it anyway removes the object from the level rather than
    coarsening it. It is **2, not 4** -- fabriks's floor is a tetrahedron's worth of faces
    because a closed surface needs four; the smallest drawable graph is one segment.

    ``declaration`` values are what land in ``encoding.pruning`` / ``encoding.simplification``,
    and they are **checked against the parameters rather than trusted**: declaring ``STRAHLER``
    while pruning nothing would be a claim nothing downstream could test.
    """

    strahler_step: int = 1
    epsilon: float = 0.5
    floor_nodes: int = 2
    pruning: str = PRUNING_STRAHLER
    simplification: str = SIMPLIFICATION_DOUGLAS_PEUCKER

    def __post_init__(self) -> None:
        """Refuse a schedule whose declaration misdescribes what it will do."""
        if self.floor_nodes < 2:
            raise FormatError(
                f"`floor_nodes` is at least one segment's worth, 2, got {self.floor_nodes}."
            )
        if self.strahler_step < 0:
            raise FormatError(f"`strahler_step` is non-negative, got {self.strahler_step}.")
        if self.epsilon < 0.0:
            raise FormatError(f"`epsilon` is non-negative, got {self.epsilon}.")
        prunes = self.strahler_step > 0
        straightens = self.epsilon > 0.0
        if prunes != (self.pruning != PRUNING_NONE):
            raise FormatError(
                f"`pruning` is {self.pruning!r} but `strahler_step` is {self.strahler_step}: a "
                f"schedule that drops no branch declares {PRUNING_NONE}, and one that does must "
                f"not. The declaration is what a reader is told happened, so it is checked here "
                f"rather than trusted."
            )
        if straightens != (self.simplification != SIMPLIFICATION_NONE):
            raise FormatError(
                f"`simplification` is {self.simplification!r} but `epsilon` is {self.epsilon}: a "
                f"schedule that moves no node declares {SIMPLIFICATION_NONE}, and one that does "
                f"must not."
            )
        for key, value in (("pruning", self.pruning), ("simplification", self.simplification)):
            if value not in _ENCODING_VOCABULARY[key]:
                raise FormatError(
                    f"`{key}` is {value!r}; the format defines "
                    f"{', '.join(sorted(_ENCODING_VOCABULARY[key]))}."
                )

    @classmethod
    def none(cls) -> Coarsening:
        """The schedule that does nothing, for a collection with one level.

        Not a degenerate case to be tolerated but the **default for small data**, and the reason
        it is a named constructor: at one level every node is exactly where the data put it,
        ``lod_error`` is zero by construction rather than a bound to measure, and the two
        declarations say ``NONE`` truthfully. A verifier reading this collection skips the
        pruning and error checks and *reports the skip*, because a check that passes for want of
        anything to check is worse than one that says so.
        """
        return cls(
            strahler_step=0,
            epsilon=0.0,
            pruning=PRUNING_NONE,
            simplification=SIMPLIFICATION_NONE,
        )

    @property
    def coarsens(self) -> bool:
        """Whether this schedule changes anything at all."""
        return self.pruning != PRUNING_NONE or self.simplification != SIMPLIFICATION_NONE

    def strahler_threshold(self, level: int) -> int:
        """The smallest Strahler order that survives into ``level``."""
        return 1 + self.strahler_step * int(level)

    def epsilon_at(self, level: int) -> float:
        """The Douglas-Peucker tolerance at ``level``, in voxels."""
        return float(self.epsilon) * (2 ** int(level))

    def moves_nodes(self) -> bool:
        """Whether this schedule ever repositions a node. It does not, and cannot.

        Worth having as a method because it is the fact the format's error bound rests on:
        pruning **removes** nodes and Douglas-Peucker **removes** nodes, and neither moves one.
        So every node in a coarse level sits exactly where the tracer put it, and ``lod_error``
        measures how far the *polyline* strayed from the original polyline -- never how far a
        node strayed from itself, which is always zero.

        This is the one place konnektion is strictly better off than a mesh format: a decimated
        surface has to move vertices to stay a surface, which is why fabriks needs a pinning
        rule and an error bound per vertex. A graph does not.
        """
        return False


@dataclass(frozen=True)
class Encoding:
    """How the blobs are packed. Every value is a claim a decoder acts on.

    Nothing here is defaulted from konnektion's side beyond the keys that have exactly one legal
    value under this format. ``codec``, ``compression`` and ``edges`` in particular are always
    stated by the writer: nothing in the bytes reveals how they were packed, and a wrong guess is
    not an error anywhere, it is geometry that decodes to garbage.
    """

    positions: str = POSITIONS_UINT16_QUANTIZED_PER_CELL
    edges: str = EDGES_UINT32_PAIRS
    node_ids: str = NODE_IDS_UINT64
    radii: str = RADII_NONE
    ghosts: str = GHOSTS_TRAILING
    codec: str = CODEC_NONE
    compression: str = COMPRESSION_NONE
    pruning: str = PRUNING_NONE
    simplification: str = SIMPLIFICATION_NONE

    def __post_init__(self) -> None:
        """Refuse a value outside the format's vocabulary, or a pair that cannot be read."""
        for attribute, key in _ENCODING_FIELDS.items():
            value = getattr(self, attribute)
            allowed = _ENCODING_VOCABULARY[key]
            if value not in allowed:
                raise FormatError(
                    f"`encoding.{key}` is {value!r}; the format defines "
                    f"{', '.join(sorted(allowed))}."
                )

    def to_dict(self) -> dict[str, str]:
        """The manifest's ``encoding`` object, always complete.

        Never sparse: a renderer configures its decoder from what it reads back, so a manifest
        that omits a key it resolved internally hands every reader an encoding that says nothing.
        """
        return {key: getattr(self, attribute) for attribute, key in _ENCODING_FIELDS.items()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Encoding:
        """Read a manifest's ``encoding`` object, refusing one that leaves a decoder guessing."""
        missing = [key for key in _REQUIRED_ENCODING_KEYS if key not in raw]
        if missing:
            raise FormatError(
                f"This manifest's `encoding` omits {', '.join(missing)}. A decoder cannot infer "
                f"them -- a wrong guess is not an error, it is geometry that decodes to garbage "
                f"-- so the collection is refused."
            )
        return cls(
            positions=str(raw["positions"]),
            edges=str(raw["edges"]),
            node_ids=str(raw["nodeIds"]),
            radii=str(raw["radii"]),
            ghosts=str(raw["ghosts"]),
            codec=str(raw["codec"]),
            compression=str(raw["compression"]),
            pruning=str(raw["pruning"]),
            simplification=str(raw["simplification"]),
        )

    @property
    def has_radii(self) -> bool:
        """Whether the collection carries a per-node radius."""
        return self.radii != RADII_NONE


def _read_shape(raw: Any) -> tuple[int, int, int] | None:  # noqa: ANN401
    """Read a manifest's ``shape``, refusing one that is not three-dimensional."""
    if raw is None:
        return None
    try:
        values = tuple(int(component) for component in raw)
    except (TypeError, ValueError) as error:
        raise FormatError(f"A manifest's `shape` is three whole numbers, got {raw!r}.") from error
    if len(values) != _SHAPE_RANK:
        raise FormatError(
            f"A network lives in three dimensions, so `shape` takes 3 values, got {raw!r}."
        )
    return values  # type: ignore[return-value]


def _read_axes(raw: Any) -> list[str] | None:  # noqa: ANN401
    """Read a manifest's ``axes``, refusing one that does not name three."""
    if raw is None:
        return None
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise FormatError(f"A manifest's `axes` is a list of three names, got {raw!r}.")
    names = [str(name) for name in raw]
    if len(names) != _SHAPE_RANK:
        raise FormatError(f"A network has three axes, so `axes` takes 3 names, got {raw!r}.")
    return names


@dataclass(frozen=True)
class Manifest:
    """``konnektion.json``: what the collection is, and how to read every other file in it.

    ``axes`` names what each position slot holds, and **it should be passed**. Left unset the
    manifest states ``null``, and a consumer then has nothing to check against and must trust
    whatever it is handed -- which for a format whose slots are deliberately anonymous is the
    difference between a collection that can be validated and one that cannot.

    ``shape`` and ``axes`` are written even when null, deliberately: a reader can then tell
    "considered and unanswerable" from "predates the question".
    """

    grid: Grid
    encoding: Encoding
    spec_version: str = SPEC_VERSION
    shape: tuple[int, int, int] | None = None
    axes: list[str] | None = None
    counts: dict[str, int] = field(default_factory=dict)
    #: What the writer actually landed: ``cells`` and ``objects`` as single entries, and
    #: ``levels`` as a mapping of level number to the list of parts that level was split into.
    #: Left empty on a built-but-unwritten collection -- the paths are the writer's decisions.
    files: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The manifest as it is written."""
        return {
            "specVersion": self.spec_version,
            "grid": self.grid.to_dict(),
            "encoding": self.encoding.to_dict(),
            "shape": None if self.shape is None else [int(v) for v in self.shape],
            "axes": None if self.axes is None else list(self.axes),
            "counts": {key: int(value) for key, value in sorted(self.counts.items())},
            "files": dict(sorted(self.files.items())),
        }

    def to_json(self) -> bytes:
        """The manifest as the bytes that land at the root of the prefix, written last."""
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=False).encode("utf-8")

    @classmethod
    def from_json(cls, body: bytes) -> Manifest:
        """Read a manifest from the bytes at the root of a prefix."""
        import json

        try:
            raw = json.loads(body)
        except ValueError as error:
            raise FormatError(f"This prefix's {MANIFEST_NAME} is not JSON: {error}") from error
        if not isinstance(raw, Mapping):
            raise FormatError(f"A manifest is a JSON object, got {type(raw).__name__}.")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Manifest:
        """Read a manifest, refusing a version or a declaration this reader cannot honour.

        Unknown top-level keys are ignored on purpose -- a later version may add one and this
        reader should not choke on it -- but an unknown ``specVersion`` is **not**, because the
        version is precisely the statement that the rest of the file means what this reader
        thinks it means.
        """
        version = str(raw.get("specVersion", ""))
        if version != SPEC_VERSION:
            raise FormatError(
                f"This manifest declares specVersion {version!r}, which konnektion cannot read. "
                f"Supported: {SPEC_VERSION}. The version selects how every byte in the prefix is "
                f"read, so an unknown one is refused rather than read as though it were familiar."
            )
        grid = raw.get("grid")
        encoding = raw.get("encoding")
        if not isinstance(grid, Mapping):
            raise FormatError(f"A manifest carries a `grid` object, got {grid!r}.")
        if not isinstance(encoding, Mapping):
            raise FormatError(f"A manifest carries an `encoding` object, got {encoding!r}.")
        counts = dict(raw.get("counts") or {})
        files = dict(raw.get("files") or {})
        return cls(
            grid=Grid.from_dict(grid),
            encoding=Encoding.from_dict(encoding),
            spec_version=version,
            shape=_read_shape(raw.get("shape")),
            axes=_read_axes(raw.get("axes")),
            counts={str(key): int(value) for key, value in counts.items()},
            files={str(key): value for key, value in files.items()},
        )


__all__ = [
    "CELL_CATALOG_PATH",
    "CODEC_NONE",
    "COMPRESSION_NONE",
    "COMPRESSION_ZSTD",
    "EDGES_UINT32_PAIRS",
    "GHOSTS_TRAILING",
    "MANIFEST_NAME",
    "MAX_ORDINAL",
    "NODE_IDS_UINT32",
    "NODE_IDS_UINT64",
    "OBJECT_CATALOG_PATH",
    "POSITIONS_UINT16_QUANTIZED_PER_CELL",
    "PRUNING_CUSTOM",
    "PRUNING_NONE",
    "PRUNING_STRAHLER",
    "RADII_FLOAT32",
    "RADII_NONE",
    "RADII_UINT16_QUANTIZED_PER_CELL",
    "SIMPLIFICATION_CUSTOM",
    "SIMPLIFICATION_DOUGLAS_PEUCKER",
    "SIMPLIFICATION_NONE",
    "SORT_KEY_MORTON",
    "SPEC_VERSION",
    "Coarsening",
    "Encoding",
    "FileEntry",
    "Grid",
    "Manifest",
    "level_part_path",
    "level_prefix",
]
