"""The errors konnektion raises, named so a caller can tell a bug from a missing dependency."""

from __future__ import annotations


class KonnektionError(Exception):
    """Base class for every error konnektion raises."""


class MissingExtraError(KonnektionError, ImportError):
    """An optional dependency is needed for what was asked and is not installed."""


class FormatError(KonnektionError, ValueError):
    """The bytes or the declarations do not describe a readable collection."""


class PartitioningError(FormatError):
    """A node does not fit the cell it was assigned to.

    Its own class because it is never a rounding problem. Quantization is per cell, so a node
    outside its cell cannot be represented at all -- and the tempting repair, clamping it onto
    the cell face, is what makes the bug invisible downstream.
    """


class ConnectivityError(FormatError):
    """A level's surviving nodes do not form a connected structure back to their roots.

    **The error konnektion exists to be able to raise.** A mesh that loses a triangle is a mesh
    with a hole, which an eye sees; a graph that loses an interior node is a graph in *pieces*,
    and the pieces still draw. Every one of them looks like data. So the ancestor-closed
    invariant -- a node kept at a level has its whole path to its object's root kept too -- is
    checked rather than assumed, and its violation is named rather than folded into
    :class:`FormatError`.

    Raised by the builder when a prune would orphan a node, and reported by
    :func:`konnektion.verify.verify` at the ``topology`` tier for a collection already written.
    """


class UnfinishedCollectionError(FormatError, FileNotFoundError):
    """A prefix carries no manifest, which is what an interrupted write leaves behind.

    The manifest is written last precisely so this is distinguishable: a tree has no atomic
    "upload finished" flag, so the completion marker has to be a file that only exists once
    everything it points at does.
    """


__all__ = [
    "ConnectivityError",
    "FormatError",
    "KonnektionError",
    "MissingExtraError",
    "PartitioningError",
    "UnfinishedCollectionError",
]
