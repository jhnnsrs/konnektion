"""The store a collection is written into, and the little that konnektion asks of one.

A collection is **a tree, not a file**: a manifest, two catalogs and one directory per
octree level. That shape is why the store is a parameter rather than a path -- the same tree
has to land on a local disk during development and in an S3 prefix in production, and nothing
above this package should care which.

What konnektion asks for is three methods::

    store.put(path, data)     # bytes -> path, relative to the store's own root
    store.get(path)           # -> bytes, or anything with a .bytes() method
    store.list(prefix)        # -> the paths under a prefix

and one it will *use if it is there*::

    store.get_range(path, start=..., length=...)   # -> the bytes in that window

That is deliberately the shape obstore already has, so ``S3Store``, ``LocalStore``,
``GCSStore``, ``AzureStore`` and ``MemoryStore`` are all usable **as they are** -- there is no
adapter, and obstore is not a dependency of konnektion. :class:`DirectoryStore` is here so that a
plain filesystem path works with no dependencies at all.

``get_range`` is optional because it is the one method a hand-rolled store is likely to be
missing, and its absence must degrade rather than fail: :func:`get_range_bytes` falls back to
fetching the whole object and slicing it, which is exactly what konnektion did everywhere before
range reads existed. What it buys when present is the whole point of the format -- reading one
cell out of a level costs one row group rather than the level.

Paths are always ``/``-joined and relative to the store, never absolute and never
``..``-relative: a collection names files inside its own tree, and a writer that could escape
it would be a path-traversal surface in whatever is holding the credentials.

The package is split the way that contract is: :mod:`~konnektion.stores.protocol` states what a
store must offer, :mod:`~konnektion.stores.access` is how konnektion calls through it,
:mod:`~konnektion.stores.file` is the seekable view pyarrow needs, and
:mod:`~konnektion.stores.directory` and :mod:`~konnektion.stores.memory` are the two implementations
that ship.
"""

from konnektion.stores.access import (
    aget_bytes,
    aget_range_bytes,
    get_bytes,
    get_range_bytes,
    list_paths,
    put_bytes,
)
from konnektion.stores.directory import DirectoryStore
from konnektion.stores.file import StoreFile
from konnektion.stores.memory import MemoryStore
from konnektion.stores.protocol import (
    AsyncReadable,
    KonnektionStore,
    RangeReadable,
    join,
    validate_relative,
)

__all__ = [
    "AsyncReadable",
    "DirectoryStore",
    "KonnektionStore",
    "MemoryStore",
    "RangeReadable",
    "StoreFile",
    "aget_bytes",
    "aget_range_bytes",
    "get_bytes",
    "get_range_bytes",
    "join",
    "list_paths",
    "put_bytes",
    "validate_relative",
]
