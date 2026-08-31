"""konnektion -- a level-of-detail graph wire format.

A **network collection** is an octree of node/edge graphs written to an object store: traced
neurons, vessel trees, skeletons, connectomes, tracking graphs with divisions. Sibling to
``fabriks`` (which partitions surfaces) and ``sporadik`` (sparse arrays); the prefix layout, the
manifest-last completion protocol and the Morton cell addressing are deliberately the same as
fabriks's, so a reader who knows one knows the other.

    import konnektion

    collection = konnektion.build_collection(
        {1: (nodes, edges)},            # (n, 3) voxel positions, (m, 2) endpoint indices
        cell_size=(128, 128, 32),       # the source array's chunk shape, (x, y, z)
        axes=("x", "y", "z"),
    )
    store = konnektion.MemoryStore()
    collection.write(store, "my-prefix")
    print(konnektion.verify(konnektion.open_collection(store, "my-prefix"), tier="topology"))

Three things distinguish it from a mesh format, and all three are consequences of the data being
a graph rather than a surface:

**Levels are optional.** Depth is chosen from the data. A traced arbor of a few thousand nodes
gets one level -- nothing pruned, nothing straightened, every node exactly where the tracer put
it -- and the manifest declares ``pruning: NONE`` / ``simplification: NONE`` so that is checkable.

**Coarsening is two independent operations.** Strahler pruning drops whole branches, which is what
makes a dense arbor legible when zoomed out; Douglas-Peucker straightens the runs that survive,
which is the only thing that helps a single long wiggly vessel. Each is declared separately.

**Nothing ever moves a node.** Both operations *remove* nodes and neither repositions one, so a
coarse level is a sub-graph of the fine one rather than an approximation, and node identity
survives coarsening -- which is what lets a ghost in one cell be matched to its owner in another.

Where each part is
------------------
:mod:`konnektion.manifest` is what a collection declares about itself; :mod:`konnektion.codecs` is
the wire format, and is normative; :mod:`konnektion.octree` is how space is addressed;
:mod:`konnektion.geometry` is Strahler order, runs and Douglas-Peucker; :mod:`konnektion.build`
assembles levels; :mod:`konnektion.writer` and :mod:`konnektion.reader` are the two ends of a
store; :mod:`konnektion.verify` checks the claims nothing downstream can see.
"""

from konnektion.build import (
    MAX_LEVELS,
    OVERVIEW_TARGET_BYTES,
    NetworkCollection,
    build_collection,
    choose_cell_size,
)
from konnektion.codecs import QUANT_MAX, decode_edges, decode_positions, encode_positions
from konnektion.errors import (
    ConnectivityError,
    FormatError,
    KonnektionError,
    MissingExtraError,
    PartitioningError,
    UnfinishedCollectionError,
)
from konnektion.frames import REQUIRED_COLUMNS, arrow_schemas, validate_columns
from konnektion.geometry import (
    douglas_peucker,
    prune_to_order,
    simplify,
    strahler_orders,
    unbranched_runs,
)
from konnektion.manifest import (
    MANIFEST_NAME,
    SPEC_VERSION,
    Coarsening,
    Encoding,
    Grid,
    Manifest,
)
from konnektion.octree import cell_box, cell_of, morton_decode, morton_encode
from konnektion.reader import Collection, DecodedCell, open_collection
from konnektion.sources import Network, coerce_network
from konnektion.stores import DirectoryStore, KonnektionStore, MemoryStore
from konnektion.verify import TIERS, Check, VerifyReport, verify
from konnektion.writer import awrite_collection, write_collection

__all__ = [
    "MANIFEST_NAME",
    "MAX_LEVELS",
    "OVERVIEW_TARGET_BYTES",
    "QUANT_MAX",
    "REQUIRED_COLUMNS",
    "SPEC_VERSION",
    "TIERS",
    "Check",
    "Coarsening",
    "Collection",
    "ConnectivityError",
    "DecodedCell",
    "DirectoryStore",
    "Encoding",
    "FormatError",
    "Grid",
    "KonnektionError",
    "KonnektionStore",
    "Manifest",
    "MemoryStore",
    "MissingExtraError",
    "Network",
    "NetworkCollection",
    "PartitioningError",
    "UnfinishedCollectionError",
    "VerifyReport",
    "arrow_schemas",
    "build_collection",
    "cell_box",
    "cell_of",
    "choose_cell_size",
    "coerce_network",
    "decode_edges",
    "decode_positions",
    "douglas_peucker",
    "encode_positions",
    "morton_decode",
    "morton_encode",
    "open_collection",
    "prune_to_order",
    "simplify",
    "strahler_orders",
    "unbranched_runs",
    "validate_columns",
    "verify",
    "awrite_collection",
    "write_collection",
]
