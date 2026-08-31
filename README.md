# konnektion

A level-of-detail **graph** wire format: octree-partitioned node/edge networks written to any
object store.

Sibling to [`fabriks`](../fabriks) (meshes) and `sporadik` (sparse arrays). Where fabriks
partitions *surfaces*, konnektion partitions *networks* — traced neurons, vessel trees,
skeletons, connectomes, tracking graphs with divisions.

```python
import konnektion

collection = konnektion.build_collection(
    {1: (nodes, edges)},          # (n, 3) voxel positions, (m, 2) endpoint indices
    cell_size=(128, 128, 32),     # the source array's chunk shape, (x, y, z)
    axes=("x", "y", "z"),
)
store = konnektion.MemoryStore()
collection.write(store, "my-prefix")

opened = konnektion.open_collection(store, "my-prefix")
print(konnektion.verify(opened, tier="topology"))
```

## What it is

One prefix, laid out like a fabriks collection so that a reader who knows one knows the other:

```
<prefix>/konnektion.json          the manifest, written LAST
<prefix>/catalog/cells.parquet
<prefix>/catalog/objects.parquet
<prefix>/level0/part-00000.parquet
<prefix>/level1/part-00000.parquet   ...
```

The manifest is the completion protocol: it names every other file, so it is written after all of
them, and a prefix without one is an interrupted write rather than a collection.

## The three things that make it a format rather than a convention

**1. Levels are optional.** Depth is chosen from the data, not fixed. A traced arbor of a few
thousand nodes gets `levels=1`: nothing pruned, nothing straightened, every node exactly where the
tracer put it, and the manifest says so with `pruning: NONE` / `simplification: NONE`. Coarsening
you did not do is not a thing to declare.

**2. Coarsening is two operations, not one.** A mesh coarsens one way — fewer triangles for the
same surface. A graph coarsens two:

- **Strahler pruning** drops whole branches. This is the one that makes a dense arbor *legible*
  when zoomed out: you do not want a dendrite drawn with three points instead of three hundred,
  you want the twigs gone and the shape of the tree left.
- **Douglas–Peucker** straightens the runs that survive, which is the only thing that helps a
  single long wiggly vessel.

Each is declared separately, because a level can do one and not the other.

**3. Nothing ever moves a node.** Both operations *remove* nodes; neither repositions one. So a
coarse level is a sub-graph of the fine one rather than an approximation of it, `lod_error` bounds
how far the drawn polyline strayed and never how far a node strayed from itself, and node identity
survives coarsening for free.

## Ghosts

An edge whose endpoints fall in two cells is the graph analogue of a mesh's clipped triangle, and
it is handled differently on purpose. A mesh is *split* at the plane and the new vertices are real
geometry. A graph cannot be split without inventing a node, and an invented degree-2 node in a
morphology is a measurement artefact, not a rendering detail.

So the edge is kept whole and the foreign endpoint is **copied** into the owning cell as a ghost,
quantized against the box of the cell that owns it. A cell stays self-contained — fetch it and you
can draw it — at the cost of one duplicated node per crossing. An edge has exactly one owning cell,
so nothing is drawn twice.

## What it does *not* claim

fabriks declares `boundary: LOCKED`: vertices on a cell face plane do not move, so a fine cell
drawn beside a coarse one meets it without a crack. **konnektion makes no such claim**, and its
absence is a design decision rather than an omission — a branch present at level 0 may be absent at
level 1 entirely, and no amount of pinning recovers that.

What it offers instead is that every level is *independently* correct: coarsening is decided per
object over the whole graph and only then partitioned into cells, so within one level every cell
agrees and a ghost is always a copy of a node that really is there. Draw a contiguous region at one
level. That is cheap here in a way it is not for meshes, a graph being far smaller than the surface
it runs through.

See [NETWORKS.md](NETWORKS.md) for the reasoning behind ghosts, pruning and optional levels.

## Verification

Three tiers, cheapest first, each including the ones before it:

- `structure` — the catalogs agree with each other and with the manifest; every locator resolves.
- `blobs` — every blob decodes and its counts match its row.
- `topology` — the claims nothing downstream can see: **ancestor-closed** pruning, no dangling
  edge, ghost consistency against the owning cell, Strahler monotonicity across levels, coarse
  levels smaller than fine ones, and `lod_error` a real bound.

A single-level collection *skips* the pruning and error checks and says so in the report, rather
than passing them for want of anything to check.
