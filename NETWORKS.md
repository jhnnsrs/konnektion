# Why konnektion is not fabriks with two-index faces

The obvious way to store a traced neuron in an existing level-of-detail format is to add a
`topology` field to [fabriks](../fabriks): `TRIANGLES | LINES | POINTS`, indices of arity 3, 2 or
0, everything else unchanged. That design was worked out in full and rejected. This document says
why, and — more usefully — what the three decisions that replaced it cost, so the next person to
ask does not have to re-derive the answer.

Every structural claim below cites a line. Every number was measured, and the method is given so
it can be re-measured.

---

## 1. Why not a topology field on fabriks

The change is genuinely small on the wire: `Encoding.topology`, an arity-aware divisor in about a
dozen places, and a `radii` blob. Roughly a dozen `*3` / `//3` sites plus the one chokepoint in
`fabriks/codecs/blobs.py`. It is small enough to be tempting.

**It is not small anywhere else.** Three things do not survive the move:

**The verify tier means something different.** fabriks's `geometry` tier checks that on-plane
vertices survive decimation (`boundary: LOCKED`) and that `lod_error` bounds vertex movement. A
graph has neither claim to make — see §3 — so the tier would be skipped for every LINES collection,
and a format whose central guarantee does not apply to half its values is two formats sharing a
manifest.

**The simplifier protocol is face-typed.** `Simplifier.simplify(vertices, faces, *, fixed,
target_faces) -> Simplified(vertices, faces, ...)`. Douglas–Peucker does not fit that shape and
Strahler pruning fits it even less; widening it is a breaking change to a released package whose
existing backends have nothing to do with graphs.

**Clipping has no shared implementation.** `fabriks/geometry.py:clip_to_cells` calls
`trimesh.intersections.slice_mesh_plane` six times per cell. There is no graph equivalent, and the
graph case needs no intersection at all (§2). fabriks would carry trimesh, shapely, scipy and
`fast-simplification` for values that never touch any of them — konnektion's whole dependency list
is numpy and pyarrow.

And the deciding one: **`SPEC_VERSION` is a single literal shared by every collection fabriks has
ever written.** `Encoding.from_dict` silently drops unknown keys, so an optional `topology` would
be *ignored* by an existing reader and the segment list decoded as triangles — no error, plausible
wrong geometry. Making it required means a version bump, which stops every mesh collection in
existing storage from being readable until every reader is updated. A sibling package costs a few
hundred lines of restated Morton addressing and store plumbing, and costs nothing already written.

---

## 2. Ghosts, and why the edge is not split

An edge whose endpoints fall in two cells is the analogue of a triangle crossing a cell plane, and
it is the one place the two formats had to diverge on substance rather than on typing.

fabriks **splits**: the triangle is cut at the plane and the new vertices are real geometry that
belongs in both cells. The split is also what makes `boundary: LOCKED` possible, because the new
vertices land exactly on the plane and can be pinned there.

konnektion **copies**. Splitting an edge would insert a node at the crossing, and *an invented
degree-2 node in a morphology is a measurement artefact*. Node count is a published statistic of a
traced arbor; branch-point degree is data. A format that silently added nodes at cell planes would
make the octree's cell size visible in the morphometry, which is indefensible — the partitioning is
a storage decision and must not be readable off the biology.

So the edge is kept whole, and the endpoint the owning cell does not have is stored as a **ghost**:
a read-only copy, at the tail of the cell's node array.

### Three sub-decisions, each of which could have gone the other way

**The ghost is quantized against the cell that owns it, not the cell holding it.** Forced rather
than chosen. Quantization is per cell precisely so a decoder needs only `level` and `cell` to
invert it, and a ghost is by definition outside this cell — the normalized coordinate lands past
1.0 and `encode_positions` refuses it. This was not theoretical: the first build attempt failed
with `PartitioningError: 1 node(s) fall outside cell 5 at level 0 ... worst normalized coordinate
1.009274`, which is the guard working exactly as intended. The alternatives are worse: widening the
box breaks the property that makes per-cell quantization work at all, and clamping the ghost onto
the cell face draws every crossing edge stopping short at a plane. The owning cell's Morton code is
therefore stored per ghost, in `ghost_cells`.

Inverting against the owner's box has a second benefit that decided it: the ghost reconstructs
**bit-identically** to what the owning cell stores. A ghost is then a copy rather than a second,
slightly different opinion about where a node is — which is what makes
`verify(tier="topology")`'s ghost check exact rather than approximate.

**An edge has exactly one owning cell** — the lower Morton code of its two endpoints' cells. Both
cells holding it would also be self-contained, and would draw the segment twice; under any blending
but opaque that is a brighter line running along every cell plane. Measured on the test fixture:
one owner gives 940 stored edges for 940 input edges, exactly.

**There is no ghost bitset.** An earlier draft stored one bit per node saying which were copies.
Once ghosts are stored in their own blob they are simply the tail of the array, so `node_count` and
`ghost_count` already say it — a bitset would be a second copy of the same fact, and two copies of
a fact are a chance for them to disagree.

### What it costs

One duplicated node per crossing. Measured on the eight-arbor fixture (40 888 nodes, 256-voxel
cells): **1 766 ghosts at level 0, 4.3%**. Scaling with cell size, since crossings scale with
surface area rather than volume: a smaller cell means more crossings and more ghosts.

---

## 3. konnektion makes no boundary claim, and says so

fabriks declares `boundary: LOCKED`: vertices on a cell face plane did not move, so a fine cell
drawn beside a coarse one meets it without a crack. It is the load-bearing claim of that format,
and the one nothing downstream can see.

**konnektion has no such key.** Two reasons, and neither is fixable:

A traced node sits at an arbitrary position. Unlike a clipped triangle's new vertices, there are
typically *no* nodes on a cell plane to pin — the set the claim would quantify over is empty, so
the claim would be vacuously true on every collection ever written.

And a branch present at level 0 may be absent at level 1 *entirely*. That is not a bug, it is what
Strahler pruning is for. No amount of pinning recovers an edge that one level has and the other
does not, so a seam between levels is inherent to topological LOD rather than a defect in the
implementation.

Declaring `LOCKED` anyway would have been the worst available option: a claim nothing can check
and everyone would rely on. The vocabulary omits it.

**What is offered instead** is that every level is *independently* correct. Coarsening is decided
per object over the whole graph and only then partitioned into cells
(`konnektion/build.py:build_collection`), so within one level every cell agrees about which nodes
exist and a ghost is always a copy of a node that really is there. Partitioning per cell and
coarsening each cell separately — the order a mesh format uses — would prune by a different
threshold on each side of a plane, because Strahler order is a property of the whole tree and a
cell holding one twig cannot know how much hangs below it.

The reading advice that follows: **draw a contiguous region at one level.** That is cheap here in a
way it is not for meshes — a graph is far smaller than the surface it runs through.

---

## 4. Strahler pruning, and why not just Douglas–Peucker

Three different things get called graph LOD:

1. **Spatial culling** — fetch only the subgraph in view. This *is* fabriks's problem and its
   octree solves it unchanged.
2. **Geometric decimation** — straighten runs. Fewer nodes, identical topology.
3. **Topological pruning** — drop whole twigs, keep the trunk.

Only the third is new, and it is the one that matters. Zoomed out you do not want a dendrite drawn
with three points instead of three hundred; you want the twigs *gone* and the arbor's shape legible.
Douglas–Peucker alone makes a dense arbor smaller without making it readable.

**Strahler order is the right metric, and one property decides it.** A leaf is order 1; an internal
node takes its children's maximum, plus one when two or more children tie. Invented for river
networks, standard in neuron morphometry — and, crucially:

> Strahler order is non-decreasing toward the root.

A parent takes at least its largest child's order. So thresholding on it — keep every node of order
≥ k — is **ancestor-closed by construction**: a kept node's parent has an order at least as large
and is therefore kept too. The invariant the format exists to protect falls out of the metric rather
than being imposed on top of it, and the builder needs no repair pass.

It is still verified. A proof about an algorithm is not a proof about a build, and the failure is
invisible: a graph that loses an interior node is a graph in *pieces*, and every piece still draws.

**Both operations are declared separately** (`pruning`, `simplification`) because a level can do
one and not the other, and a schedule that does neither must be able to say so. `Coarsening`
cross-checks the declaration against the parameters: `Coarsening(strahler_step=0, epsilon=1.0,
pruning="STRAHLER")` raises, because declaring a scheme you did not run is a claim no check could
catch.

**Douglas–Peucker's error metric is the perpendicular distance from the chord**, which is exactly
what `lod_error` is spent as — so the epsilon handed in *is* the bound and the level needs no second
measurement. That coincidence is the reason for Douglas–Peucker over, say, dropping every other
node.

### The one thing that had to be rebuilt rather than inherited

Simplification **re-links** consecutive survivors along a run; it does not inherit edges. This is
the one place where the induced-subgraph rule is exactly wrong: straightening `A-x-y-B` to `A-B`
drops both interior nodes, and every original edge along that run has at least one dropped endpoint
— so inheriting would leave the survivors connected by nothing. Measured before the fix: a 2 551-node
pruned arbor simplified to **513 nodes and 1 edge**. After: 513 nodes, 512 edges, one connected
component.

Pruning is the opposite case and *does* inherit: removing a whole branch removes its edges too, and
re-linking would reconnect what was deliberately cut.

---

## 5. Levels are optional

Depth is chosen from the data. The ladder grows only while the coarsest level is still above
`OVERVIEW_TARGET_BYTES` (256 KiB — a level that arrives in one request is small enough).

**`levels=1` is the expected case for traced data, not a degenerate one.** Measured: two arbors of
942 nodes total → **one level**, nothing pruned, nothing straightened, `lod_error` zero by
construction. Eight arbors of 40 888 nodes → **two levels**, L1/L0 = 0.10.

A format that forced a ladder would spend build time producing levels that are claims somebody then
has to verify. At one level the manifest declares `pruning: NONE` and `simplification: NONE`, which
is checkable, and the verifier **skips** the cross-level checks and *reports the skip* — a check
that passes for want of anything to check reads identically to one that really ran, which makes the
report worthless exactly where it is most needed.

---

## 6. What connectivity is checked against

An earlier draft compared each level's component count against level 0. A negative test caught that
this is vacuous for a one-level collection — which is the common case — so **the object catalog
declares `component_count`**, measured once at build time, and every level including level 0 is held
to it.

A disconnected object is legitimate: two unlinked vessel segments are two components and always
were. What is never legitimate is *gaining* pieces, which is what a lost edge, a mis-assigned ghost
and a non-ancestor-closed prune all look like from the outside.

Measured: severing one cell's edges while keeping every count valid takes object 1 from 1 piece to
**172**, and the check names it. Emptying the edge blob instead is caught one tier earlier by the
count check — which is why the test severs count-preservingly, or it would be exercising the wrong
check.

---

## 7. Bounding boxes are grown by one quantization step

A cell's box is what a viewer culls against, and what it will be culling is the *decoded* geometry.
Quantization can move a node by up to half a step, so a box measured from the pre-quantization
floats can exclude the very node it describes — and a cell culled away by its own bounding box is a
hole in the drawing with no error anywhere. Caught by a test asserting decoded positions lie inside
their catalog box; the discrepancy was 5.6e-4 voxels against a 1.95e-3 quantum.

The box also covers the cell's **ghosts**, because this cell is the one that draws those edges. A
box stopping at the cell face would cull away geometry the row is responsible for.
