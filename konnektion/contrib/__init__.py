"""Adapters between konnektion and the graph libraries a caller already has.

**Nothing here is imported by konnektion itself**, and nothing here is re-exported from the
package root: the format's dependency list is numpy and pyarrow, and an adapter is the one
place a graph library may be imported. Each lives in ``konnektion/contrib/<library>.py`` behind
an extra of the same name, and raises :class:`~konnektion.errors.MissingExtraError` when its
library is absent.

What an adapter is
------------------
A graph library's node is a key; a konnektion node is a position in voxels with an integer id.
Everything an adapter does is that translation, and the rules are the same for every library so
that the next one is written against them rather than designed again:

* ``from_<library>(graph, ...)`` returns ``(Network, ids)``, where ``ids`` maps each of the
  graph's node keys to the node id it was given. **Always returned, never applied silently** --
  a table of per-node measurements is joined on those ids, so a renumbering the caller cannot
  see is a join that is quietly wrong.
* ``to_<library>(network, ...)`` is the inverse for an in-memory network.
* **Nothing is guessed.** Which attribute holds the positions, the order of its components,
  which node is the root and which attributes are carried are all arguments.
* **What the format cannot say is refused**, with the fix in the message, rather than dropped:
  a parallel edge, a self-loop, a value that is not a number.
* The result goes through :func:`~konnektion.sources.coerce_network` before it is returned, so
  an adapter adds checks about the *graph* and repeats none about the network.

An edge carries no attributes in the format. Per-edge values belong in a table keyed by the two
node ids of each edge, which is what the returned ``ids`` mapping is for.
"""
