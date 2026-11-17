"""JSON serialization of discovered circuits for the web visualization.

The searches produce either a tree of Nodes (TreeMessagePatching*) or a list of
scored root->leaf paths (PathMessagePatching*). For the grid view (layers on the
y-axis, token positions on the x-axis) both are collapsed into the same DAG:
every tree node / path element that shares a grid identity (component + token
position, see `grid_key`) becomes one grid node, and parent->child links are
deduplicated into edges carrying a multiplicity. This mirrors the ASCII
`render_position_grid` in experiments/tree_vs_path.py.

Grid identity deliberately ignores the key/value-position and the patch flags:
a head repeated across branches (reading different positions, patched on Q vs
KV, ...) collapses to a single grid node, with its reads expressed by the edges
and the merged variants listed in the node's metadata.
"""

from collections import Counter
import math
from torch import Tensor

from ipe.nodes import Node, EMBED_Node, MLP_Node, ATTN_Node, FINAL_Node


def grid_key(n: Node) -> tuple:
	"""Identity of a node as placed on the grid: component + token position."""
	if isinstance(n, FINAL_Node):
		return ("final", n.position)
	if isinstance(n, EMBED_Node):
		return ("embed", n.position)
	if isinstance(n, MLP_Node):
		return ("mlp", n.layer, n.position)
	if isinstance(n, ATTN_Node):
		return ("attn", n.layer, n.head, n.position)
	return ("unknown", id(n))


def grid_id(n: Node) -> str:
	"""Stable string form of `grid_key`, usable as a DOM/JSON identifier."""
	return ":".join("*" if p is None else str(p) for p in grid_key(n))


def attn_variant(n: ATTN_Node) -> str:
	"""Which stream(s) of the head this tree node patches: 'Q', 'K', 'V', 'KV', ..."""
	v = ""
	if n.patch_query:
		v += "Q"
	if n.patch_key:
		v += "K"
	if n.patch_value:
		v += "V"
	return v or "?"


def node_to_dict(n: Node) -> dict:
	"""Public JSON form of a single tree node (used for live 'admit' events)."""
	kind = grid_key(n)[0]
	d = {
		"id": grid_id(n),
		"kind": kind,
		"layer": getattr(n, "layer", None) if not isinstance(n, EMBED_Node) else None,
		"position": n.position,
		"contribution": getattr(n, "contribution", None),
	}
	if isinstance(n, ATTN_Node):
		d["head"] = n.head
		d["kv_position"] = n.keyvalue_position
		d["variant"] = attn_variant(n)
	return d


def _subtree_reaches_embed(n: Node, memo: dict) -> bool:
	"""True if some root->leaf branch through `n` ends at an EMBED node, i.e. the
    branch is complete rather than pruned mid-way.
    """
	if id(n) in memo:
		return memo[id(n)]
	if isinstance(n, EMBED_Node):
		memo[id(n)] = True
		return True
	res = any(_subtree_reaches_embed(c, memo) for c in n.children)
	memo[id(n)] = res
	return res


class _GraphAccumulator:
	"""Accumulates (parent, child) links of tree nodes / path elements into the
    collapsed grid DAG, merging by grid identity.
    """

	def __init__(self):
		self.nodes: dict[str, dict] = {}
		self.edges: Counter = Counter()
		self.edge_best: dict[tuple[str, str], float] = {}
		self.edge_complete: set[tuple[str, str]] = set()

	def add_node(self, n: Node, complete: bool = True) -> str:
		nid = grid_id(n)
		contribution = getattr(n, "contribution", None)
		if nid not in self.nodes:
			d = node_to_dict(n)
			d.update({"merged": 0, "complete": False, "variants": [], "kv_positions": []})
			d.pop("kv_position", None)
			d.pop("variant", None)
			self.nodes[nid] = d
		d = self.nodes[nid]
		d["merged"] += 1
		d["complete"] = d["complete"] or complete
		if contribution is not None:
			# Keep the signed contribution of the strongest merged branch.
			if d["contribution"] is None or abs(contribution) > abs(d["contribution"]):
				d["contribution"] = contribution
		if isinstance(n, ATTN_Node):
			v = attn_variant(n)
			if v not in d["variants"]:
				d["variants"].append(v)
			if n.keyvalue_position is not None and n.keyvalue_position not in d["kv_positions"]:
				d["kv_positions"].append(n.keyvalue_position)
		return nid

	def add_edge(self, child: Node, parent: Node, complete: bool = True) -> None:
		# Data-flow orientation: the child feeds its message into the parent.
		src, dst = grid_id(child), grid_id(parent)
		if src == dst:
			return
		self.edges[(src, dst)] += 1
		if complete:
			# The edge lies on at least one branch that reaches the embeddings.
			self.edge_complete.add((src, dst))
		c = getattr(child, "contribution", None)
		if c is not None:
			best = self.edge_best.get((src, dst))
			if best is None or abs(c) > abs(best):
				self.edge_best[(src, dst)] = c

	def to_dict(self) -> dict:
		return {
			"nodes": list(self.nodes.values()),
			"edges": [
				{"source": s, "target": t, "count": c,
				 "contribution": self.edge_best.get((s, t)),
				 "complete": (s, t) in self.edge_complete}
				for (s, t), c in self.edges.items()
			],
		}


class _PathProvenance:
	"""Store the original paths and node occurrences discovered
    during circuit search.

    Paths follow the data-flow direction, from embedding to root.
    Original node occurrences are kept separate even when they
    correspond to the same node in the merged graph.
    """

	def __init__(self, source: str, order: str) -> None:
		"""Initialize the structure used to store path provenance.

        Args:
            source (str):
                The search method, either 'tree' or 'path'.
            order (str):
                The ordering label used for stored paths, either 'discovery' or 'tree'/'result'.
        """
		self.source = source
		self.order = order
		self.occurrences: dict[int, dict] = {}
		self.paths: list[dict] = []

	def add(self, path: list[Node], score: float | Tensor | None = None, *, score_kind: str = "unknown") -> None:
		"""Store an original path and its node occurrences, preserving
        their original identities.

        Args:
            path (list[Node]):
                The path nodes ordered from embedding to root.
            score (float or torch.Tensor, optional):
                The path score, stored only for complete paths
                with finite values. Defaults to None.
            score_kind (str, optional):
                The type of score associated with the path.
                Defaults to 'unknown'.
        """
		complete = bool(path and isinstance(path[0], EMBED_Node))
		value = float(score) if complete and score is not None else None
		if value is not None and not math.isfinite(value):
			value = None
		occurrence_ids = []
		for n in path:
			if id(n) not in self.occurrences:
				d = node_to_dict(n)
				d["node_id"] = d.pop("id")
				d["id"] = f"occurrence-{len(self.occurrences) + 1}"
				d["patch_type"] = getattr(n, "patch_type", None)
				if isinstance(n, ATTN_Node):
					for flag in ("patch_query", "patch_key", "patch_value"):
						d[flag] = getattr(n, flag)
				self.occurrences[id(n)] = d
			occurrence_ids.append(self.occurrences[id(n)]["id"])
		self.paths.append({
			"id": f"{self.source}-{len(self.paths) + 1}",
			"occurrence_ids": occurrence_ids,
			"node_ids": [grid_id(n) for n in path],
			"complete": complete,
			"score": value,
			"score_kind": score_kind if complete else None,
		})

	def to_dict(self) -> dict:
		"""Convert the stored path provenance into a dictionary.

        Returns:
            dict:
                The provenance information, including its version,
                source, ordering label, node occurrences and paths.
        """
		return {
			"version": 1,
			"source": self.source,
			"order": self.order,
			"occurrences": list(self.occurrences.values()),
			"paths": self.paths,
		}


def _tree_child_order(n: Node) -> tuple[int | float, str, str, int]:
	"""Create a key to order tree nodes based on their
    discovery index and component information.

    Args:
        n (Node):
            The tree node to order.

    Returns:
        tuple[int | float, str, str, int]:
            The discovery index, grid ID, attention variant
            and key/value position.
    """
	kv = getattr(n, "keyvalue_position", None)
	return (getattr(n, "discovery_index", float("inf")), grid_id(n),
			attn_variant(n) if isinstance(n, ATTN_Node) else "",
			-1 if kv is None else kv)


def graph_from_tree(root: Node, *, joint_scoring: bool | None = None) -> dict:
	"""Convert the discovered tree into a merged graph, preserving
    the original paths and their provenance.

    The function traverses the tree, collects its branches and
    identifies which ones reach an embedding node. The resulting
    graph includes the merged components and their connections,
    together with the original path information.

    Args:
        root (Node):
            The root node of the discovered tree.
        joint_scoring (bool, optional):
            The scoring mode used during the search.
            True indicates joint scoring, False indicates isolated
            path scoring, and None indicates an unknown mode. Defaults to None.

    Returns:
        dict:
            The merged graph containing 'nodes' and 'edges',
            together with 'path_provenance' describing the
            original paths and their scores.
    """
	acc = _GraphAccumulator()
	memo: dict[int, bool] = {}
	branches = []

	def walk(n: Node, ancestors: list[Node]) -> None:
		"""Traverse the tree to collect its nodes, edges and
        original paths.

        Args:
            n (Node):
                The current node in the tree.
            ancestors (list[Node]):
                The nodes visited along the path from the root
                to the current node.
        """
		branch = ancestors + [n]
		if not n.children:
			branches.append(list(reversed(branch)))
		acc.add_node(n, complete=_subtree_reaches_embed(n, memo))
		for child in n.children:
			acc.add_edge(child, n, complete=_subtree_reaches_embed(child, memo))
			walk(child, branch)

	walk(root, [])
	has_order = all(hasattr(branch[0], "discovery_index") for branch in branches)

	if has_order:
		branches.sort(key=lambda branch: branch[0].discovery_index)
	else:
		branches.sort(key=lambda branch: tuple(_tree_child_order(n) for n in reversed(branch)))
	provenance = _PathProvenance("tree", "discovery" if has_order else "tree")
	score_kind = ("isolated_path_effect" if joint_scoring is False else
				  "joint_candidate_effect" if joint_scoring is True else "unknown")
	for branch in branches:
		score = getattr(branch[0], "contribution", None) if isinstance(branch[0], EMBED_Node) else None
		provenance.add(branch, score, score_kind=score_kind)
	return {**acc.to_dict(), "path_provenance": provenance.to_dict()}


def graph_from_paths(paths: list[tuple[float | Tensor | None, list[Node]]]) -> dict:
	"""Convert scored paths into a grid graph while retaining their provenance.

    Each path is added to the graph, merging components with the
    same grid identity. Original path scores are stored separately,
    while missing node contributions can be filled using the
    corresponding path score.

    Args:
        paths (list[tuple[float | torch.Tensor | None, list[Node]]]):
            The discovered paths and their associated scores,
            with nodes ordered from embedding to root.

    Returns:
        dict:
            The merged graph containing 'nodes' and 'edges',
            together with 'path_provenance' describing the
            original paths and their scores.
    """
	acc = _GraphAccumulator()
	has_order = all(path and hasattr(path[0], "discovery_index") for _, path in paths)
	ordered = sorted(paths, key=lambda item: item[1][0].discovery_index) if has_order else paths
	provenance = _PathProvenance("path", "discovery" if has_order else "result")
	for score, path in ordered:
		provenance.add(path, score, score_kind="isolated_path_effect")
	for score, path in paths:
		for n in path:
			# The leaf-side score is the whole path's contribution; per-node
			# admission scores were stored on the nodes by the search when
			# available and take precedence in add_node.
			if getattr(n, "contribution", None) is None and score is not None:
				n.contribution = float(score)
			acc.add_node(n, complete=True)
		for child, parent in zip(path[:-1], path[1:]):
			acc.add_edge(child, parent)
	return {**acc.to_dict(), "path_provenance": provenance.to_dict()}
