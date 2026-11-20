from ipe.nodes import Node, FINAL_Node
import torch
from transformer_lens import HookedTransformer
from typing import Callable

@torch.no_grad()
def evaluate_path(path: list[Node], metric: Callable[[torch.Tensor], float], *, on_evaluated: Callable[[list[Node], torch.Tensor], None] | None = None) -> float | torch.Tensor | None:
	"""Evaluate an isolated path by propagating its message to the root.

    Subtract the propagated message from the root's unmodified output
    and pass the patched residual to the metric.

    Args:
        path (list[Node]):
            The sequence of nodes representing the path to be evaluated.
        metric (Callable):
            A function to evaluate the contribution or importance of the path. 
			It must accept a single parameter, the output of the last node when 
			the path is removed.
        on_evaluated (Callable, optional):
            Observer called with (path, patched residual) after scoring.
            It reuses the evaluated intervention without propagating the
            path again. Defaults to None.

    Returns:
        float or torch.Tensor or None:
            The contribution score of the path as determined by the metric function.
    """
	message = None
	if len(path) == 0:
		return message

	for i in range(len(path)):
		message = path[i].forward(message=message)

	corrupted_resid = path[-1].forward() - message
	score = metric(corrupted_resid=corrupted_resid)
	if on_evaluated is not None:
		on_evaluated(path, corrupted_resid)
	return score

@torch.no_grad()
def get_tree_msg(node: Node) -> torch.Tensor:
	"""
	Recursively computes the message a node sends to its parent in a tree.

	The message arriving at a node is the sum of the messages of all its children, removed jointly from the node's clean input before its component is re-evaluated. A leaf (no children) emits forward(None), matching path semantics where the deepest node sends its full output.

	Args:
		node (Node):
			The node whose outgoing message is computed.

	Returns:
		torch.Tensor:
			The message produced by the node given the joint ablation of its children's subtrees.
	"""
	if not node.children:
		return node.forward(message=None)
	incoming = sum(get_tree_msg(child) for child in node.children)
	return node.forward(message=incoming)

@torch.no_grad()
def evaluate_tree(root: Node, metric: Callable[[torch.Tensor], float]) -> float:
	"""
	Evaluates the contribution of a whole tree by jointly ablating every branch and applying the metric. evaluate_path is the degenerate single-child case.

	Args:
		root (Node):
			The root of the tree (e.g. a FINAL_Node) whose children point backwards towards the EMBED_Nodes.
		metric (Callable):
			A function to evaluate the contribution of the tree. It must accept a single parameter, the output of the root when the tree is removed.

	Returns:
		float:
			The contribution score of the tree as determined by the metric function.
	"""
	message = get_tree_msg(root) if root.children else 0
	return metric(corrupted_resid=root.forward() - message)

def get_path(node: Node) -> list[Node]:
	"""
	Constructs the path from the given node back to the root by following parent links.

	Args:
		node (Node): The node from which to start constructing the path.
	
	Returns:
		list of Node:
			The sequence of nodes representing the path from the root to the given node.
	"""
	path = [node]
	while path[-1].parent is not None:
		path.append(path[-1].parent)
	return path


def get_path_msg(path: list[Node], message: torch.Tensor = None) -> torch.Tensor:
	"""
	Recursively computes the message by applying the forward method of each node in the path.

	Args:
		path (list of Node): 
			The sequence of nodes representing the path.
		message (torch.Tensor, default=None):
			Initial message to be passed to the first node in the path.

	Returns:
		torch.Tensor:
			The final message after applying all nodes in the path.
	"""
	if len(path) == 0:
		return message
	message = path[0].forward(message=message)
	return get_path_msg(path[1:], message=message)

def get_path_msgs(path: list[Node], messages: list[torch.Tensor] = [], msg_cache: dict = None, cf_cache: dict = None, model: HookedTransformer = None) -> list[torch.Tensor]:
	"""
	Recursively computes and collects messages by applying the forward method of each node in the path.
	
	Args:
		path (list of Node): 
			The sequence of nodes representing the path.
		messages (list of torch.Tensor, default=[]):
			list to collect messages at each step.
		msg_cache (dict, optional):
			Cache for messages to optimize computation.
		model (HookedTransformer, optional):
			The transformer model to be used in the nodes.
	Returns:
		list of torch.Tensor:
			The list of all messages flowing through the path including intermediate ones."""
	if not path:
		return messages
	
	# Determine the patch tensor
	message = messages[-1] if messages else None
	
	# Compute the next message and append it
	if msg_cache is not None:
		path[0].msg_cache = msg_cache
	if cf_cache is not None:
		path[0].cf_cache = cf_cache
	if model is not None:
		path[0].model = model
	next_message = path[0].forward(message=message)
	messages.append(next_message)

	# Recurse with the rest of the path
	return get_path_msgs(path[1:], messages=messages, msg_cache=msg_cache, cf_cache=cf_cache, model=model)

def clean_paths(paths: list[tuple[float, list]], inplace: bool = False) -> list[tuple[float, list]]:
	"""Cleans up the paths by removing references to models, parents, children, and caches to save memory. It is useful to call this function before saving the outputs of graph search to a file, avoiding saving cache, gradients, and model.
	
	Args:
		paths (list[tuple[float, list]]): 
			A list of tuples where each tuple contains a path weight and a list of Node instances representing the path.
	Returns:
		list[tuple[float, list]]: 
			The cleaned list of paths with unnecessary references removed.
	"""
	cleaned = []
	if inplace:
		for c, path in paths:
			for node in path:
				node.model = None  # remove model reference to save memory
				node.parent = None  # remove parent reference to save memory
				node.children = []  # remove children reference to save memory
				node.msg_cache = None  # remove msg_cache reference to save memory
				node.cf_cache = None  # remove cf_cache reference to save memorys
				node.gradient = None  # remove gradient reference to save memory
				if isinstance(node, FINAL_Node):
					node.metric = None  # remove metric reference to save memory
			if isinstance(c, torch.Tensor):
				c = c.item()  # convert tensor to float for serialization
			cleaned.append((c, path))
	else:
		for c, path in paths:
			cleaned_path = []
			for node in path:
				cleaned_node = type(node)(
					model=node.model,
					layer=node.layer,
					position=node.position,
					patch_type=node.patch_type
				)
				cleaned_node.model = None  # remove model reference to save memory
				# Copy other attributes if they exist
				if hasattr(node, 'head'):
					cleaned_node.head = node.head
				if hasattr(node, 'keyvalue_position'):
					cleaned_node.keyvalue_position = node.keyvalue_position
				if hasattr(node, 'patch_query'):
					cleaned_node.patch_query = node.patch_query
				if hasattr(node, 'patch_key'):
					cleaned_node.patch_key = node.patch_key
				if hasattr(node, 'patch_value'):
					cleaned_node.patch_value = node.patch_value
				cleaned_path.append(cleaned_node)
			if isinstance(c, torch.Tensor):
				c = c.item()  # convert tensor to float for serialization
			cleaned.append((c, cleaned_path))
	return cleaned

@torch.no_grad()
def tree_messages(root: Node) -> dict[int, torch.Tensor]:
	"""
	Computes the message emitted by every node of a tree, keyed by object identity.

	Same recursion as :func:`get_tree_msg`, but every intermediate message is retained so that a
	change confined to a single branch can be re-propagated (see :func:`evaluate_tree_branch`)
	without re-evaluating the whole tree.

	Args:
		root (Node):
			The root of the tree (e.g. a FINAL_Node) whose children point backwards towards the EMBED_Nodes.

	Returns:
		dict[int, torch.Tensor]:
			A mapping from ``id(node)`` to the message that node emits to its parent.

	Notes:
		- Object identity is the key on purpose: :meth:`Node.__hash__` hashes by component identity,
		  so the same head appearing in two different branches would otherwise collide.
	"""
	messages: dict[int, torch.Tensor] = {}

	def walk(node: Node) -> torch.Tensor:
		if not node.children:
			message = node.forward(message=None)
		else:
			message = node.forward(message=sum(walk(child) for child in node.children))
		messages[id(node)] = message
		return message

	walk(root)
	return messages

@torch.no_grad()
def evaluate_tree_branch(root: Node, node: Node, message: torch.Tensor, metric: Callable[[torch.Tensor], float], messages: dict[int, torch.Tensor]) -> torch.Tensor:
	"""
	Evaluates the tree when `node` emits `message` instead of the message recorded for it in
	`messages`, every other branch being held at its recorded value.

	Only the `node` -> root chain is re-evaluated: a change under `node` cannot reach the root by any
	other route, and the siblings it meets at each ancestor are unaffected. The cost is therefore
	O(depth) rather than O(size of the tree), which is what makes joint scoring during the search
	affordable.

	Args:
		root (Node):
			The root of the tree the messages in `messages` were computed from.
		node (Node):
			The node whose emitted message is replaced. It must belong to that tree.
		message (torch.Tensor):
			The message `node` emits instead of its recorded one.
		metric (Callable):
			A function to evaluate the contribution of the tree. It must accept a single parameter, the output of the root when the tree is removed.
		messages (dict[int, torch.Tensor]):
			Messages of the unmodified tree, as returned by :func:`tree_messages`.

	Returns:
		torch.Tensor:
			The contribution score of the modified tree as determined by the metric function.
	"""
	current = node
	while current is not root:
		parent = current.parent
		incoming = message
		for sibling in parent.children:
			if sibling is not current:
				incoming = incoming + messages[id(sibling)]
		message = parent.forward(message=incoming)
		current = parent
	return metric(corrupted_resid=root.forward() - message)
