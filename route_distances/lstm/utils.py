""" Module for tree utilities """
from typing import Dict, List, Any
from collections import defaultdict

import torch

from route_distances.utils.type_utils import StrDict


def accumulate_stats(stats: List[Dict[str, float]]) -> Dict[str, float]:
    """Accumulate statistics from a list of statistics"""
    accum: StrDict = defaultdict(float)
    for output in stats:
        for key, value in output.items():
            accum[key] += value
    return accum


def add_node_index(node: StrDict, n: int = 0) -> int:
    """Add an index to the node and all its children"""
    node["index"] = n
    for child in node.get("children", []):
        n += 1
        n = add_node_index(child, n)
    return n


def collate_batch(batch: List[StrDict]) -> StrDict:
    """
    Collate a batch of tree data

    Collate the first tree of all pairs together, and then collate
    the second tree of all pairs.

    Convert all matrices to pytorch tensors.

    The output dictionary has the following keys:
        - tree1: the collated first tree for all pairs
        - tree2: the collated second tree for all pairs
        - ted: the TED for each pair of trees

    :param batch: the list of tree data
    :return: the collated batch
    """

    def _make_tensor(key, dtype):
        return torch.tensor([sample[key] for sample in batch], dtype=dtype)

    trees1 = collate_trees([sample["tree1"] for sample in batch])
    trees2 = collate_trees([sample["tree2"] for sample in batch])
    teds = _make_tensor("ted", torch.float32)
    return {"tree1": trees1, "tree2": trees2, "ted": teds}


def collate_trees(trees: List[StrDict]) -> StrDict:
    """
    Collate a list of trees by stacking the feature vectors, the node orders and the
    edge orders. The adjacency list if adjusted with an offset.

    This is a modified version from treelstm package that also converts all matrices to tensors

    The output dictionary has the following keys:
        - features: the stacked node features
        - node_order: the stacked node orders
        - edge_order: the stacked edge orders
        - adjacency_list: the stack and adjusted adjacency list
        - tree_size: the number of nodes in each tree

    :param trees: the trees to collate
    :return: the collated tree data
    """

    def _make_tensor(key, dtype):
        return torch.cat([torch.tensor(tree[key], dtype=dtype) for tree in trees])

    tree_sizes = [tree["num_nodes"] for tree in trees]

    batched_features = _make_tensor("features", torch.float32)
    batched_node_order = _make_tensor("node_order", torch.int64)
    batched_edge_order = _make_tensor("edge_order", torch.int64)

    batched_adjacency_list = []
    offset = 0
    for nnodes, tree in zip(tree_sizes, trees):
        batched_adjacency_list.append(
            torch.tensor(tree["adjacency_list"], dtype=torch.int64) + offset
        )
        offset += nnodes
    batched_adjacency_list = torch.cat(batched_adjacency_list)  # noqa

    return {
        "features": batched_features,
        "node_order": batched_node_order,
        "edge_order": batched_edge_order,
        "adjacency_list": batched_adjacency_list,
        "tree_sizes": tree_sizes,
    }


def gather_adjacency_list(node: StrDict) -> List[List[int]]:
    """
    Create the adjacency list of a tree

    :param node: the current node in the tree
    :return: the adjacency list
    """
    adjacency_list = []
    for child in node.get("children", []):
        adjacency_list.append([node["index"], child["index"]])
        adjacency_list.extend(gather_adjacency_list(child))

    return adjacency_list


def gather_node_attributes(node: StrDict, key: str) -> List[Any]:
    """
    Collect node attributes by recursively traversing the tree

    :param node: the current node in the tree
    :param key: the name of the attribute to extract
    :return: the list of attributes gathered
    """
    features = [node[key]]
    for child in node.get("children", []):
        features.extend(gather_node_attributes(child, key))
    return features
