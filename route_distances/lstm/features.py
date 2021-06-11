""" Module for calculating feature and utility vectors for the LSTM-based model """
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from treelstm import calculate_evaluation_orders

import route_distances.lstm.defaults as defaults
from route_distances.lstm.utils import (
    add_node_index,
    gather_adjacency_list,
    gather_node_attributes,
)
from route_distances.validation import validate_dict
from route_distances.utils.type_utils import StrDict


def add_fingerprints(
    tree: StrDict,
    radius: int = 2,
    nbits: int = defaults.FP_SIZE,
) -> None:
    """
    Add Morgan fingerprints to the input tree

    :param tree: the input tree
    :param radius: the radius of the Morgan calculation
    :param nbits: the length of the bitvector
    """
    mol = Chem.MolFromSmiles(tree["smiles"])
    rd_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=nbits, useFeatures=False, useChirality=True
    )
    np_fp = np.empty(radius, np.int8)
    DataStructs.ConvertToNumpyArray(rd_fp, np_fp)
    tree["fingerprint"] = np_fp
    for child in tree.get("children", []):
        add_fingerprints(child, radius, nbits)


def remove_reactions(tree: StrDict) -> StrDict:
    """
    Remove reaction nodes from the input tree

    Does not overwrite the original tree.
    """
    new_tree = {"smiles": tree["smiles"]}
    if tree.get("children"):
        new_tree["children"] = [
            remove_reactions(grandchild)
            for grandchild in tree["children"][0]["children"]
        ]
    return new_tree


def preprocess_reaction_tree(
    tree: StrDict, nfeatures: int = defaults.FP_SIZE
) -> StrDict:
    """
    Preprocess a reaction tree as produced by AiZynthFinder

    :param tree: the input tree
    :param nfeatures: the number of features, i.e. fingerprint length
    :return: a tree that could be fed to the LSTM-based model
    """
    validate_dict(tree)
    tree = remove_reactions(tree)
    add_fingerprints(tree, nbits=nfeatures)
    add_node_index(tree)

    features = np.asarray(gather_node_attributes(tree, "fingerprint"))
    adjacency_list = gather_adjacency_list(tree)

    if adjacency_list:
        node_order, edge_order = calculate_evaluation_orders(
            adjacency_list, len(features)
        )
    else:
        node_order = np.asarray([0])
        edge_order = np.asarray([])

    return {
        "features": features,
        "node_order": node_order,
        "adjacency_list": np.array(adjacency_list),
        "edge_order": edge_order,
        "num_nodes": len(features),
        "num_trees": 1,
    }
