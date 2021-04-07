""" Module containing utilities for TED calculations """
from __future__ import annotations
from typing import List
import random
from enum import Enum
from operator import itemgetter

from apted import Config as BaseAptedConfig
from scipy.spatial.distance import jaccard as jaccard_dist

from route_distances.utils.type_utils import StrDict


class TreeContent(str, Enum):
    """Possibilities for distance calculations on reaction trees"""

    MOLECULES = "molecules"
    REACTIONS = "reactions"
    BOTH = "both"


class AptedConfig(BaseAptedConfig):
    """
    This is a helper class for the tree edit distance
    calculation. It defines how the substitution
    cost is calculated and how to obtain children nodes.

    :param randomize: if True, the children will be shuffled
    :param sort_children: if True, the children will be sorted
    """

    def __init__(self, randomize: bool = False, sort_children: bool = False) -> None:
        super().__init__()
        self._randomize = randomize
        self._sort_children = sort_children

    def rename(self, node1: StrDict, node2: StrDict) -> float:
        if node1["type"] != node2["type"]:
            return 1

        fp1 = node1["fingerprint"]
        fp2 = node2["fingerprint"]
        return jaccard_dist(fp1, fp2)

    def children(self, node: StrDict) -> List[StrDict]:
        if self._sort_children:
            return sorted(node["children"], key=itemgetter("sort_key"))
        if not self._randomize:
            return node["children"]
        children = list(node["children"])
        random.shuffle(children)
        return children
