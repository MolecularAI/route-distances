""" Module containing helper routines for routes """
from typing import Dict, Any, Set, List, Tuple

import numpy as np

from route_distances.utils.type_utils import StrDict


def calc_depth(tree_dict: StrDict, depth: int = 0) -> int:
    """
    Calculate the depth of a route, recursively

    :param tree_dict: the route
    :param depth: the current depth, don't specify for route
    """
    children = tree_dict.get("children", [])
    if children:
        return max(calc_depth(child, depth + 1) for child in children)
    return depth


def calc_llr(tree_dict: StrDict) -> int:
    """
    Calculate the longest linear route for a synthetic route

    :param tree_dict: the route
    """
    return calc_depth(tree_dict) // 2


def extract_leaves(
    tree_dict: StrDict,
) -> Set[str]:
    """
    Extract a set with the SMILES of all the leaf nodes, i.e.
    starting material

    :param tree_dict: the route
    :return: a set of SMILE strings
    """

    def traverse(tree_dict: StrDict, leaves: Set[str]) -> None:
        children = tree_dict.get("children", [])
        if children:
            for child in children:
                traverse(child, leaves)
        else:
            leaves.add(tree_dict["smiles"])

    leaves = set()
    traverse(tree_dict, leaves)
    return leaves


def is_solved(route: StrDict) -> bool:
    """
    Find if a route is solved, i.e. if all starting material
    is in stock.

    To be accurate, each molecule node need to have an extra
    boolean property called `in_stock`.

    :param route: the route to analyze
    """

    def find_leaves_not_in_stock(tree_dict: StrDict) -> None:
        children = tree_dict.get("children", [])
        if not children and not tree_dict.get("in_stock", True):
            raise ValueError(f"child not in stock {tree_dict}")
        elif children:
            for child in children:
                find_leaves_not_in_stock(child)

    try:
        find_leaves_not_in_stock(route)
    except ValueError:
        return False
    return True


def route_score(
    tree_dict: StrDict,
    mol_costs: Dict[bool, float] = None,
    average_yield=0.8,
    reaction_cost=1.0,
) -> float:
    """
    Calculate the score of route using the method from
    (Badowski et al. Chem Sci. 2019, 10, 4640).

    The reaction cost is constant and the yield is an average yield.
    The starting materials are assigned a cost based on whether they are in
    stock or not. By default starting material in stock is assigned a
    cost of 1 and starting material not in stock is assigned a cost of 10.

    To be accurate, each molecule node need to have an extra
    boolean property called `in_stock`.

    :param tree_dict: the route to analyze
    :param mol_costs: the starting material cost
    :param average_yield: the average yield, defaults to 0.8
    :param reaction_cost: the reaction cost, defaults to 1.0
    :return: the computed cost
    """
    mol_cost = mol_costs or {True: 1, False: 10}

    reactions = tree_dict.get("children", [])
    if not reactions:
        return mol_cost[tree_dict.get("in_stock", True)]

    child_sum = sum(
        1 / average_yield * route_score(child) for child in reactions[0]["children"]
    )
    return reaction_cost + child_sum


def route_scorer(routes: List[StrDict]) -> Tuple[List[StrDict], List[float]]:
    """
    Scores and sort a list of routes.
    Returns a tuple of the sorted routes and their costs.

    :param routes: the routes to score
    :return: the sorted routes and their costs
    """
    scores = np.asarray([route_score(route) for route in routes])
    sorted_idx = np.argsort(scores)
    routes = [routes[idx] for idx in sorted_idx]
    return routes, scores[sorted_idx].tolist()


def route_ranks(scores: List[float]) -> List[int]:
    """
    Compute the rank of route scores. Rank starts at 1

    :param scores: the route scores
    :return: a list of ranks for each route
    """
    ranks = [1]
    for idx in range(1, len(scores)):
        if abs(scores[idx] - scores[idx - 1]) < 1e-8:
            ranks.append(ranks[idx - 1])
        else:
            ranks.append(ranks[idx - 1] + 1)
    return ranks
