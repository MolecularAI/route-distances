""" Module contain method to compute distance matrix using TED """
import time

import numpy as np

from route_distances.ted.reactiontree import ReactionTreeWrapper
from route_distances.utils.type_utils import RouteList


def distance_matrix(
    routes: RouteList, content: str = "both", timeout: int = None
) -> np.ndarray:
    """
    Compute the distance matrix between each pair of routes

    :param routes: the routes to calculate pairwise distance on
    :param content: determine what part of the tree to include in the calculation
    :param timeout: if given, raises an exception if timeout is taking longer time
    :return: the square distance matrix
    """
    distances = np.zeros([len(routes), len(routes)])
    distance_wrappers = [ReactionTreeWrapper(route, content) for route in routes]
    time0 = time.perf_counter()
    for i, iwrapper in enumerate(distance_wrappers):
        # fmt: off
        for j, jwrapper in enumerate(distance_wrappers[i + 1:], i + 1):
            distances[i, j] = iwrapper.distance_to(jwrapper)
            distances[j, i] = distances[i, j]
        # fmt: on
        time_past = time.perf_counter() - time0
        if timeout is not None and time_past > timeout:
            raise ValueError(f"Unable to compute distance matrix in {timeout} s")
    return distances
