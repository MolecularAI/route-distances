""" Module containing a factory function for making predictions of route distances """
import functools
from typing import Any, List

from route_distances.ted.distances import distance_matrix
from route_distances.utils.type_utils import RouteDistancesCalculator, StrDict

SUPPORTED_MODELS = ["ted"]


def route_distances_calculator(model: str, **kwargs: Any) -> RouteDistancesCalculator:
    """
    Return a callable that given a list routes as dictionaries
    calculate the squared distance matrix

    :param model: the model identifier
    :param kwargs: the model parameters
    :return: the calculator callable
    """
    if model == "ted":
        model_kwargs = _copy_kwargs(["content", "timeout"], **kwargs)
        return functools.partial(distance_matrix, **model_kwargs)

    raise ValueError(f"Model must be one in {SUPPORTED_MODELS}")


def _copy_kwargs(keys_to_copy: List[str], **kwargs: Any) -> StrDict:
    new_kwargs = {}
    for key in keys_to_copy:
        if key in kwargs:
            new_kwargs[key] = kwargs[key]
    return new_kwargs
