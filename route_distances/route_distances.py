""" Module containing a factory function for making predictions of route distances """
import functools
from typing import Any, List

from route_distances.ted.distances import distance_matrix as ted_distance_matrix
from route_distances.lstm.inference import (
    distances_calculator as lstm_distances_calculator,
)
from route_distances.utils.type_utils import RouteDistancesCalculator, StrDict


def route_distances_calculator(model: str, **kwargs: Any) -> RouteDistancesCalculator:
    """
    Return a callable that given a list routes as dictionaries
    calculate the squared distance matrix

    :param model:
    :param kwargs:
    :return:
    """
    if model not in ["ted", "lstm"]:
        raise ValueError("Model must be either 'ted' or 'lstm'")

    if model == "ted":
        model_kwargs = _copy_kwargs(["content", "timeout"], **kwargs)
        return functools.partial(ted_distance_matrix, **model_kwargs)

    model_kwargs = _copy_kwargs(["model_path"], **kwargs)
    return lstm_distances_calculator(**model_kwargs)


def _copy_kwargs(keys_to_copy: List[str], **kwargs: Any) -> StrDict:
    new_kwargs = {}
    for key in keys_to_copy:
        if key in kwargs:
            new_kwargs[key] = kwargs[key]
    return new_kwargs
