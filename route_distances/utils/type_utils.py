""" Module containing type definitions """
from typing import Dict, Callable, Any, List

import numpy as np


StrDict = Dict[str, Any]
RouteList = List[StrDict]
RouteDistancesCalculator = Callable[[RouteList], np.ndarray]
