""" Module containing class to make predictions of route distance matrix """
import numpy as np
from scipy.spatial.distance import squareform

from route_distances.lstm.features import preprocess_reaction_tree
from route_distances.lstm.utils import collate_trees
from route_distances.lstm.models import RouteDistanceModel
from route_distances.utils.type_utils import RouteList


class _InferenceHelper:
    """
    Helper class for calculating route distances using LSTM model

    The predictions are made by calling the instantiated class with
    a list of routes (in dictionary format).

    :param model_path: the path to the model checkpoint file
    """

    def __init__(self, model_path: str) -> None:
        self._model = RouteDistanceModel.load_from_checkpoint(model_path)
        self._model.eval()

    def __call__(self, routes: RouteList) -> np.ndarray:
        trees = [
            preprocess_reaction_tree(route, self._model.hparams.fp_size)
            for route in routes
        ]
        tree_data = collate_trees(trees)
        pred_torch = self._model(tree_data)
        pred_np = pred_torch.detach().numpy()
        return squareform(pred_np)


_inst_model = {}


def distances_calculator(model_path):
    global _inst_model
    if model_path not in _inst_model:
        _inst_model[model_path] = _InferenceHelper(model_path)
    return _inst_model[model_path]
