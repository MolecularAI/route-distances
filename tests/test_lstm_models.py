import pickle

import pytest
import numpy as np

from route_distances.lstm.inference import distances_calculator
from route_distances.lstm.models import RouteDistanceModel
from route_distances.lstm.utils import collate_trees, collate_batch
from route_distances.lstm.data import InMemoryTreeDataset


@pytest.fixture
def mock_distance_model(mocker):
    class MockedRouteDistanceModel(mocker.MagicMock):
        @property
        def hparams(self):
            pass

    class MockedHparams(mocker.MagicMock):
        @property
        def fp_size(self):
            return 1024

    mocker.patch.object(MockedRouteDistanceModel, "hparams", MockedHparams())
    patched_model_cls = mocker.patch(
        "route_distances.lstm.inference.RouteDistanceModel"
    )
    patched_model_cls.load_from_checkpoint.return_value = MockedRouteDistanceModel()
    return patched_model_cls


def test_distance_calculator_singleton(mocker):
    mocker.patch("route_distances.lstm.inference.RouteDistanceModel")

    calculator1 = distances_calculator("path1")
    calculator2 = distances_calculator("path1")
    calculator3 = distances_calculator("path2")

    assert calculator1 is calculator2
    assert calculator1 is not calculator3


def test_distance_calculator_call(
    mock_distance_model,
    load_reaction_tree,
):
    routes = [
        load_reaction_tree("example_routes.json", 0),
        load_reaction_tree("example_routes.json", 1),
    ]
    calculator1 = distances_calculator("path10")
    patched_model = mock_distance_model.load_from_checkpoint.return_value
    patched_model.return_value.detach.return_value.numpy.return_value = np.asarray(
        [2.0]
    )

    dist_mat = calculator1(routes)

    assert dist_mat.shape == (2, 2)
    assert dist_mat[0, 0] == 0.0
    assert dist_mat[1, 0] == 2.0
    assert dist_mat[0, 1] == 2.0
    assert dist_mat[1, 1] == 0.0


def test_dummy_distance_model(shared_datadir, mocker):
    pickle_path = str(shared_datadir / "test_data.pickle")
    with open(pickle_path, "rb") as fileobj:
        data = pickle.load(fileobj)
    dataset = InMemoryTreeDataset(**data)
    batch = collate_batch([batch for batch in dataset])
    model = RouteDistanceModel(fp_size=32, lstm_size=16, dropout_prob=0.0)
    model.trainer = mocker.MagicMock()
    model._current_fx_name = "training_step"

    assert model.forward(collate_trees(data["trees"])).shape[0] == 45
    assert model.training_step(batch, None).item() == pytest.approx(12.50, rel=1e-2)

    val_data = model.validation_step(batch, None)
    assert all(
        key in val_data for key in ["val_loss", "val_mae", "val_monitor", "val_r2"]
    )
    assert val_data["val_mae"].item() == pytest.approx(1.93, rel=1e-2)

    test_data = model.test_step(batch, None)
    assert all(
        key in test_data for key in ["test_loss", "test_mae", "test_monitor", "test_r2"]
    )
