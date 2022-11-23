import sys
import os
import glob
import pickle

import pytest
import pandas as pd

from route_distances.tools.cluster_aizynth_output import (
    main as calc_route_dist_main,
)
from route_distances.tools.train_lstm_model import main as training_main
from route_distances.tools.prepare_aizynthfinder_output import (
    main as prepare_input_main,
)
from route_distances.lstm.inference import distances_calculator


@pytest.fixture
def add_cli_arguments():
    saved_argv = list(sys.argv)

    def wrapper(args):
        sys.argv = [sys.argv[0]] + args.split(" ")

    yield wrapper
    sys.argv = saved_argv


@pytest.fixture
def run_in_tempdir(tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(old_cwd)


@pytest.fixture
def run_distance_calc(shared_datadir, add_cli_arguments):
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
        "--model ted",
        f"--output {shared_datadir / 'temp_out.hdf5'}",
        "--nclusters 1",
    ]

    def wrapper():
        add_cli_arguments(" ".join(arguments))
        calc_route_dist_main()

    return wrapper


def test_calc_route_distances(shared_datadir, add_cli_arguments):
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
        "--model ted",
        f"--output {shared_datadir/ 'temp_out.hdf5'}",
    ]
    add_cli_arguments(" ".join(arguments))

    calc_route_dist_main()

    assert os.path.exists(str(shared_datadir / "temp_out.hdf5"))
    data = pd.read_hdf(str(shared_datadir / "temp_out.hdf5"), "table")

    assert "distances_time" in data.columns
    assert "cluster_time" not in data.columns
    assert "cluster_labels" not in data.columns

    dist_mat = data.iloc[0].distance_matrix
    assert len(dist_mat) == 3
    assert pytest.approx(dist_mat[0][1], abs=1e-2) == 4.0596
    assert pytest.approx(dist_mat[0][2], abs=1e-2) == 4.7446
    assert pytest.approx(dist_mat[2][1], abs=1e-2) == 1.3149

    dist_mat = data.iloc[1].distance_matrix
    assert len(dist_mat) == 2
    assert pytest.approx(dist_mat[0][1], abs=1e-2) == 4.0596

    assert data.iloc[2].distance_matrix == [[0.0]]


def test_calc_route_distances_with_lstm(shared_datadir, add_cli_arguments):
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
        f"--model {shared_datadir / 'dummy_model.ckpt'}",
        f"--output {shared_datadir/ 'temp_out.hdf5'}",
    ]
    add_cli_arguments(" ".join(arguments))

    calc_route_dist_main()

    assert os.path.exists(str(shared_datadir / "temp_out.hdf5"))


def test_calc_route_clustering(shared_datadir, add_cli_arguments):
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
        "--model ted",
        f"--output {shared_datadir/ 'temp_out.hdf5'}",
        "--nclusters 0",
    ]
    add_cli_arguments(" ".join(arguments))

    calc_route_dist_main()

    data = pd.read_hdf(str(shared_datadir / "temp_out.hdf5"), "table")

    assert "distances_time" in data.columns
    assert "cluster_time" in data.columns
    assert "cluster_labels" in data.columns

    assert data.iloc[0].cluster_labels == [1, 0, 0]
    assert data.iloc[1].cluster_labels == []
    assert data.iloc[2].cluster_labels == []


def test_calc_route_only_clustering(shared_datadir, add_cli_arguments):
    temp_file = str(shared_datadir / "temp_out.hdf5")
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
        "--model ted",
        f"--output {temp_file}",
    ]
    add_cli_arguments(" ".join(arguments))
    calc_route_dist_main()
    # Read in the created file and remove distances_time column
    data = pd.read_hdf(temp_file, "table")
    data = data[["trees", "distance_matrix"]]
    data.to_hdf(temp_file, "table")

    arguments = [
        f"--files {temp_file}",
        "--model ted",
        f"--output {shared_datadir / 'temp_out2.hdf5'}",
        "--nclusters 0",
        "--only_clustering",
    ]
    add_cli_arguments(" ".join(arguments))

    calc_route_dist_main()

    data = pd.read_hdf(str(shared_datadir / "temp_out2.hdf5"), "table")

    assert "distances_time" not in data.columns
    assert "cluster_time" in data.columns
    assert "cluster_labels" in data.columns

    assert data.iloc[0].cluster_labels == [1, 0, 0]
    assert data.iloc[1].cluster_labels == []
    assert data.iloc[2].cluster_labels == []


def test_training_simple(
    run_in_tempdir, tmp_path, shared_datadir, add_cli_arguments, load_reaction_tree
):
    arguments = [
        f"--trees {shared_datadir/'test_data.pickle'}",
        "--epochs 10",
        f"--dropout 0.0",
        "--fp_size 32",
        "--lstm_size 16",
        "--split_part 0.6",
    ]
    add_cli_arguments(" ".join(arguments))
    routes = [
        load_reaction_tree("example_routes.json", 0),
        load_reaction_tree("example_routes.json", 1),
    ]

    # Action
    training_main(seed=1984)

    checkpoints = glob.glob(f"{tmp_path}/route-dist/*/checkpoints/last.ckpt")
    checkpoints += glob.glob(f"{tmp_path}/tb_logs/route-dist/*/checkpoints/last.ckpt")
    assert len(checkpoints) == 1

    model1 = distances_calculator(checkpoints[0])
    dist1 = model1(routes)
    model2 = distances_calculator(str(shared_datadir / "dummy_model.ckpt"))
    dist2 = model2(routes)
    assert (dist1 - dist2).sum() < 0.0001


def test_prepare_input_simple(run_distance_calc, shared_datadir, add_cli_arguments):
    run_distance_calc()
    arguments = [
        f"--files {shared_datadir / 'temp_out.hdf5'}",
        f"--output {shared_datadir / 'prep_data.pickle'}",
    ]
    add_cli_arguments(" ".join(arguments))

    prepare_input_main()

    assert os.path.exists(shared_datadir / "prep_data.pickle")

    with open(shared_datadir / "prep_data.pickle", "rb") as fileobj:
        data = pickle.load(fileobj)
    assert "trees" in data
    assert "pairs" in data
    assert len(data["trees"]) == 6
    assert len(data["pairs"]) == 14
    # Just check a few
    assert data["pairs"][0] == (0, 0, 0.0, 1, 1)
    assert data["pairs"][1] == (0, 1, pytest.approx(4.0596, abs=1e-2), 0, -1)
    assert data["pairs"][3] == (1, 0, pytest.approx(4.0596, abs=1e-2), 0, -1)


def test_prepare_input_reduced(run_distance_calc, shared_datadir, add_cli_arguments):
    run_distance_calc()
    arguments = [
        f"--files {shared_datadir / 'temp_out.hdf5'}",
        f"--output {shared_datadir / 'prep_data.pickle'}",
        "--use_reduced",
    ]
    add_cli_arguments(" ".join(arguments))

    prepare_input_main()

    assert os.path.exists(shared_datadir / "prep_data.pickle")

    with open(shared_datadir / "prep_data.pickle", "rb") as fileobj:
        data = pickle.load(fileobj)
    assert "trees" in data
    assert "pairs" in data
    assert len(data["trees"]) == 6
    assert len(data["pairs"]) == 10
