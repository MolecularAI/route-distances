import sys

import pytest
import pandas as pd

from route_distances.tools.cluster_aizynth_output import (
    main as calc_route_dist_main,
)


@pytest.fixture
def add_cli_arguments():
    saved_argv = list(sys.argv)

    def wrapper(args):
        sys.argv = [sys.argv[0]] + args.split(" ")

    yield wrapper
    sys.argv = saved_argv


def test_calc_route_distances(shared_datadir, add_cli_arguments):
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
        f"--output {shared_datadir/ 'temp_out.hdf5'}",
    ]
    add_cli_arguments(" ".join(arguments))

    calc_route_dist_main()

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


def test_calc_route_clustering(shared_datadir, add_cli_arguments):
    arguments = [
        f"--files {shared_datadir / 'finder_output_example.hdf5'}",
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
