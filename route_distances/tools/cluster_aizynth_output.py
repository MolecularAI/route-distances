""" Module containing CLI tool calculate route distances and do clustering """
from __future__ import annotations
import argparse
import warnings
import time
import math
from typing import List

import pandas as pd
from tqdm import tqdm

import route_distances.lstm.defaults as defaults
from route_distances.route_distances import route_distances_calculator
from route_distances.clustering import ClusteringHelper
from route_distances.utils.type_utils import RouteDistancesCalculator


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Tool to calculate pairwise distances for AiZynthFinder output"
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--fp_size", type=int, default=defaults.FP_SIZE)
    parser.add_argument("--lstm_size", type=int, default=defaults.LSTM_SIZE)
    parser.add_argument("--model", required=True)
    parser.add_argument("--only_clustering", action="store_true", default=False)
    parser.add_argument("--nclusters", type=int, default=None)
    parser.add_argument("--min_density", type=int, default=None)
    parser.add_argument("--output", default="finder_output_dist.hdf5")
    return parser.parse_args()


def _merge_inputs(filenames: List[str]) -> pd.DataFrame:
    data = None
    for filename in filenames:
        temp_data = pd.read_hdf(filename, "table")
        assert isinstance(temp_data, pd.DataFrame)
        if data is None:
            data = temp_data
        else:
            data = pd.concat([data, temp_data])
    return data


def _calc_distances(row: pd.Series, calculator: RouteDistancesCalculator) -> pd.Series:
    if len(row.trees) == 1:
        return pd.Series({"distance_matrix": [[0.0]], "distances_time": 0})

    time0 = time.perf_counter_ns()
    distances = calculator(row.trees)
    dict_ = {
        "distance_matrix": distances.tolist(),
        "distances_time": (time.perf_counter_ns() - time0) * 1e-9,
    }
    return pd.Series(dict_)


def _do_clustering(
    row: pd.Series, nclusters: int, min_density: int = None
) -> pd.Series:
    if row.distance_matrix == [[0.0]] or len(row.trees) < 3:
        return pd.Series({"cluster_labels": [], "cluster_time": 0})

    if min_density is None:
        max_clusters = min(len(row.trees), 10)
    else:
        max_clusters = int(math.ceil(len(row.trees) / min_density))

    time0 = time.perf_counter_ns()
    labels = ClusteringHelper.cluster(
        row.distance_matrix, nclusters, max_clusters=max_clusters
    ).tolist()
    cluster_time = (time.perf_counter_ns() - time0) * 1e-9
    return pd.Series({"cluster_labels": labels, "cluster_time": cluster_time})


def main() -> None:
    """Entry-point for CLI tool"""
    args = _get_args()
    tqdm.pandas()

    data = _merge_inputs(args.files)

    if args.only_clustering:
        calculator = None
    elif args.model == "ted":
        calculator = route_distances_calculator("ted", content="both")
    else:
        calculator = route_distances_calculator(
            "lstm",
            model_path=args.model,
            fp_size=args.fp_size,
            lstm_size=args.lstm_size,
        )

    if not args.only_clustering:
        dist_data = data.progress_apply(_calc_distances, axis=1, calculator=calculator)
        data = data.assign(
            distance_matrix=dist_data.distance_matrix,
            distances_time=dist_data.distances_time,
        )

    if args.nclusters is not None:
        cluster_data = data.progress_apply(
            _do_clustering,
            axis=1,
            nclusters=args.nclusters,
            min_density=args.min_density,
        )
        data = data.assign(
            cluster_labels=cluster_data.cluster_labels,
            cluster_time=cluster_data.cluster_time,
        )

    with warnings.catch_warnings():  # This wil suppress a PerformanceWarning
        warnings.simplefilter("ignore")
        data.to_hdf(args.output, "table")


if __name__ == "__main__":
    main()
