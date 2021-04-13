""" Module containing CLI tool calculate route distances and do clustering """
from __future__ import annotations
import argparse
import warnings
import functools
import time
from typing import List

import pandas as pd
from tqdm import tqdm

from route_distances.route_distances import route_distances_calculator
from route_distances.clustering import ClusteringHelper
from route_distances.utils.type_utils import RouteDistancesCalculator


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Tool to calculate pairwise distances for AiZynthFinder output"
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--only_clustering", action="store_true", default=False)
    parser.add_argument("--nclusters", type=int, default=None)
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
            data = data.append(temp_data)
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


def _do_clustering(row: pd.Series, nclusters: int) -> pd.Series:
    if row.distance_matrix == [[0.0]] or len(row.trees) < 3:
        return pd.Series({"cluster_labels": [], "cluster_time": 0})

    time0 = time.perf_counter_ns()
    labels = ClusteringHelper.cluster(row.distance_matrix, nclusters).tolist()
    cluster_time = (time.perf_counter_ns() - time0) * 1e-9
    return pd.Series({"cluster_labels": labels, "cluster_time": cluster_time})


def main() -> None:
    """ Entry-point for CLI tool """
    args = _get_args()
    tqdm.pandas()

    data = _merge_inputs(args.files)

    if args.only_clustering:
        calculator = None
    else:
        calculator = route_distances_calculator("ted", content="both")

    if not args.only_clustering:
        func = functools.partial(
            _calc_distances, calculator=calculator
        )
        dist_data = data.progress_apply(func, axis=1)
        data = data.assign(
            distance_matrix=dist_data.distance_matrix,
            distances_time=dist_data.distances_time,
        )

    if args.nclusters is not None:
        func = functools.partial(_do_clustering, nclusters=args.nclusters)
        cluster_data = data.progress_apply(func, axis=1)
        data = data.assign(
            cluster_labels=cluster_data.cluster_labels,
            cluster_time=cluster_data.cluster_time,
        )

    with warnings.catch_warnings():  # This wil suppress a PerformanceWarning
        warnings.simplefilter("ignore")
        data.to_hdf(args.output, "table")


if __name__ == "__main__":
    main()
