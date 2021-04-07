""" Module containing CLI tool calculate route distances and do clustering """
from __future__ import annotations
import argparse
import warnings
import functools
import time
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from route_distances.route_distances import route_distances_calculator
from route_distances.clustering import ClusteringHelper
from route_distances.utils.type_utils import StrDict, RouteDistancesCalculator


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Tool to calculate pairwise distances for AiZynthFinder output"
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--nclusters", type=int, default=None)
    parser.add_argument("--output", default="finder_output_dist.hdf5")
    return parser.parse_args()


def _make_empty_dict(nclusters: Optional[int]) -> StrDict:
    dict_ = {"distance_matrix": [[0.0]], "distances_time": 0}
    if nclusters is not None:
        dict_["cluster_labels"] = []
        dict_["cluster_time"] = 0.0
    return dict_


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


def _calc_distances(
    row: pd.Series, nclusters: Optional[int], calculator: RouteDistancesCalculator
) -> pd.Series:
    if len(row.trees) == 1:
        return pd.Series(_make_empty_dict(nclusters))

    time0 = time.perf_counter_ns()
    distances = calculator(row.trees)
    dict_ = {
        "distance_matrix": distances.tolist(),
        "distances_time": (time.perf_counter_ns() - time0) * 1e-9,
    }

    if nclusters is not None:
        time0 = time.perf_counter_ns()
        dict_["cluster_labels"] = ClusteringHelper.cluster(
            distances, nclusters
        ).tolist()
        dict_["cluster_time"] = (time.perf_counter_ns() - time0) * 1e-9

    return pd.Series(dict_)


def main() -> None:
    """ Entry-point for CLI tool """
    args = _get_args()
    tqdm.pandas()

    calculator = route_distances_calculator("ted", content="both")
    data = _merge_inputs(args.files)

    func = functools.partial(
        _calc_distances, nclusters=args.nclusters, calculator=calculator
    )
    dist_data = data.progress_apply(func, axis=1)
    data = data.assign(
        distance_matrix=dist_data.distance_matrix,
        distances_time=dist_data.distances_time,
    )
    if args.nclusters is not None:
        data = data.assign(
            cluster_labels=dist_data.cluster_labels,
            cluster_time=dist_data.cluster_time,
        )

    with warnings.catch_warnings():  # This wil suppress a PerformanceWarning
        warnings.simplefilter("ignore")
        data.to_hdf(args.output, "table")


if __name__ == "__main__":
    main()
