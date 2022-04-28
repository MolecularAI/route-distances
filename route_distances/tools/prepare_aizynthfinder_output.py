""" Module for CLI tool to prepare model training input """
import argparse
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

import route_distances.lstm.defaults as defaults
from route_distances.lstm.features import preprocess_reaction_tree


def _get_args():
    parser = argparse.ArgumentParser(
        "Tool to prepare output from AiZynthFinder for model training"
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--fp_size", type=int, default=defaults.FP_SIZE)
    parser.add_argument("--use_reduced", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _similarity(idx1, idx2, labels):
    if len(labels) == 0 or (labels[idx1] == labels[idx2]):
        return 1, 1
    return -1, 0


def main():
    """Entry-point for CLI tool"""
    args = _get_args()

    offset = 0
    tree_list = []
    pairs = []
    for filename in tqdm(args.files, desc="# of files processed: "):
        data = pd.read_hdf(filename, "table")

        for trees, distances, labels in zip(
            tqdm(data.trees.values, leave=False, desc="# of targets processed"),
            data.distance_matrix.values,
            data.cluster_labels.values,
        ):
            np_distances = np.asarray(distances)
            for i, tree1 in enumerate(trees):
                tree_list.append(preprocess_reaction_tree(tree1, args.fp_size))
                for j, _ in enumerate(trees):
                    if j < i and args.use_reduced:
                        continue
                    loss_target, pair_similarity = _similarity(i, j, labels)
                    pairs.append(
                        (
                            i + offset,
                            j + offset,
                            np_distances[i, j],
                            pair_similarity,
                            loss_target,
                        )
                    )
            offset += len(trees)

    print(f"Preprocessed {len(tree_list)} trees in {len(pairs)} pairs")

    with open(args.output, "wb") as fileobj:
        pickle.dump(
            {
                "trees": tree_list,
                "pairs": pairs,
            },
            fileobj,
        )


if __name__ == "__main__":
    main()
