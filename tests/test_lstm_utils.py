import pytest
from treelstm import calculate_evaluation_orders

from route_distances.lstm.utils import (
    accumulate_stats,
    add_node_index,
    collate_batch,
    collate_trees,
    gather_node_attributes,
    gather_adjacency_list,
)


@pytest.fixture
def toy_tree():
    return {
        "features": [1, 0],
        "children": [
            {"features": [0, 1], "children": [{"features": [1, 1], "children": []}]},
            {"features": [0, 0], "children": []},
        ],
    }


@pytest.fixture
def tree_data(toy_tree):
    add_node_index(toy_tree)
    features = gather_node_attributes(toy_tree, "features")
    adjacency_list = gather_adjacency_list(toy_tree)
    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
    return {
        "features": features,
        "node_order": node_order,
        "adjacency_list": adjacency_list,
        "edge_order": edge_order,
        "num_nodes": len(features),
        "num_trees": 1,
    }


def test_add_node_index(toy_tree):
    add_node_index(toy_tree)

    assert toy_tree["index"] == 0
    assert toy_tree["children"][0]["index"] == 1
    assert toy_tree["children"][1]["index"] == 3
    assert toy_tree["children"][0]["children"][0]["index"] == 2


def test_collate_trees(tree_data):
    collated_trees = collate_trees([tree_data, tree_data])

    assert collated_trees["features"].shape == (8, 2)
    assert collated_trees["node_order"].shape == (8,)
    assert collated_trees["edge_order"].shape == (6,)
    assert collated_trees["adjacency_list"].shape == (6, 2)
    assert collated_trees["tree_sizes"] == [4, 4]

    first_features = collated_trees["features"].numpy()[:4, :].tolist()
    second_features = collated_trees["features"].numpy()[4:, :].tolist()
    assert first_features == second_features
    assert first_features[0] == [1, 0]
    assert first_features[1] == [0, 1]
    assert first_features[2] == [1, 1]
    assert first_features[3] == [0, 0]

    first_node_ord = collated_trees["node_order"].numpy()[:4].tolist()
    second_node_ord = collated_trees["node_order"].numpy()[4:].tolist()
    assert first_node_ord == second_node_ord

    first_edge_ord = collated_trees["edge_order"].numpy()[:3].tolist()
    second_edge_ord = collated_trees["edge_order"].numpy()[3:].tolist()
    assert first_edge_ord == second_edge_ord

    first_adjace = collated_trees["adjacency_list"].numpy()[:3, :].tolist()
    second_adjace = collated_trees["adjacency_list"].numpy()[3:, :].tolist()
    assert first_adjace[0] == [0, 1]
    assert first_adjace[1] == [1, 2]
    assert first_adjace[2] == [0, 3]
    assert second_adjace[0] == [4, 5]
    assert second_adjace[1] == [5, 6]
    assert second_adjace[2] == [4, 7]


def test_collate_batch(tree_data):
    batch = [
        {"tree1": tree_data, "tree2": tree_data, "ted": 0.0},
        {"tree1": tree_data, "tree2": tree_data, "ted": 5.0},
    ]

    collated_batch = collate_batch(batch)

    assert collated_batch["ted"].shape == (2,)
    assert collated_batch["ted"].numpy().tolist() == [0.0, 5.0]

    assert collated_batch["tree1"]["features"].shape == (8, 2)
    assert collated_batch["tree1"]["node_order"].shape == (8,)
    assert collated_batch["tree1"]["edge_order"].shape == (6,)
    assert collated_batch["tree1"]["adjacency_list"].shape == (6, 2)
    assert collated_batch["tree1"]["tree_sizes"] == [4, 4]

    assert collated_batch["tree2"]["features"].shape == (8, 2)
    assert collated_batch["tree2"]["node_order"].shape == (8,)
    assert collated_batch["tree2"]["edge_order"].shape == (6,)
    assert collated_batch["tree2"]["adjacency_list"].shape == (6, 2)
    assert collated_batch["tree2"]["tree_sizes"] == [4, 4]


def test_gather_features(toy_tree):
    features = gather_node_attributes(toy_tree, "features")

    assert len(features) == 4
    assert features[0] == [1, 0]
    assert features[1] == [0, 1]
    assert features[2] == [1, 1]
    assert features[3] == [0, 0]


def test_gather_adjacency_list(toy_tree):
    add_node_index(toy_tree)

    adjacency_list = gather_adjacency_list(toy_tree)

    assert len(adjacency_list) == 3
    assert adjacency_list[0] == [0, 1]
    assert adjacency_list[1] == [1, 2]
    assert adjacency_list[2] == [0, 3]


def test_accumulation():
    stats = [{"one": 1, "two": 5}, {"one": 7, "two": 10}]

    accum = accumulate_stats(stats)

    assert accum["one"] == 8
    assert accum["two"] == 15
