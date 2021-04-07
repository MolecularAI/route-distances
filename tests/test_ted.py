import json

import pytest

from route_distances.ted.utils import (
    AptedConfig,
    TreeContent,
)
from route_distances.ted.reactiontree import ReactionTreeWrapper
from route_distances.ted.distances import distance_matrix


def collect_smiles(tree, query_type, smiles_list):
    if tree["type"] == query_type:
        smiles_list.append(tree["smiles"])
    for child in tree.get("children", []):
        collect_smiles(child, query_type, smiles_list)


node1 = {"type": "mol", "fingerprint": [0, 1, 0], "children": ["A", "B", "C"]}

node2 = {"type": "mol", "fingerprint": [1, 1, 0]}


def test_rename_cost_different_types():
    config = AptedConfig()

    cost = config.rename({"type": "type1"}, {"type": "type2"})

    assert cost == 1


def test_rename_cost_same_types():
    config = AptedConfig()

    cost = config.rename(node1, node2)

    assert cost == 0.5


def test_get_children_fixed():
    config = AptedConfig()

    assert config.children(node1) == ["A", "B", "C"]


def test_get_children_random():
    config = AptedConfig(randomize=True)

    children = config.children(node1)

    assert len(children) == 3
    for expected_child in ["A", "B", "C"]:
        assert expected_child in children


@pytest.mark.parametrize(
    "route_index",
    [1, 2],
)
def test_create_wrapper(load_reaction_tree, route_index):
    tree = load_reaction_tree("example_routes.json", route_index)

    wrapper = ReactionTreeWrapper(tree)

    assert wrapper.info["content"] == TreeContent.MOLECULES
    assert wrapper.info["tree count"] == 4
    assert wrapper.first_tree["type"] == "mol"
    assert len(wrapper.trees) == 4

    wrapper = ReactionTreeWrapper(tree, TreeContent.REACTIONS)

    assert wrapper.info["content"] == TreeContent.REACTIONS
    assert wrapper.info["tree count"] == 1
    assert wrapper.first_tree["type"] == "reaction"
    assert len(wrapper.trees) == 1

    wrapper = ReactionTreeWrapper(tree, TreeContent.BOTH)

    assert wrapper.info["content"] == TreeContent.BOTH
    assert wrapper.info["tree count"] == 4
    assert len(wrapper.trees) == 4


def test_create_wrapper_no_reaction():
    tree = {"smiles": "CCC", "type": "mol"}

    wrapper = ReactionTreeWrapper(tree)
    assert wrapper.info["tree count"] == 1
    assert len(wrapper.trees) == 1

    with pytest.raises(ValueError):
        ReactionTreeWrapper(tree, TreeContent.REACTIONS)

    wrapper = ReactionTreeWrapper(tree, TreeContent.BOTH)
    assert wrapper.info["tree count"] == 1
    assert wrapper.first_tree["type"] == "mol"
    assert len(wrapper.trees) == 1


def test_create_one_tree_of_molecules(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)

    wrapper = ReactionTreeWrapper(tree, exhaustive_limit=1)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 1

    assert wrapper.first_tree["smiles"] == tree["smiles"]
    assert len(wrapper.first_tree["children"]) == 2

    child_smiles = [child["smiles"] for child in wrapper.first_tree["children"]]
    expected_smiles = [node["smiles"] for node in tree["children"][0]["children"]]
    assert child_smiles == expected_smiles


def test_create_one_tree_of_reactions(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)

    wrapper = ReactionTreeWrapper(
        tree, content=TreeContent.REACTIONS, exhaustive_limit=1
    )

    assert wrapper.info["tree count"] == 1
    assert len(wrapper.trees) == 1

    rxn_nodes = []
    collect_smiles(tree, "reaction", rxn_nodes)
    assert wrapper.first_tree["smiles"] == rxn_nodes[0]
    assert len(wrapper.first_tree["children"]) == 0


def test_create_one_tree_of_everything(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)

    wrapper = ReactionTreeWrapper(tree, content=TreeContent.BOTH, exhaustive_limit=1)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 1

    mol_nodes = []
    collect_smiles(tree, "mol", mol_nodes)
    rxn_nodes = []
    collect_smiles(tree, "reaction", rxn_nodes)
    assert wrapper.first_tree["smiles"] == tree["smiles"]
    assert len(wrapper.first_tree["children"]) == 1

    child1 = wrapper.first_tree["children"][0]
    assert child1["smiles"] == rxn_nodes[0]
    assert len(child1["children"]) == 2

    child_smiles = [child["smiles"] for child in child1["children"]]
    assert child_smiles == mol_nodes[1:]


def test_create_all_trees_of_molecules(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)

    wrapper = ReactionTreeWrapper(tree)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 2

    mol_nodes = []
    collect_smiles(tree, "mol", mol_nodes)
    # Assert first tree
    assert wrapper.first_tree["smiles"] == mol_nodes[0]
    assert len(wrapper.first_tree["children"]) == 2

    child_smiles = [child["smiles"] for child in wrapper.first_tree["children"]]
    assert child_smiles == mol_nodes[1:]

    # Assert second tree
    assert wrapper.trees[1]["smiles"] == mol_nodes[0]
    assert len(wrapper.trees[1]["children"]) == 2

    child_smiles = [child["smiles"] for child in wrapper.trees[1]["children"]]
    assert child_smiles == mol_nodes[1:][::-1]


def test_create_two_trees_of_everything(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)

    wrapper = ReactionTreeWrapper(tree, content=TreeContent.BOTH)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 2

    mol_nodes = []
    collect_smiles(tree, "mol", mol_nodes)
    rxn_nodes = []
    collect_smiles(tree, "reaction", rxn_nodes)
    # Assert first tree
    assert wrapper.first_tree["smiles"] == mol_nodes[0]
    assert len(wrapper.first_tree["children"]) == 1

    child1 = wrapper.first_tree["children"][0]
    assert child1["smiles"] == rxn_nodes[0]
    assert len(child1["children"]) == 2

    child_smiles = [child["smiles"] for child in child1["children"]]
    assert child_smiles == mol_nodes[1:]

    # Assert second tree
    assert wrapper.trees[1]["smiles"] == mol_nodes[0]
    assert len(wrapper.trees[1]["children"]) == 1

    child1 = wrapper.trees[1]["children"][0]
    assert child1["smiles"] == rxn_nodes[0]
    assert len(child1["children"]) == 2

    child_smiles = [child["smiles"] for child in child1["children"]]
    assert child_smiles == mol_nodes[1:][::-1]


def test_route_self_distance(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    wrapper = ReactionTreeWrapper(tree, exhaustive_limit=1)

    assert wrapper.distance_to(wrapper) == 0.0


def test_route_distances_random(load_reaction_tree):
    tree1 = load_reaction_tree("example_routes.json", 0)
    wrapper1 = ReactionTreeWrapper(tree1, exhaustive_limit=1)
    tree2 = load_reaction_tree("example_routes.json", 1)
    wrapper2 = ReactionTreeWrapper(tree2, exhaustive_limit=1)

    distances = list(wrapper1.distance_iter(wrapper2, exhaustive_limit=1))

    assert len(distances) == 2
    assert pytest.approx(distances[0], abs=1e-2) == 2.6522


def test_route_distances_exhaustive(load_reaction_tree):
    tree1 = load_reaction_tree("example_routes.json", 0)
    wrapper1 = ReactionTreeWrapper(tree1, exhaustive_limit=2)
    tree2 = load_reaction_tree("example_routes.json", 1)
    wrapper2 = ReactionTreeWrapper(tree2, exhaustive_limit=2)

    distances = list(wrapper1.distance_iter(wrapper2, exhaustive_limit=40))

    assert len(distances) == 2
    assert pytest.approx(distances[0], abs=1e-2) == 2.6522
    assert pytest.approx(min(distances), abs=1e-2) == 2.6522


def test_route_distances_semi_exhaustive(load_reaction_tree):
    tree1 = load_reaction_tree("example_routes.json", 0)
    wrapper1 = ReactionTreeWrapper(tree1, exhaustive_limit=1)
    tree2 = load_reaction_tree("example_routes.json", 1)
    wrapper2 = ReactionTreeWrapper(tree2, exhaustive_limit=2)

    distances = list(wrapper1.distance_iter(wrapper2, exhaustive_limit=1))

    assert len(distances) == 2
    assert pytest.approx(distances[0], abs=1e-2) == 2.6522
    assert pytest.approx(min(distances), abs=1e-2) == 2.6522


def test_route_distances_longer_routes(load_reaction_tree):
    tree1 = load_reaction_tree("longer_routes.json", 0)
    wrapper1 = ReactionTreeWrapper(tree1, content="both")
    tree2 = load_reaction_tree("longer_routes.json", 1)
    wrapper2 = ReactionTreeWrapper(tree2, content="both")

    distances = list(wrapper1.distance_iter(wrapper2))

    assert len(distances) == 21
    assert pytest.approx(distances[0], abs=1e-2) == 4.14


def test_distance_matrix(load_reaction_tree):
    reaction_trees = [
        load_reaction_tree("example_routes.json", idx) for idx in range(3)
    ]

    dist_mat = distance_matrix(reaction_trees, content="molecules")

    assert len(dist_mat) == 3
    assert pytest.approx(dist_mat[0, 1], abs=1e-2) == 2.6522
    assert pytest.approx(dist_mat[0, 2], abs=1e-2) == 3.0779
    assert pytest.approx(dist_mat[2, 1], abs=1e-2) == 0.7483


def test_distance_matrix_timeout(load_reaction_tree):
    reaction_trees = [
        load_reaction_tree("example_routes.json", idx) for idx in range(3)
    ]

    with pytest.raises(ValueError):
        distance_matrix(reaction_trees, content="molecules", timeout=0)
