import pytest

from route_distances.lstm.features import (
    add_fingerprints,
    remove_reactions,
    preprocess_reaction_tree,
)


@pytest.fixture
def toy_tree():
    return {
        "smiles": "CCn1nc(CC(C)C)cc1C(=O)NCc1c(C)cc(C)nc1OC",
        "type": "mol",
        "children": [
            {
                "smiles": "dummy",
                "type": "reaction",
                "children": [
                    {
                        "smiles": "CCn1nc(CC(C)C)cc1C(=O)O",
                        "type": "mol",
                    },
                    {
                        "smiles": "COc1nc(C)cc(C)c1CN",
                        "type": "mol",
                    },
                ],
            }
        ],
    }


def test_remove_reactions(toy_tree):
    assert len(toy_tree["children"]) == 1

    new_tree = remove_reactions(toy_tree)

    assert len(new_tree["children"]) == 2


def test_add_fingerprints(toy_tree):
    new_tree = remove_reactions(toy_tree)

    add_fingerprints(new_tree, nbits=10)

    assert len(new_tree["fingerprint"]) == 10
    assert list(new_tree["fingerprint"]) == [1] * 10
    assert list(new_tree["children"][0]["fingerprint"]) == [1] * 10


def test_preprocessing(toy_tree):

    output = preprocess_reaction_tree(toy_tree, nfeatures=10)

    assert len(output["features"]) == 3
    assert list(output["node_order"]) == [1, 0, 0]
    assert list(output["edge_order"]) == [1, 1]
    assert list(output["adjacency_list"][0]) == [0, 1]
    assert list(output["adjacency_list"][1]) == [0, 2]
    assert output["num_nodes"] == 3
