import json

import pytest


@pytest.fixture
def load_reaction_tree(shared_datadir):
    def wrapper(filename, index=0):
        filename = str(shared_datadir / filename)
        with open(filename, "r") as fileobj:
            trees = json.load(fileobj)
        if isinstance(trees, dict):
            return trees
        elif index == -1:
            return trees
        else:
            return trees[index]

    return wrapper
