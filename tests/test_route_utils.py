from route_distances.utils.routes import (
    calc_depth,
    calc_llr,
    extract_leaves,
    is_solved,
    route_score,
    route_scorer,
    route_ranks,
)


def remove_in_stock(tree_dict):
    if "in_stock" in tree_dict:
        del tree_dict["in_stock"]
    for child in tree_dict.get("children", []):
        remove_in_stock(child)


def test_route_depth(load_reaction_tree):
    routes = load_reaction_tree("example_routes.json", index=-1)

    assert calc_depth(routes[0]) == 2
    assert calc_depth(routes[1]) == 4

    assert calc_llr(routes[0]) == 1
    assert calc_llr(routes[1]) == 2


def test_route_leaves(load_reaction_tree):
    route = load_reaction_tree("example_routes.json", index=0)

    assert extract_leaves(route) == {
        "Cc1ccc2nc3ccccc3c(Cl)c2c1",
        "Nc1ccc(NC(=S)Nc2ccccc2)cc1",
    }


def test_route_solved(load_reaction_tree):
    route = load_reaction_tree("example_routes.json", index=0)

    assert is_solved(route)


def test_route_not_solved(load_reaction_tree):
    route = load_reaction_tree("example_routes.json", index=0)
    route["children"][0]["children"][0]["in_stock"] = False

    assert not is_solved(route)


def test_route_solved_unspec(load_reaction_tree):
    route = load_reaction_tree("example_routes.json", index=0)
    remove_in_stock(route)

    assert is_solved(route)


def test_route_score(load_reaction_tree):
    routes = load_reaction_tree("example_routes.json", index=-1)

    assert route_score(routes[0]) == 3.5
    assert route_score(routes[1]) == 6.625


def test_route_score_unsolved(load_reaction_tree):
    route = load_reaction_tree("example_routes.json", index=0)
    route["children"][0]["children"][0]["in_stock"] = False

    assert route_score(route) == 14.75


def test_route_score_unspec(load_reaction_tree):
    route = load_reaction_tree("example_routes.json", index=0)
    remove_in_stock(route)

    assert route_score(route) == 3.5


def test_route_scorer(load_reaction_tree):
    routes = load_reaction_tree("example_routes.json", index=-1)
    routes2 = [routes[2], routes[0], routes[1]]

    sorted_routes, route_scores = route_scorer(routes2)

    assert route_scores == [3.5, 6.625, 6.625]
    assert sorted_routes[0] == routes[0]
    assert sorted_routes[1] == routes[2]
    assert sorted_routes[2] == routes[1]


def test_route_rank():

    assert route_ranks([4.0, 5.0, 5.0]) == [1, 2, 2]
    assert route_ranks([4.0, 4.0, 5.0]) == [1, 1, 2]
    assert route_ranks([4.0, 5.0, 6.0]) == [1, 2, 3]
    assert route_ranks([4.0, 5.0, 5.0, 6.0]) == [1, 2, 2, 3]
