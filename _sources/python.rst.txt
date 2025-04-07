Python interface
================

This page gives a quick overview of how to use the Python classes and methods to compute
distances between routes.

Input format
------------

The routes or reaction tree are input as dictionaries that can be loaded from e.g. a JSON file.
The structure of the dictionary follows the output format used by AiZynthFinder, but only a small
number of fields are necessary. The structure is checked by the code, and if the validation fails,
an exception will be raised.

The structure definition and validation code can be found in the `route_distance.validation` module.

The input structure should be a the type `MoleculeNode` which is a dictionary with the following
fields

    * smiles - the SMILES string of the molecule represented by the node
    * type - should be the string "mol"
    * children - an *optional* list, containing at most one dictionary of type `ReactionNode`

The `ReactionNode` type is a dictionary with the following fields

    * type - should be the string "reaction"
    * children - a list of dictionaries of type `MoleculeNode`

It is easy to realize that this is a recursive definition. All extra fields in the dictionaries are ignored.
And example dictionary is shown below

.. code-block:: python

    {
        "smiles": "CCCCOc1ccc(CC(=O)N(C)O)cc1",
        "type": "mol",
        "children": [
            {
                "type": "reaction",
                "children": [
                    {
                        "smiles": "CCCCOc1ccc(CC(=O)Cl)cc1",
                        "type": "mol"
                    },
                    {
                        "smiles": "CNO",
                        "type": "mol"
                    }
                ]
            }
        ]
    }


Calculating TED
---------------

To compute the distance between two routes, we will first load them from a JSON file

.. code-block:: python

    import json
    with open("my_routes.json", "r") as fileobj:
        routes = json.load(fileobj)

Then we will create two wrapper objects that will take care of the calculations

.. code-block:: python

    from route_distances.ted.reactiontree import ReactionTreeWrapper
    wrapper1 = ReactionTreeWrapper(routes[0], content="both")
    wrapper2 = ReactionTreeWrapper(routes[1], content="both")

The argument `content` can be `molecules`, `reactions` or `both` and specify what nodes are included
in the calculation.

To calculate the distance between these two routes, we simple do

.. code-block:: python

    wrapper1.distance_to(wrapper2)

If we have many routes, a distance matrix can be calculate using

.. code-block:: python

    from route_distances.ted.distances import distance_matrix
    distance_matrix(routes, content="both")


The docstrings of all modules, classes and methods can be consulted :doc:`here <route_distances>`
