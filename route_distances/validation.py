""" Module containing routes to validate AiZynthFinder-like input dictionaries """
from __future__ import annotations
from typing import Optional, List
import pydantic

from route_distances.utils.type_utils import StrDict


class MoleculeNode(pydantic.BaseModel):
    """Node representing a molecule"""

    smiles: str
    type: pydantic.constr(regex=r"^mol$")
    children: Optional[pydantic.conlist(ReactionNode, min_items=1, max_items=1)]


class ReactionNode(pydantic.BaseModel):
    """Node representing a reaction"""

    type: pydantic.constr(regex=r"^reaction$")
    children: List[MoleculeNode]


MoleculeNode.update_forward_refs()


def validate_dict(dict_: StrDict) -> None:
    """
    Check that the route dictionary is a valid structure

    :param dict_: the route as dictionary
    """
    try:
        MoleculeNode(**dict_, extra=pydantic.Extra.ignore)
    except pydantic.ValidationError as err:
        raise ValueError(f"Invalid input: {err.json()}")
