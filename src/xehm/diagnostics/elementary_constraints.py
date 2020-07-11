from typing import List
from ..utils import Plugin

__all__ = ["diagnostic_elementary_constraints"]


# Entry point
def diagnostic_elementary_constraints(**kwargs) -> List[Plugin]:
    return [diagnostic_function]


# This diagnostic uses the variable definitions to check the outputs are in the input support
def diagnostic_function(**kwargs):
    vars = kwargs["variables"]
    outs = kwargs["predicted_outputs"]
    for i, p in enumerate(outs):
        if p > vars[i].max_support or p < vars[i].min_support:
            return False
    return True
