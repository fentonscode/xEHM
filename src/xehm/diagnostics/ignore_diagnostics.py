from ..utils import Plugin, print_kwargs
from typing import List

__all__ = ["ignore_diagnostics"]


def ignore_diagnostics(**kwargs) -> List[Plugin]:
    return [null_function]


# ignore_diagnostics - simply returns true to pass straight through the diagnostic stage
def null_function(**kwargs) -> bool:
    if "debug_print" in kwargs and kwargs["debug_print"]:
        print_kwargs(**kwargs)
    return True
