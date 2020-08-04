from ..utils.plugin import ReturnState
from ..utils import print_kwargs

__all__ = ["ignore_diagnostics"]


# ignore_diagnostics - simply returns true to pass straight through the diagnostic stage
def ignore_diagnostics(**kwargs) -> (int, bool):
    if "debug_print" in kwargs and kwargs["debug_print"]:
        print_kwargs(**kwargs)
    return ReturnState.ok, True
