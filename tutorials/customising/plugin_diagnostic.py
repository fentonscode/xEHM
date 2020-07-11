from xehm.utils import Plugin
from typing import List


# Null diagnostic suite - invokes ignore diagnostics
def diagnostic_none(**kwargs) -> List[Plugin]:
    if "debug_print" in kwargs:
        if kwargs["debug_print"]:
            k_string = '\n'.join(str(kwargs).strip('{}').split(','))
            print(f"Calling diagnostic_none with parameters:\n\n{k_string}")
    return [ignore_diagnostics]


# ignore_diagnostics - simply returns true to pass straight through the diagnostic stage
def ignore_diagnostics(**kwargs) -> bool:
    if "debug_print" in kwargs:
        if kwargs["debug_print"]:
            print("ignore_diagnostics: passing through diagnostic stage")
    return True
