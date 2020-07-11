from ..utils import Plugin, import_plugin
from typing import Union, List, Callable
import sys

__all__ = ["build_custom_diagnostic"]


#
# Builds a custom diagnostic plugin supplied by the user, this also uses a wrapper to inject
# default behaviour and tests
#
# By default the diagnostic is searched for in the xEHM API, current imported symbols, then the disk
#
def build_custom_diagnostic(func: Union[str, Plugin, Callable[..., List[Plugin]]], parent_ident: str = "__main__") \
        -> Plugin:
    import_exception_message: str = f"Could not import the diagnostic '{func}'"

    # If a function name was passed, then try to find it in the current scope / xehm library
    if isinstance(func, str):
        print(f"Importing a custom diagnostic plugin from {func}")

        # This re-fills the namespace with all the symbols
        import xehm.diagnostics

        # is it a function in the xEHM API?
        try:
            match = getattr(xehm.diagnostics, func)
            return build_custom_diagnostic(match)
        except AttributeError:
            pass

        # do we have the function in the current namespace?
        try:
            match = getattr(sys.modules[__name__], func)
            return build_custom_diagnostic(match)
        except AttributeError:
            pass

        # how about the __name__ namespace of the caller
        try:
            match = getattr(sys.modules[parent_ident], func)
            return build_custom_diagnostic(match)
        except AttributeError:
            pass

        # how about the __main__ namespace (e.g. if this was loaded in a script)
        try:
            match = getattr(sys.modules["__main__"], func)
            return build_custom_diagnostic(match)
        except AttributeError:
            pass

        # Try an import using module::function syntax
        if func.find("::") != -1:
            parts = func.split("::")
            func_name = parts[-1]
            func_module = "::".join(parts[0:-1])
        else:
            print(f"Plugin functions must be specified using the {func}::your_function syntax")
            raise ImportError(import_exception_message)

        module = import_plugin(func_module)

        if module is None:
            # cannot find the code
            print(f"Cannot find the diagnostic '{func}'. Please check the path and spelling")
            raise ImportError(import_exception_message)

        else:
            if hasattr(module, func_name):
                return build_custom_diagnostic(getattr(module, func_name))
            print(f"Could not find the function '{func_name}' in the module '{func_module}'. Check that the spelling "
                  f"is correct and that the right version of the module is being loaded.")
            raise ImportError(import_exception_message)

    # Check for a callable attribute
    if not callable(func):
        print(f"Cannot import {func} as it is not recognised as a function.")
        raise ValueError(f"Custom diagnostic '{func}' cannot be called as a function")

    print(f"Building a custom extractor for {func}")
    built_code = lambda **kwargs: wrapper(func=func, **kwargs)
    return built_code


# Wrapper for a custom diagnostic to allow injecting any default behaviour
# func should be a custom plugin that returns a list of functions to run
def wrapper(func: Callable[..., List[Plugin]], **kwargs) -> List[Plugin]:
    if func is None:
        raise NotImplementedError("No default behaviour implemented, a function must be supplied")
    return func(**kwargs)
