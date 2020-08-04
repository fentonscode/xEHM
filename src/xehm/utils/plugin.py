#
# Python plugin system
#

from typing import Any, Callable, List, Tuple, Union
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
from types import ModuleType
from os.path import exists
from sys import modules
from enum import IntEnum

__all__ = ["Plugin", "import_plugin", "build_custom_plugin", "ReturnState"]

# Plugins are functions that take variable arguments but return status codes and data
Plugin = Callable[..., Tuple[int, Any]]


# PEP 435 Says do this, check this is right
class ReturnState(IntEnum):
    ok = 1
    fail_stop = 0
    fail_ignore = -1
    fail_retry = -2


# Search for the function in the pre-defined API list if the user has provided one
# This allows code to be re-used from large packages as well imported from basic scripts
#
# If a package string is supplied then only that package will be searched
#   - e.g.: xehm.diagnostics searches only the diagnostics API
#
# If a package string is not supplied then the subpackages of api_ident will be searched one level down
#   - NOTE: Any module imports that fail will be skipped
#
def search_in_api(plugin_name: str, package: str = None, api_ident: str = "xehm"):
    match = None
    search_modules = []

    # If there is no package to filter by, load the top level package and check one level down
    # (this API is designed to store customisable units as sub-packages)
    if package is None:
        top_level_package = import_module(api_ident)
        potential_subpackages: List[str] = [s for i, s in enumerate(dir(top_level_package))
                                            if isinstance(getattr(top_level_package, s), ModuleType)]
        for sub in potential_subpackages:
            try:
                module = import_module(f"{api_ident}.{sub}")
                search_modules.append(module)
            except ImportError as e:
                print(f"Unable to load module {e}. It will not be searched")
                pass
    else:
        search_modules.append(import_module(package))

    for test in search_modules:
        try:
            match = getattr(test, plugin_name)
            return match
        except AttributeError:
            pass

    return match


# Try to import a plugin from various sources, errors will return None
def import_plugin(module: str):
    try:
        m = import_module(module)
    except SyntaxError as e:
        print(f"\nSyntax error when importing {module}\n"
              f"{e.__class__.__name__}:{e}\n"
              f"Line {e.lineno}.{e.offset}:{(e.offset - 1) * ' '} |\n"
              f"Line {e.lineno}.{e.offset}:{(e.offset - 1) * ' '}\\|/\n"
              f"Line {e.lineno}.{e.offset}: {e.text}")
        m = None
    except ImportError:
        # this is ok and expected if the module is in a python file
        m = None
    except Exception:
        # something else went wrong
        m = None

    # Try to load from a *.py file
    if m is None:
        try:
            # Try to find with no extension and py extension
            if exists(module):
                pyfile = module
            elif exists(f"{module}.py"):
                pyfile = f"{module}.py"
            else:
                pyfile = None

            if pyfile:
                from pathlib import Path as _Path
                module = _Path(pyfile).stem
                spec = spec_from_file_location(module, pyfile)
                if spec is None:
                    raise ImportError(f"Cannot build a spec for the module from the file {pyfile}")

                m = module_from_spec(spec)
                spec.loader.exec_module(m)
                print(f"Loaded {module} from {pyfile}")

        except SyntaxError as e:
            print(
                f"\nSyntax error when reading {pyfile}\n"
                f"{e.__class__.__name__}:{e}\n"
                f"Line {e.lineno}.{e.offset}:{(e.offset - 1) * ' '} |\n"
                f"Line {e.lineno}.{e.offset}:{(e.offset - 1) * ' '}\\|/\n"
                f"Line {e.lineno}.{e.offset}: {e.text}")
        except Exception as e:
            print(
                f"\nError when importing {module}\n"
                f"{e.__class__.__name__}: {e}\n")
            m = None

    if m is not None:
        print(f"IMPORT {m}")

    return m


#
# Builds a custom plugin supplied by the user and wraps it into a callable Python function
#
# By default the priority order is:
#   - 1) Pre-defined functions in the xEHM API
#   - 2) Current imported symbols
#   - 3) From the disk
#   - 4) Direct wrapping of the object supplied as a parameter
#
def build_custom_plugin(plugin_name: Union[str, Plugin], parent_ident: str = "__main__", **build_parameters) -> Plugin:
    import_exception_message: str = f"Could not import the plugin '{plugin_name}'"

    # If a function name was passed, then try to find it in the current scope / xehm library
    if isinstance(plugin_name, str):
        print(f"Importing a custom plugin from {plugin_name}")

        # Search for the function in the pre-defined API list if the user has provided one
        match = search_in_api(plugin_name)
        if match is not None:
            return build_custom_plugin(match, **build_parameters)

        # do we have the function in the current namespace?
        try:
            match = getattr(modules[__name__], plugin_name)
            return build_custom_plugin(match, **build_parameters)
        except AttributeError:
            pass

        # how about the __name__ namespace of the caller
        try:
            match = getattr(modules[parent_ident], plugin_name)
            return build_custom_plugin(match, **build_parameters)
        except AttributeError:
            pass

        # how about the __main__ namespace (e.g. if this was loaded in a script)
        try:
            match = getattr(modules["__main__"], plugin_name)
            return build_custom_plugin(match, **build_parameters)
        except AttributeError:
            pass

        # Try an import using module::function syntax
        if plugin_name.find("::") != -1:
            parts = plugin_name.split("::")
            func_name = parts[-1]
            func_module = "::".join(parts[0:-1])
        else:
            print(f"Plugin functions must be specified using the {plugin_name}::your_function syntax")
            raise ImportError(import_exception_message)

        module = import_plugin(func_module)

        if module is None:
            # cannot find the code
            print(f"Cannot find the plugin '{plugin_name}'. Please check the path and spelling")
            raise ImportError(import_exception_message)

        else:
            if hasattr(module, func_name):
                return build_custom_plugin(getattr(module, func_name), **build_parameters)
            print(f"Could not find the function '{func_name}' in the module '{func_module}'. Check that the spelling "
                  f"is correct and that the right version of the module is being loaded.")
            raise ImportError(import_exception_message)

    # Check for a callable attribute
    if not callable(plugin_name):
        print(f"Cannot import {plugin_name} as it is not recognised as a function.")
        raise ValueError(f"Custom plugin '{plugin_name}' cannot be called as a function")

    print(f"Building a custom plugin for {plugin_name}")
    built_code = lambda **kwargs: wrapper(func=plugin_name, **kwargs)
    return built_code


# Wrapper for a custom plugin to allow injecting any default behaviour
# func should be a custom plugin that returns a function to run
def wrapper(func: Plugin, **kwargs) -> Tuple[int, Any]:
    if func is None:
        raise NotImplementedError("No default behaviour implemented, a function must be supplied")
    print(f"Calling custom plugin function {func}")
    return func(**kwargs)
