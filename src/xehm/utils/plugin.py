#
# Python plugin system
#
# Usage:
# if obj is None:
#     obj = xehm.type.default
# else:
#     obj = build_plugin(obj)
#

from typing import Callable
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
import os

__all__ = ["Plugin", "import_plugin"]

# Define a plugin as a function with variable arguments that returns a bool
Plugin = Callable[...,  bool]


def import_plugin(module: str):
    """This will try to import the passed module. This will return
       the module if it was imported, or will return 'None' if
       it should not be imported.

       Parameters
       ----------
       module: str
         The name of the module to import
    """
    try:
        m = import_module(module)
    except SyntaxError as e:
        print(
            f"\nSyntax error when importing {module}\n"
            f"{e.__class__.__name__}:{e}\n"
            f"Line {e.lineno}.{e.offset}:{(e.offset-1)*' '} |\n"
            f"Line {e.lineno}.{e.offset}:{(e.offset-1)*' '}\\|/\n"
            f"Line {e.lineno}.{e.offset}: {e.text}")
        m = None
    except ImportError:
        # this is ok and expected if the module is in a python file
        # that will be loaded below
        m = None
    except Exception:
        # something else went wrong
        m = None

    # Try to load from a *.py file
    if m is None:
        try:
            # Try to find with no extension and py extension
            if os.path.exists(module):
                pyfile = module
            elif os.path.exists(f"{module}.py"):
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
                f"Line {e.lineno}.{e.offset}:{(e.offset-1)*' '} |\n"
                f"Line {e.lineno}.{e.offset}:{(e.offset-1)*' '}\\|/\n"
                f"Line {e.lineno}.{e.offset}: {e.text}")
        except Exception as e:
            print(
                f"\nError when importing {module}\n"
                f"{e.__class__.__name__}: {e}\n")
            m = None

    if m is not None:
        print(f"IMPORT {m}")

    return m
