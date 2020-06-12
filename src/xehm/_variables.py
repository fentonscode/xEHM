# Define input / output dimensions
from .utils.serialisation import contains_required_keys
from typing import List

__all__ = ["Variable"]


class Variable:
    def __init__(self):
        self.name = ""
        self.min_support = 0.0
        self.max_support = 0.0
        self._required_keys = ["name", "min", "max"]

    def from_dict(self, description: dict):
        if not contains_required_keys(description, self._required_keys):
            raise ValueError(f"Invalid dictionary format\n"
                             f" - required keys: {','.join(self._required_keys)}\n"
                             f" - missing keys: {set(self._required_keys).difference(description.keys())}")

        self.name = description["name"]
        self.min_support = description["min"]
        self.max_support = description["max"]

    def to_dict(self) -> dict:
        return {"name": self.name, "min": self.min_support, "max": self.max_support}


def import_variables_from_file(f_name: str, format_string: str) -> List[Variable]:
    fmt_dict = \
        {
            "csv": import_variables_from_csv,
            "json": import_variables_from_json
        }

    if format_string not in fmt_dict:
        raise ValueError(f"Invalid format specified. Supported formats are {','.join(fmt_dict.keys())}")
    return fmt_dict[format_string](f_name)


def import_variables_from_csv(f_name: str) -> List[Variable]:
    raise NotImplementedError("TODO")


def import_variables_from_json(f_name: str) -> List[Variable]:
    raise NotImplementedError("TODO")