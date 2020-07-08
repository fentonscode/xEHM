# Define input / output dimensions
from .utils.serialisation import contains_required_keys
from typing import List, Tuple

__all__ = ["Variable", "make_variable_set"]


class Variable:
    def __init__(self, name="null", minimum=0.0, maximum=1.0):
        self.name = name
        self.min_support = minimum
        self.max_support = maximum
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


#
# make_variable_set: Creates a set of named variable definitions based on the min/max limits
#
# To prevent some silent errors, LBYL is employed here
#
def make_variable_set(v_count: int, prefix: str = "x", min_value: List[float] = None,
                      max_value: List[float] = None) -> Tuple[Variable]:
    if min_value is None:
        min_value = [0.0] * v_count
    if max_value is None:
        max_value = [1.0] * v_count
    if isinstance(min_value, float):
        min_value = [min_value] * v_count
    if isinstance(max_value, float):
        max_value = [max_value] * v_count
    if not (len(min_value) == len(max_value) == v_count):
        raise ValueError("min_value and max_value must be the same length and contain v_count elements")

    if v_count == 1:
        return tuple(Variable(name=f"{prefix}", minimum=min_value[0], maximum=max_value[0]))
    return tuple(Variable(name=f"{prefix}{v + 1}", minimum=v_min, maximum=v_max)
                 for v, v_min, v_max in zip(range(v_count), min_value, max_value))
