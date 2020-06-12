# Define input / output dimensions
from .utils.serialisation import contains_required_keys

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
