#
# Generic emulator interface to load in pre-built or our own modules
#
from numpy import ndarray


# TODO: Complete this
class Emulator:
    def __init__(self):
        if type(self) is Emulator:
            raise Exception("Emulator is an abstract base, inherit and define your own")
        self._design_inputs = []
        self._design_outputs = []
        self.ident = "Abstract emulator"

    def train(self, inputs: ndarray, outputs: ndarray, parameter_defaults=None):
        raise NotImplementedError("Custom emulators must define a build function")

    def evaluate(self, points):
        raise NotImplementedError("Custom emulators must define an evaluation function")

    def validate(self, inputs: ndarray, outputs: ndarray, diag):
        raise NotImplementedError("Custom emulators must define a validate function")