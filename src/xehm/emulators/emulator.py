#
# Generic emulator interface to load in pre-built or our own modules
#


# TODO: Complete this
class Emulator:
    def __init__(self):
        if type(self) is Emulator:
            raise Exception("Emulator is an abstract base, inherit and define your own")

    def train(self, inputs, outputs, parameter_defaults):
        raise NotImplementedError("Custom emulators must define a build function")

    def evaluate(self, points):
        raise NotImplementedError("Custom emulators must define an evaluation function")