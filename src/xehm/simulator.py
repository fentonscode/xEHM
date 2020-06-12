#
# This is a wrapper for simulators which need to be called
#
# This is here because people will want to do things that aren't in Python and have it all work
#

class Simulator:
    def __init__(self):
        pass

    def evaluate(self):
        raise NotImplementedError("Custom simulators must define an evaluate function")
