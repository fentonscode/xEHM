#
# This is a wrapper for simulators which need to be called
#
# This is here because people will want to do things that aren't in Python and have it all work
#


#
# Simulators are in three various types
#   - Python code that should be executed
#   - External code that should be executed from a shell command
#   - External runs that have already been completed and must be loaded in
#
class Simulator:
    def __init__(self):
        pass

    def evaluate(self):
        raise NotImplementedError("Custom simulators must define an evaluate function")

