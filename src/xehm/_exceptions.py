class GenericError(Exception):
    def __init__(self, message: str = "xEHM: An unhandled exception has been raised"):
        super().__init__(message)
        self.ident = "xEHM Generic Error object"


class SimulationFailure(GenericError):
    def __init__(self, message: str = "xEHM: The simulator has failed"):
        super().__init__(message)
        self.ident = "xEHM Simulation failure object"
