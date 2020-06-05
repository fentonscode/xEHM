from .._distributions import Distribution

__all__ = ["Sampler"]


# Abstract sampler code: This object should never be created directly (inherit from this)
class Sampler:

    # All samplers must operate over a distribution
    def __init__(self, pdf: Distribution):
        if type(self) is Sampler:
            raise Exception("Sampler is an abstract base, inherit and define your own")
        if pdf is None:
            raise ValueError("A distribution must be provided to sample from")
        self.distribution = pdf
        self.last_run = None
        self.acceptance = 0.0
        self.ident = "Abstract Sampler"

    # Run the sample algorithm with a target number of samples and parameters
    def run(self, num_samples: int, params):
        raise NotImplementedError("Custom samplers must implement a run function")
