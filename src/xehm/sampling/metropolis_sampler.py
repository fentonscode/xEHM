import numpy as np
from .samplers import Sampler
from .._distributions import Distribution
from .._distributions import Proposal
from .._distributions import UniformStepProposal


# Metropolis sampling
#
# Implements the Metropolis algorithm which forms a Markov Chain based on accepting ratios of probabiities
#
#
class MetropolisSampler(Sampler):
    def __init__(self, pdf: Distribution, proposal: Proposal):
        super().__init__(pdf)
        if proposal is None:
            self.proposal = UniformStepProposal(pdf.num_dimensions, pdf.support_limits)
        else:
            self.proposal = proposal
        self.ident = "Metropolis Sampler"

    def run(self, num_samples: int, params):
        initial: np.ndarray = np.asarray(params[0])
        self.last_run, self.acceptance = np.asarray(_metropolis_fast(self.distribution.probability,
                                                                       self.proposal.random_draw,
                                                                       self.distribution.num_dimensions,
                                                                       num_samples, initial))
        self.acceptance /= num_samples
        return self


def _metropolis_fast(pdf, proposal, ndims: int, nsamples: int, initial):
    chain = np.zeros((nsamples, ndims))
    chain_x = initial
    accept: int = 0
    # Reduce the number of calls to pdf by calculating now and only if we move (~ 20% speedup)
    old_p = pdf(chain_x)  # speedup 2
    # Trade some memory for speed (~15/20%, possible proportional to acceptance ratio)
    rand_u = np.random.uniform(0.0, 1.0, nsamples)  # speedup 1
    for i in range(nsamples):
        # Get new data
        new_x = proposal(chain_x)
        new_p = pdf(new_x)
        if new_p > 0.0:
            # Only progress the chain if the proposal is valid
            if old_p > 0.0:
                acceptance = new_p / old_p
            else:
                # if the old point was out of bounds, then accept new point
                acceptance = 1.0
            if rand_u[i] < acceptance:
                chain_x = new_x
                old_p = new_p  # Only call if we move, speedup 3: assign instead (7%)
                accept += 1
        chain[i] = chain_x
    return chain, accept


# Some pre-baked proposals for metropolis samplers (faster batching)
#
# Uniform random walking
#
def _metropolis_uniform(pdf, step, ndims: int, nsamples: int, initial):
    chain = np.zeros((nsamples, ndims))
    chain_x = initial
    accept: int = 0
    walks = np.multiply(np.random.uniform(-1.0, 1.0, (nsamples, ndims)), step)
    old_p = pdf(chain_x)
    rand_u = np.random.uniform(0.0, 1.0, nsamples)
    for i in range(nsamples):
        # Get new data
        new_x = chain_x + walks[i]
        new_p = pdf(new_x)
        if new_p > 0.0:
            # Only progress the chain if the proposal is valid
            if old_p > 0.0:
                acceptance = new_p / old_p
            else:
                # if the old point was out of bounds, then accept new point
                acceptance = 1.0
            if rand_u[i] < acceptance:
                chain_x = new_x
                old_p = new_p
                accept += 1
        chain[i] = chain_x
    return chain, accept


# Uniform random walker with unequal step sizes per dimension
def _metropolis_uniform_multi(pdf, mins, maxs, ndims: int, nsamples: int, initial):
    chain = np.zeros((nsamples, ndims))
    chain_x = initial
    accept: int = 0
    walks = np.asarray([np.random.uniform(low=mins[i], high=maxs[i], size=nsamples)
                        for i in range(ndims)]).transpose().squeeze()
    old_p = pdf(chain_x)
    rand_u = np.random.uniform(0.0, 1.0, nsamples)
    for i in range(nsamples):
        # Get new data
        new_x = chain_x + walks[i]
        new_p = pdf(new_x)
        if new_p > 0.0:
            # Only progress the chain if the proposal is valid
            if old_p > 0.0:
                acceptance = new_p / old_p
            else:
                # if the old point was out of bounds, then accept new point
                acceptance = 1.0
            if rand_u[i] < acceptance:
                chain_x = new_x
                old_p = new_p
                accept += 1
        chain[i] = chain_x
    return chain, accept
