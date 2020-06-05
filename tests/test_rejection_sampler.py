from xehm import sampling
from xehm import Tree2D


def test_rejection_sampler():

    dist = Tree2D()
    results = sampling.RejectionSampler(dist).run(10000, [15.0])

    assert results.acceptance <= 1.0
    assert len(results.last_run) <= 10000
    assert results.last_run.shape[1] == 2


if __name__ == '__main__':
    test_rejection_sampler()