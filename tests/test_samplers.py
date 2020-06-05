import pytest
from xehm.sampling import Sampler
from xehm import Tree2D


# Sampler is an abstract base, we shouldn't be able to create it
def test_sampler_constructor():
    sampler_test_dist = Tree2D()
    with pytest.raises(Exception) as exception_info:
        test_obj = Sampler(sampler_test_dist)


if __name__ == '__main__':
    test_sampler_constructor()