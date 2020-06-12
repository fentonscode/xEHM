import xehm.utils.hypercubes as hyper
import numpy as np


# TODO: Test could be more thorough?
def test_point_in_box():

    # Generate a random set on [0.0, 1.0]
    x1 = np.random.uniform(low=0.0, high=1.0, size=(1000, 100))

    # Check that point_in_box validates when using the right limits
    box_limits = np.broadcast_to(np.asarray([[0.0], [1.0]]), (2, x1.shape[1]))
    assert hyper.point_in_box(x1, box_limits)

    # Move the limits and verify that all point_in_box fails
    box_limits = np.broadcast_to(np.asarray([[-1.0], [-0.5]]), (2, x1.shape[1]))
    assert not hyper.point_in_box(x1, box_limits)

    # Generate a set for unequal limits
    x2_a = np.random.uniform(low=-1.0, high=1.0, size=(1000, 1))
    x2_b = np.random.uniform(low=0.0, high=0.5, size=(1000, 1))
    x2_c = np.random.uniform(low=-10.0, high=-5.0, size=(1000, 1))
    x2 = np.c_[x2_a, x2_b, x2_c]

    # Check that point_in_box validates when using the right limits
    box_limits = np.broadcast_to(np.asarray([[-1.0, 0.0, -10.0], [1.0, 0.5, 10.0]]), (2, x2.shape[1]))
    assert hyper.point_in_box(x2, box_limits)

    # Move the limits and verify that all point_in_box fails
    box_limits = np.broadcast_to(np.asarray([[-5.0, 0.6, -1.0], [-2.0, 0.7, 1.0]]), (2, x2.shape[1]))
    assert not hyper.point_in_box(x2, box_limits)


# Test uniform_box by creating vectors and checking they are in the right place
def test_uniform_box():

    # Define a 5D hypercube on [-1, 1] and ensure points are valid
    num_dims = 5
    num_points = 1000
    limits = np.broadcast_to(np.asarray([[-1.0], [1.0]]), (2, num_dims))

    # Check creating multiple rows
    cube = hyper.uniform_box(limits, num_points)
    assert cube.shape[0] == num_points
    assert cube.shape[1] == num_dims
    assert hyper.point_in_box(cube, limits)


def test_scale_matrix_columns():
    matrix1 = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    matrix2 = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    matrix3 = np.asarray([[1, 0, -1], [2, 0, -1], [0, -1, 0]])

    col_scales = [1, 2, -1]


def test_transform_minus_one_one():
    pass


if __name__ == '__main__':
    test_point_in_box()
    test_uniform_box()
    test_transform_minus_one_one()
