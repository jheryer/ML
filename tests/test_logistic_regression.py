import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.testing._private.utils import assert_equal
from logistic_regression import initalize_with_zeros, propagate, sigmoid


class LogisticRegressionTest(unittest.TestCase):
    def test_sigmoid(self):
        x = np.array([0, 2])
        result = np.array([0.5, 0.88079708])
        value = sigmoid(x)
        assert_array_almost_equal(value, result)

    def test_initalize_zero(self):
        dim = 2
        w, b = initalize_with_zeros(dim)
        assert_array_almost_equal(w, np.zeros((dim, 1)))
        assert_equal(b, 0.0)

    def test_propagate(self):
        w = np.array([[1.0], [2.0]])
        b = 2.0
        X = np.array([[1.0, 2.0, -1.0], [3.0, 4.0, -3.2]])
        Y = np.array([[1, 0, 1]])
        grads, cost = propagate(w, b, X, Y)

        assert type(grads["dw"]) == np.ndarray
        assert grads["dw"].shape == (2, 1)
        assert type(grads["db"]) == np.float64

        assert_equal(cost, 5.801545319394553)
        assert_array_almost_equal(grads["dw"], np.array([[0.99845601], [2.39507239]]))
        assert_equal(grads["db"], 0.001455578136784208)
