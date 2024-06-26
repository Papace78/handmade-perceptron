from handmade_perceptron import batchify
import numpy as np


def test_batchify():
    X = np.random.random((46, 5))
    y = np.random.random(46)
    X_batch, y_batch = next(batchify(X, y, batch_size=32))

    np.testing.assert_equal(X[:32], X_batch)
    np.testing.assert_equal(y[:32], y_batch)
