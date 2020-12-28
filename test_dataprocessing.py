from data_processing import Data

import numpy as np

data = Data("data/train.csv")


def test_processing_test_data():
    X1, y1 = data.processing("revenue")
    X2 = data.processing_test_data("data/train.csv")
    assert np.array_equal(X1, X2)


def test_processing_test_data():
    X1, y1 = data.processing("revenue")
    X2 = data.processing_test_data()
    assert X1.shape == X2.shape
