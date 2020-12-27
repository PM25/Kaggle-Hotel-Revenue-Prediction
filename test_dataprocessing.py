from data_processing import Data

import numpy as np

data = Data("data/train.csv")


def test_eat():
    X1, y1 = data.processing("adr")
    X2 = data.processing_test_data("data/train.csv")
    assert np.array_equal(X1, X2)
