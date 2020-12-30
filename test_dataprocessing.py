from data_processing import Data

import numpy as np

data = Data("data/train.csv")

# test processing_test_data & processing give same output
def test_processing_test_data():
    X1, y1 = data.processing(["revenue"])
    X2 = data.processing_test_data("data/train.csv")
    assert np.array_equal(X1, X2)


# test features amount are same
def test_test_train_features():
    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data()
    assert X_train.shape[1] == X_test.shape[1]


# test use use_dummies
def test_use_dummies():
    dummies_data = Data("data/train.csv", use_dummies=True)
    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data()
    assert X_train.shape[1] == X_test.shape[1]

    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data("data/train.csv")
    assert np.array_equal(X_train, X_test)


def test_normalize():
    dummies_data = Data("data/train.csv", normalize=True)
    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data()
    assert X_train.shape[1] == X_test.shape[1]

    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data("data/train.csv")
    assert np.array_equal(X_train, X_test)


def test_use_dummies_normalize():
    dummies_data = Data("data/train.csv", use_dummies=True, normalize=True)
    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data()
    assert X_train.shape[1] == X_test.shape[1]

    X_train, y_train = data.processing(["revenue"])
    X_test = data.processing_test_data("data/train.csv")
    assert np.array_equal(X_train, X_test)


def test_train_test_split_by_date():
    X_df, y_df = data.processing()
    X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date()
    assert y_train_df.shape[1] == y_test_df.shape[1] == 1
    assert X_train_df.shape[0] + X_test_df.shape[0] == X_df.shape[0]
    assert X_df.shape[1] == X_train_df.shape[1] == X_test_df.shape[1]

