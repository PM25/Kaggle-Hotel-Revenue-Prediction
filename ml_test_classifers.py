#%%
from utils import *
from datapreprocessing import processing_data

from sklearn.model_selection import train_test_split

# start from here!
if __name__ == "__main__":
    (X_np, y_np), _ = processing_data()
    train_X, test_X, train_y, test_y = train_test_split(
        X_np, y_np, test_size=0.25, random_state=0
    )

    mlmodelwrapper = MLModelWrapper(train_X, train_y, test_X, test_y)
    mlmodelwrapper.test_classifiers()
