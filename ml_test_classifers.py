#%%
from utils import *
from data_processing import Data

# start from here!
if __name__ == "__main__":
    data = Data(use_dummies=False, normalize=False)
    # test classifiers
    X_df, y_df = data.processing(["is_canceled"])
    mlmodelwrapper = MLModelWrapper(X_df.to_numpy(), y_df.to_numpy())
    mlmodelwrapper.quick_test("classifier")

    # test regressors
    X_df, y_df = data.processing(["adr"])
    mlmodelwrapper = MLModelWrapper(X_df.to_numpy(), y_df.to_numpy())
    mlmodelwrapper.quick_test("regressor")

    X_df, y_df = data.processing(["revenue"])
    mlmodelwrapper = MLModelWrapper(X_df.to_numpy(), y_df.to_numpy())
    mlmodelwrapper.quick_test("regressor")
