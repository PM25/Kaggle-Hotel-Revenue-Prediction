#%%
from utils import *
from data_processing import Data

# start from here!
if __name__ == "__main__":
    # test classifiers
    data = Data()
    X_df, y_df = data.processing("is_canceled", use_dummies=False, normalize=False)
    mlmodelwrapper = MLModelWrapper(X_np, y_np)
    mlmodelwrapper.quick_test("classifier")

    # test regressors
    data = Data()
    X_df, y_df = data.processing("adr", use_dummies=False, normalize=False)
    mlmodelwrapper = MLModelWrapper(X_np, y_np)
    mlmodelwrapper.quick_test("regressor")

    data = Data()
    X_np, y_np = data.processing("revenue", use_dummies=False, normalize=False)
    mlmodelwrapper = MLModelWrapper(X_np, y_np)
    mlmodelwrapper.quick_test("regressor")
