#%%
from utils import *
from datapreprocessing import Data
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingRegressor,
    BaggingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split


# test classifiers
data = Data()
X_np, y_np = data.processing("revenue", use_dummies=False, normalize=False)

eval_reg = RandomForestRegressor()
train_X, test_X, train_y, test_y = train_test_split(X_np, y_np, test_size=0.25)
eval_reg.fit(train_X, train_y)
print("reg", eval_reg.score(test_X, test_y))

reg = RandomForestRegressor()
reg.fit(X_np, y_np)
print(X_np.shape)

#%%
test_X_np = data.processing_test_data()
pred_revenue = reg.predict(test_X_np)

#%%
import pandas as pd

test_df = pd.read_csv("data/test.csv")
revenue = pd.DataFrame(pred_revenue)
revenue.columns = ["pred_revenue"]
results = pd.concat([test_df, revenue], axis=1)
results.to_csv("test_results.csv")


# # %% --------------------------
# test_data = Data("data/train.csv")

# test_X_np_is_canceled, _ = test_data.processing(
#     "is_canceled", use_dummies=False, normalize=False, test=False
# )
# print(test_X_np_is_canceled.shape)
# pred_is_canceled = clf.predict(test_X_np_is_canceled)

# #%%
# test_X_np, _ = test_data.processing(
#     "adr", use_dummies=False, normalize=False, test=False
# )
# pred_adr = reg.predict(test_X_np)
# print(test_X_np.shape)

# #%%
# import pandas as pd

# test_df = pd.read_csv("data/train.csv")
# adr = pd.DataFrame(pred_adr)
# adr.columns = ["pred_adr"]
# cancel = pd.DataFrame(pred_is_canceled)
# cancel.columns = ["pred_is_canceled"]
# results = pd.concat([test_df, adr, cancel], axis=1)
# results.to_csv("train_results.csv")

#%%
