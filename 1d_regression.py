# %%
from utils import *
from datapreprocessing import Data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

#%%
data = Data()
X_np, y_np = data.processing("revenue", use_dummies=False, normalize=False)
print(f"X_np's shape: {X_np.shape}")
print(f"y_np's shape: {y_np.shape}")

reg = BaggingRegressor()
train_x, test_x, train_y, test_y = train_test_split(X_np, y_np, test_size=0.25)
reg.fit(train_x, train_y)
score = reg.score(test_x, test_y)
print(score)

#%%
X = data.processing_test_data()
pred = reg.predict(X)
test_df = pd.read_csv("data/test.csv", index_col="ID")
a = pd.DataFrame(pred)
a.columns = ["pred"]
print(test_df.shape)
print(a.shape)
result = pd.concat([test_df, a], axis=1)

# print(result)
#%%
result.to_csv("test_result.csv")


#%%
train_loader, val_loader, test_loader = LoadData(
    X_y=(X_np, y_np), X_y_dtype=("float", "float")
).get_dataloader([0.7, 0.2, 0.1], batch_size=64)


# %% start from here!
if __name__ == "__main__":
    # setting
    model = Input1DModel(X_np.shape[1], 1)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=50)

    # evaluate the model
    modelwrapper.regression_report(test_loader)

# %%
