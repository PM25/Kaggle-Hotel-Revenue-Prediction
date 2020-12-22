# %%
from utils import *
from datapreprocessing import Data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

#%%
X_np, y_np = Data().processing(target="adr", use_dummies=False)
print(f"X_np's shape: {X_np.shape}")
print(f"y_np's shape: {y_np.shape}")


train_loader, val_loader, test_loader = LoadData(
    X_y=(X_np, y_np), X_y_dtype=("float", "float")
).get_dataloader([0.7, 0.2, 0.1], batch_size=64)


# %% start from here!
if __name__ == "__main__":
    # setting
    model = Input1DModelSimplified(X_np.shape[1], 1)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=50)

    # evaluate the model
    modelwrapper.regression_report(test_loader)
