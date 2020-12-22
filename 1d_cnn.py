# %%
from utils import *
from datapreprocessing import Data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%%
data = Data()
X, y = data.processing_1d_cnn()

loaddata = LoadData(X_y=(X, y), X_y_dtype=("float", "float"))
train_loader, val_loader, test_loader = loaddata.get_dataloader([0.75, 0.1, 0.1])


class Model(nn.Module):
    def __init__(self, sz, output):
        super().__init__()
        self.conv1 = nn.Conv1d(sz[0], sz[1], 5)
        self.fc1 = nn.Linear(86016, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 86016)
        x = self.fc1(x)
        return x


# %% start from here!
if __name__ == "__main__":
    loss_func = nn.MSELoss()
    for data in train_loader:
        x, y = data
        sz = x.shape
        pred = model(x.cuda())
        print(pred.shape)
        loss = loss_func(y, pred)
        print(loss.item())
        print(sz)
        break

    # setting
    model = Model((sz[1], sz[2]), 10)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=50)

    # evaluate the model
    modelwrapper.classification_report(test_loader, visualize=True)


# %%
