# %%
from utils import *
from data_processing import Data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%%
data = Data()
X, y = data.processing_cnn()

loaddata = LoadData(X_y=(X, y), X_y_dtype=("float", "long"))
train_loader, val_loader, test_loader = loaddata.get_dataloader([0.75, 0.1, 0.1])


class Model(nn.Module):
    def __init__(self, c, output):
        super().__init__()
        self.conv1 = nn.Conv2d(c, 3, 5)
        self.fc1 = nn.Linear(444 * 24 * 3, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 444 * 24 * 3)
        x = self.fc1(x)
        return x


# %% start from here!
if __name__ == "__main__":
    for data in train_loader:
        x, y = data
        sz = x.shape
        print(sz)
        break

    # setting
    model = Model(sz[1], 10)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=50)

    # evaluate the model
    modelwrapper.classification_report(test_loader, visualize=True)

# %%
