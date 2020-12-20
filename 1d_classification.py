# %%
from utils import *
from datapreprocessing import processing_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%%
(X_np, y_np), reservation_status_cats = processing_data()
print(f"X_np's shape: {X_np.shape}")
print(f"y_np's shape: {y_np.shape}")

train_loader, val_loader, test_loader = LoadData(
    X_y=(X_np, y_np), X_y_dtype=("float", "long")
).get_dataloader([0.7, 0.2, 0.1], batch_size=64)


# %% start from here!
if __name__ == "__main__":
    # setting
    model = ClassificationModel(X_np.shape[1], len(reservation_status_cats))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=1)

    # evaluate the model
    modelwrapper.classification_report(
        test_loader, reservation_status_cats, visualize=True
    )

    # get prediction results
    model.cpu()
    preds = []
    for data in train_loader:
        X, y = data
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        preds += predicted.squeeze().tolist()
