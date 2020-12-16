import torch
from torch import from_numpy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset

torch.manual_seed(0)

######### Example Usage #########
# from sklearn.datasets import load_boston

# train_loader, val_loader, test_loader = LoadData(
#     X_y=load_boston(return_X_y=True), X_y_dtype=("float", "float")
# ).get_dataloader([0.7, 0.2, 0.1])
#################################


class LoadData:
    def __init__(self, dataset=None, X_y=None, X_y_dtype=("float", "float")):
        # default convert data to float
        if X_y != None:
            X, y = X_y
            X_dtype, y_dtype = X_y_dtype
            X = from_numpy(X)
            X = self.tensor_to_type(X, X_dtype)
            y = from_numpy(y).squeeze()
            y = self.tensor_to_type(y, y_dtype)
            dataset = TensorDataset(X, y)

        self.dataset = dataset

    def tensor_to_type(self, tensor, dtype="float"):
        dtype = dtype.casefold()
        if dtype == "float":
            tensor = tensor.float()
        elif dtype == "long":
            tensor = tensor.long()
        return tensor

    def get_dataloader(self, split_ratio=[1], batch_size=128):
        dataloaders = []
        for splitted_dataset in self.split(split_ratio):
            dataloaders.append(
                DataLoader(splitted_dataset, batch_size=batch_size, shuffle=True)
            )
        if len(dataloaders) == 1:
            dataloaders = dataloaders[0]
        return dataloaders

    def split(self, ratio=[0.7, 0.2, 0.1]):
        counts = []
        for r in ratio:
            counts.append(int(len(self.dataset) * r))

        if sum(counts) != len(self.dataset):
            missing_count = len(self.dataset) - sum(counts)
            counts[-1] += missing_count

        return random_split(self.dataset, counts)
