import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassificationModel(nn.Module):
    def __init__(self, nfeatures):
        super().__init__()
        self.fc1 = nn.Linear(nfeatures, 512)
        self.batchnorm1d1 = nn.BatchNorm1d(nfeatures)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.batchnorm1d2 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.batchnorm1d1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.batchnorm1d2(x)
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.fc7(x).squeeze()
        return self.sigmoid(x)


class Input1DModelSimplified(nn.Module):
    def __init__(self, nfeatures, nout):
        super().__init__()
        self.fc1 = nn.Linear(nfeatures, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, nout)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x


class Input1DModel(nn.Module):
    def __init__(self, nfeatures, nout):
        super().__init__()
        self.fc1 = nn.Linear(nfeatures, 512)
        self.batchnorm1d1 = nn.BatchNorm1d(nfeatures)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.batchnorm1d2 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, nout)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.batchnorm1d1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.batchnorm1d2(x)
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.fc7(x)
        return x
