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


class ClassificationModel(nn.Module):
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


class ImageClassificationModel(nn.Module):
    def __init__(self, nfeatures, nout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.batchnorm2d1 = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.batchnorm1d1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, nout)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.batchnorm2d1(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.batchnorm1d1(x)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
