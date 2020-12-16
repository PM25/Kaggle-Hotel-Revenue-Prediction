import torch
import torch.nn as nn
import torch.optim as optim


# default setting
class DefaultSetting:
    def __init__(self, device=None, loss_func=None):
        if device is None:
            self.device = self.default_device(verbose=True)
        else:
            self.device = device

        if loss_func is None:
            self.loss_func = self.default_loss_func()
        else:
            self.loss_func = loss_func

    def get_device(self):
        return self.device

    def get_loss_func(self):
        return self.loss_func

    def default_device(self, verbose=False):
        gpu = True if torch.cuda.is_available() else False
        if gpu:
            if verbose:
                print(f"*Using GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda:0")
            # torch.backends.cudnn.benchmark = True
        else:
            if verbose:
                print("*Using CPU")
            device = torch.device("cpu")
        return device

    def default_optimizer(self, model, lr=0.001):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return optimizer

    def default_loss_func(self):
        loss_func = nn.CrossEntropyLoss()
        return loss_func
