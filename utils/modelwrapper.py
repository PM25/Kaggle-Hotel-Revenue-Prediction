from utils.checkpoint import Checkpoint
from utils.default import DefaultSetting
from utils.earlystopping import EarlyStopping
from utils.visualization import Visualization

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

# TODO: add support for tensorboard & clean code
class ModelWrapper(DefaultSetting):
    def __init__(
        self,
        model,
        loss_func=None,
        optimizer=None,
        device=None,
        multi_gpus=True,
        log=100,
    ):
        super().__init__(device, loss_func)
        self.model = model
        if optimizer is None:
            self.optimizer = self.default_optimizer(model)
        else:
            self.optimizer = optimizer
        self.multi_gpus = multi_gpus
        self.log = log
        self.checkpoint = None
        self.early_stopping = None

    # TODO: haven't check this this function (__call__) yet
    # update model setting
    def __call__(
        self,
        model=None,
        loss_func=None,
        optimizer=None,
        device=None,
        multi_gpus=None,
        log=None,
    ):
        if model is not None:
            self.model = model

        if optimizer is None:
            self.optimizer = self.default_optimizer(self.model)
        else:
            self.optimizer = optimizer

        if loss_func is not None:
            self.loss_func = loss_func

        if device is not None:
            self.device = device

        if multi_gpus is not None:
            self.multi_gpus = multi_gpus

        if self.log is not None:
            self.log = log

        self.checkpoint = None
        self.early_stopping = None

    # train model
    def train(
        self, train_loader, val_loader=None, max_epochs=1000, enable_early_stopping=True
    ):
        if val_loader is None:
            enable_early_stopping = False

        print()
        print("-" * 2, "Training Setup", "-" * 2)
        print(f"Maximum Epochs: {max_epochs}")
        print(f"Enable Early Stoping: {enable_early_stopping}")
        print("-" * 20)
        print("*Start Training.")

        # model setup
        self.model.train().to(self.device)
        if self.multi_gpus and torch.cuda.device_count() > 1:
            print(f"*Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        # early stopping instance
        if enable_early_stopping:
            if self.early_stopping is None:
                self.early_stopping = EarlyStopping(patience=5)
            else:
                self.early_stopping.reset_counter()

        # training start!
        for epoch in range(1, max_epochs + 1):
            running_loss = 0.0

            for step, data in enumerate(train_loader, start=1):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()

                if step % 100 == 0 or step == len(train_loader):
                    print(
                        f"[{epoch}/{max_epochs}, {step}/{len(train_loader)}] loss: {running_loss / step :.3f}"
                    )

            # train & validation loss
            train_loss = running_loss / len(train_loader)
            if val_loader is None:
                print(f"train loss: {train_loss:.3f}")
            else:
                # FIXME: fixed the problem that first validation is not correct
                val_loss = self.validation(val_loader)
                print(f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

                if enable_early_stopping:
                    self.early_stopping(self.model, val_loss, self.optimizer)
                    if self.early_stopping.get_early_stop() == True:
                        print("*Early Stopping.")
                        break

        print("*Finished Training!")
        if enable_early_stopping:
            checkpoint = self.early_stopping.get_checkpoint()
        else:
            checkpoint = Checkpoint()
            checkpoint.tmp_save(self.model, self.optimizer, epoch, val_loss)
        self.checkpoint = checkpoint
        self.model = checkpoint.load(self.model, self.optimizer)["model"]
        return self.model

    # %% validation
    @torch.no_grad()
    def validation(self, val_loader):
        self.model.eval().to(self.device)
        running_loss = 0.0
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            running_loss += loss.item()
        return running_loss / len(val_loader)

    # classification report of the model on test data
    @torch.no_grad()
    def classification_report(
        self, test_loader, target_names=None, binary=False, visualize=False
    ):
        print("-" * 10, "Classification Report", "-" * 10)
        print(f"loss: {self.validation(test_loader)}")
        model = self.model
        model.eval().to(self.device)

        y_pred, y_true = [], []
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device).long()
            outputs = model(inputs)
            if not binary:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = torch.round(outputs)

            y_true += labels.squeeze().cpu().tolist()
            y_pred += predicted.squeeze().cpu().tolist()

        if visualize:
            vis = Visualization(y_true, y_pred, target_names)
            vis.confusion_matrix()
            vis.classification_report()
            vis.show()
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report)
        return report
