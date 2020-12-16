import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime


#  save & load model for later train
class Checkpoint:
    def __init__(self, base_folder="model"):
        now = datetime.now().strftime("%m-%d-%y_%H.%M.%S")
        self.save_folder = Path(base_folder) / Path(now)
        self.last_save = None
        self.checkpoint_dict = None

    #  save checkpoint model
    def save(self, model, optimizer, loss=None, epoch=None):
        save_path = self.save_folder / Path(f"loss_{loss:.3f}.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"*Saving Model Checkpoint: {save_path}")
        # check if model is on DataParallel format
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        # save model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            save_path,
        )
        self.last_save = save_path
        self.clear_tmp()

    #  load checkpoint model
    def load(self, model=None, optimizer=None, fname=None):
        if self.checkpoint_dict != None:
            return self.checkpoint_dict

        if fname != None:
            checkpoint = torch.load(self.save_folder / Path(fname))
        else:
            checkpoint = torch.load(self.last_save)

        # check if model is on DataParallel format
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        return self.to_dict(model, optimizer, epoch, loss)

    # turn data to dictionary type
    def to_dict(self, model, optimizer, epoch=None, loss=None):
        return {"model": model, "optimizer": optimizer, "epoch": epoch, "loss": loss}

    # save model & optimizer inside the class
    def tmp_save(self, model, optimizer, epoch=None, loss=None):
        self.checkpoint_dict = self.to_dict(model, optimizer, epoch, loss)

    # clear previous temporary save data
    def clear_tmp(self):
        self.checkpoint_dict = None
