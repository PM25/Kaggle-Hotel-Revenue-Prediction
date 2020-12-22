from utils.checkpoint import Checkpoint


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.checkpoint = Checkpoint("model")

    def __call__(self, model, val_loss, optimizer, epoch=None):
        score = -val_loss
        if self.best_score == None or score > self.best_score + self.delta:
            self.counter = 0
            self.best_score = score
            self.checkpoint.save(model, optimizer, val_loss, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    # reset the counter
    def reset_counter(self):
        self.counter = 0

    # check if the early stopping criteria is meet
    def get_early_stop(self):
        return self.early_stop

    # return class checkpoint (for loading & saving model)
    def get_checkpoint(self):
        return self.checkpoint
