import numpy as np
import torch
from copy import deepcopy
from forecasting_module.callbacks.early_stopping import EarlyStopping
from forecasting_module.callbacks.loss_history import LossHistory
from forecasting_module.callbacks.prune_trial import PruneTrial
from forecasting_module.callbacks.reduce_LR_on_plateau import ReduceLROnPlateau
from forecasting_module.callbacks.return_best_model import ReturnBestModel
from forecasting_module.callbacks.val_loss_history import ValLossHistory

loss_fn_dir = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss
}

optimizer_dir = {
    "adam": torch.optim.Adam,
}
torch.autograd.set_detect_anomaly(True)
class TorchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        loss_fn: str,
        optimizer: str,
        learning_rate: float,
        validation_size : float,
        pipeline_id: str,
        callbacks: dict,
        weight_decay: dict,
        verbose: bool,
    ):
        self.callbacks = self.callbacks_interpreter(callbacks,pipeline_id)
        self.model = model
        self.epochs = epochs
        self.loss_fn = loss_fn_dir.get(loss_fn, torch.nn.MSELoss)()
        self.optimizer = optimizer_dir.get(optimizer, torch.optim.Adam)(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay

        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_size = validation_size
        self.verbose = verbose

    def callbacks_interpreter(self, callbacks, pipeline_id):
        callback_list =[]
        for callback, params_dict in callbacks.items():
            callback_list.append(eval(callback)(pipeline_id,**params_dict))
        return callback_list

    def train(self, train_dataloader, val_dataloader=()):
        self.model.to(self.device)
        self.logs = {}
        if self.verbose:
            print("Training started")
        [callback.on_train_begin(logs=self.logs, model=self.model) for callback in self.callbacks]
        for epoch in range(self.epochs) :
            if self.logs.get("Keep_training",True):
                [callback.on_epoch_begin(epoch=epoch, logs=self.logs, model=self.model) for callback in self.callbacks]
                val_loss_records = []
                loss_records = []
                for batch, sample in enumerate(train_dataloader):
                    # print(batch)
                    self.model.train()
                    [callback.on_batch_begin(batch=batch, logs=self.logs) for callback in self.callbacks]
                    self.model.zero_grad()
                    loss = self.compute_loss(sample)
                    loss.backward()
                    self.optimizer.step()
                    loss_records.append(loss.data.cpu().numpy())
                    [callback.on_batch_end(batch=batch, logs=self.logs) for callback in self.callbacks]
                self.logs["loss"] = np.array(loss_records).mean()
                for batch, sample in enumerate(val_dataloader):
                    self.model.eval()
                    val_loss = self.compute_loss(sample)
                    val_loss_records.append(val_loss.data.cpu().numpy())
                self.logs["val_loss"] = np.array(val_loss_records).mean() if val_loss_records else self.logs["loss"]
                if self.verbose:
                    print(f"epoch {epoch}/{self.epochs-1}: LOSS {self.logs['loss']:2f} VLOSS {self.logs['val_loss']:2f}" )
                [callback.on_epoch_end(epoch=epoch, logs=self.logs, model=self.model, optimizer=self.optimizer) for callback in self.callbacks]
        [callback.on_train_end(logs=self.logs, model=self.model) for callback in self.callbacks]
        self.model.to("cpu")
        return self.logs
    

    def compute_loss(self, sample, return_preds=False):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model.forward(x)
        loss = self.loss_fn(pred, y)
        if return_preds:
            return loss, pred
        return loss


    def test(self, dataloader):
        torch.no_grad()
        total_loss = 0
        preds= []
        self.model.to(self.device)
        self.model.eval()
        for batch, sample in enumerate(dataloader):
            loss = self.compute_loss(sample, return_preds=True)
            total_loss += loss[0].data.cpu().numpy()
            preds.append(loss[1].data.cpu().numpy())
        preds = np.vstack(preds)
        self.model.to("cpu")
        return preds, total_loss/len(dataloader)
