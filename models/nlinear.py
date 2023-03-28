import numpy as np
import torch
from api_module.model.model_config import ModelParam
from forecasting_module.callbacks.loss_history import LossHistory
from forecasting_module.callbacks.val_loss_history import ValLossHistory
from forecasting_module.callbacks.early_stopping import EarlyStopping
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.problems.base import BaseProblem
from forecasting_module.models.framework_utils.torch.dataloader import acquire_dataloader
from forecasting_module.models.framework_utils.torch.dataloader import acquire_test_tensor
from forecasting_module.models.framework_utils.torch.trainer import TorchTrainer
from forecasting_module.models.base_model import BaseModel


class NLinear(BaseModel, PytorchService):
    model_name = "NLinear"

    def __init__(
        self,
        renormalize: bool = False,
        learning_rate: float = 0.0010721,
        epochs: int = 100,
        batch_size: int = 256,
        shuffle: bool = True,
        validation_size: float = 0.2,
        callbacks: dict = None,
        optimizer: str = 'adam',
        loss_fn: str = 'mse',
        weight_decay: float = 1e-4

    ):
        callbacks = {"EarlyStopping": {"patience": 5}, "ReduceLROnPlateau":{}} if callbacks is None else callbacks
        self.model_class = NLinearImplementation
        self.model_params = dict(renormalize=renormalize)
        self.training_params = dict(
            learning_rate=learning_rate,
            epochs=epochs,
            validation_size=validation_size,
            callbacks=callbacks,
            loss_fn=loss_fn,
            optimizer=optimizer,
            weight_decay=weight_decay,
        )
        self.dataloader_params = dict(
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.model_params_tuning={
            "renormalize" :{"type":"bool"}
        }
        self.training_params_tuning = {
            "learning_rate": {"type": "float", "params": {
                "lower": 1e-4,
                "upper": 1e-2
            },
                "kparams": {"log": True}},
        }
        self.dataloader_params_tuning = {
            "batch_size": {"type": "cate", "params": [16, 32, 64]}
        }

class NLinearImplementation(torch.nn.Module):

    def __init__(
        self,
        input_horizon: int,
        output_horizon: int,
        N_cols: int,
        renormalize: bool
    ):
        super(NLinearImplementation, self).__init__()
        self.seq_len = input_horizon
        self.pred_len = output_horizon
        self.N_cols = N_cols
        self.Linear = torch.nn.Linear(self.seq_len*N_cols, self.pred_len)
        self.renormalize = renormalize

    def forward(self, x):
        seq_last = x[:, :, -1:]

        x = x - seq_last
        x = x.reshape((x.shape[0], self.seq_len*self.N_cols))
        x = self.Linear(x).reshape((x.shape[0], 1, self.pred_len))
        if self.renormalize:
            x = x + seq_last.mean(axis=1).reshape((x.shape[0], 1, 1))
        return x
