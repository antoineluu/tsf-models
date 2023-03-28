import torch
from api_module.model.model_config import ModelParam
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.models.base_model import BaseModel


class Linear(BaseModel, PytorchService):

    model_name = "Linear"

    def __init__(
        self,
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
        self.model_class = LinearImplementation
        self.model_params = dict()
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
        self.model_params_tuning = {}
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


class LinearImplementation(torch.nn.Module):

    def __init__(
        self,
        input_horizon: int,
        output_horizon: int,
        N_cols: int,

    ):
        super(LinearImplementation, self).__init__()
        self.seq_len = input_horizon
        self.pred_len = output_horizon
        self.N_cols = N_cols
        self.Linear = torch.nn.Linear(self.seq_len*N_cols, self.pred_len)

    def forward(self, x):
        x = x.reshape((x.shape[0], self.seq_len*self.N_cols))
        x = self.Linear(x).reshape((x.shape[0], 1, self.pred_len))
        return x
