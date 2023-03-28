import torch
from api_module.model.model_config import ModelParam
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.models.base_model import BaseModel


class lstm(BaseModel, PytorchService):
    model_name = "LSTM"

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 3,
        n_fc_layers: int = 2,
        dense_units: int = 128,
        bidirectional: bool = False,
        dropout_ratio: float = 0.209,
        activation="leakyrelu",

        learning_rate: float = 0.00227,
        epochs: int = 100,
        batch_size: int = 16,
        shuffle: bool = True,
        validation_size: float = 0.2,
        callbacks: dict = None,
        optimizer: str = 'adam',
        loss_fn: str = 'mse',
        weight_decay: float = 1e-4):

        callbacks = {"EarlyStopping": {}} if callbacks is None else callbacks
        self.model_class = LSTMImplementation
        self.model_params = dict(
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_fc_layers=n_fc_layers,
            dense_units=dense_units,
            bidirectional=bidirectional,
            dropout_ratio=dropout_ratio,
            activation=activation
        )
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
        self.model_params_tuning = {
            "n_fc_layers": {"type": "int", "params": {
                "lower": 1,
                "upper": 3
            }
            },
            "dense_units": {"type": "cate", "params": [32, 64, 128]},
            "dropout_ratio": {"type": "float", "params": {
                "lower": 0.0,
                "upper": 0.3
            }
            },
            "num_layers": {"type": "int", "params": {
                "lower": 1,
                "upper": 3
            }
            },
            "hidden_size": {"type": "cate", "params": [32, 64, 96, 128, 160, 192, 224, 256]},
            "activation": {"type": "cate", "params": ['relu', 'leakyrelu', 'silu']}
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


class LSTMImplementation(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        input_horizon: int,
        output_horizon: int,
        N_cols: int,
        num_layers: int,
        bidirectional: bool,
        dropout_ratio: float,
        dense_units: int,
        activation: str,
        n_fc_layers: int,
    ):

        activation_dir = {
            "relu": torch.nn.ReLU,
            "leakyrelu": torch.nn.LeakyReLU,
            "silu": torch.nn.SiLU,
            "selu": torch.nn.SELU,
            "prelu": torch.nn.PReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid
        }
        super(LSTMImplementation, self).__init__()
        self.pred_len = output_horizon

        self.lstm = torch.nn.LSTM(
            input_size=N_cols,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_ratio
        )

        # self.bn_1 = torch.nn.BatchNorm1d(hidden_size)
        # self.flatten = torch.nn.Flatten()
        fc_layers = [torch.nn.Linear(
            hidden_size, dense_units), activation_dir[activation]()]
        for _ in range(n_fc_layers-1):
            fc_layers.append(torch.nn.Linear(dense_units, dense_units))
            fc_layers.append(activation_dir[activation]())
            fc_layers.append(torch.nn.BatchNorm1d(dense_units))
        fc_layers.append(torch.nn.Linear(dense_units, output_horizon))
        self.fc = torch.nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.permute((2, 0, 1))
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        # x = self.flatten(hidden[-1])
        x = self.fc(x).reshape((x.shape[0], 1, self.pred_len))
        return x
