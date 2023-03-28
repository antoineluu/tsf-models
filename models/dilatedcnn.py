import torch
from api_module.model.model_config import ModelParam
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.models.base_model import BaseModel
from typing import Union, List, Dict


class DilatedCNN(BaseModel, PytorchService):
    """
    Dilated CNN for Univariate Time Series forecasting.
    References: https://github.com/Azure/DeepLearningForTimeSeriesForecasting (Notebook 1_CNN_dilated.ipynb)
    We will use 3 dilated CNN layers followed by 3 Dense layers.

    Args:
    -------
    NEEDS UPDATE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """

    model_name = "DilatedCNN"
    h = [
        ModelParam(name="cnn_units", type="int", lower_bound=0,
                   exclude_lower=True, default=5, description="CNN units"),
        ModelParam(
            name='cnn_units', type='int',
            lower_bound=0, exclude_lower=True,
            default=32,
            description='CNN units'
        ),
        ModelParam(
            name='kernel_size', type='int',
            lower_bound=0, exclude_lower=True,
            default=2,
            description='Kernel size'
        ),
        ModelParam(
            name='learning_rate', type='float',
            lower_bound=0.0001, upper_bound=0.1,
            exclude_upper=False, exclude_lower=False,
            default=0.001,
            description="Learning rate",
        ),
        ModelParam(
            name="dense_units", type="int", lower_bound=0, exclude_lower=True, default=32, description="MLP/Dense units"
        ),
        ModelParam(name="epochs", type="int", lower_bound=0,
                   exclude_lower=True, default=50, description="Epochs"),
        ModelParam(
            name="batch_size",
            type="int",
            lower_bound=1,
            upper_bound=256,
            exclude_upper=True,
            exclude_lower=True,
            default=64,
            description='Batch size'
        ),
        ModelParam(
            name='shuffle', type='int',
            default=True,
            description='Shuffle training data'
        ),
        # ModelParam(name="apply_lunar", type="int", default=True, description="Add lunar features"),
    ]

    def __init__(
        self,
        n_fc_layers: int = 2,
        cnn_units_list: int = None,
        kernel_size: int = 3,
        dense_units: int = 128,
        dropout_ratio: float = 0.1,
        activation="relu",

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

        cnn_units_list = [512//(2**i) for i in range(3)
                          ] if cnn_units_list is None else cnn_units_list
        callbacks = {"EarlyStopping": {"patience": 5}
                     } if callbacks is None else callbacks
        self.model_class = DilatedCNNImplementation
        self.model_params = dict(
            cnn_units_list=cnn_units_list,
            kernel_size=kernel_size,
            n_fc_layers=n_fc_layers,
            dense_units=dense_units,
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
                "lower": 0,
                "upper": 2}},
            "dense_units": {"type": "cate", "params": [32, 64, 128]},
            "dropout_ratio": {"type": "float", "params": {
                "lower": 0.0,
                "upper": 0.3
            }},
            "activation": {"type": "cate", "params": ['relu', 'leakyrelu', 'silu']},
            "cnn_units_list": {"type": "cate", "params": [64, 128, 256], "n_layers": {
                "type": "int", "params": {
                    "lower": 1,
                    "upper": 4
                }
            }},
            "kernel_size": {"type": "int", "params": {
                "lower": 1,
                "upper": 4}}
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


class DilatedCNNImplementation(torch.nn.Module):

    def __init__(
        self,
        input_horizon: int,
        output_horizon: int,
        N_cols: int,
        n_fc_layers: int,
        cnn_units_list: List,
        kernel_size: int,
        dense_units: int,
        dropout_ratio: float,
        activation: str
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
        super(DilatedCNNImplementation, self).__init__()

        layers = [
            torch.nn.Conv1d(
                in_channels=N_cols,
                out_channels=cnn_units_list[0],
                kernel_size=kernel_size,
                dilation=2,
                padding=kernel_size-1
            ),
            activation_dir[activation](),
            torch.nn.Dropout(p=dropout_ratio)
        ]
        for i in range(len(cnn_units_list)-1):
            conv = torch.nn.Conv1d(
                in_channels=cnn_units_list[i],
                out_channels=cnn_units_list[i+1],
                kernel_size=kernel_size,
                dilation=2,
                padding=kernel_size-1
            )
            layers.append(conv)
            layers.append(activation_dir[activation]())
            # layers.append(torch.nn.BatchNorm1d(cnn_units_list[i+1]))
            layers.append(torch.nn.Dropout(p=dropout_ratio))

        self.conv = torch.nn.Sequential(*layers)

        self.flatten = torch.nn.Flatten()

        if n_fc_layers == 0:
            self.fc = torch.nn.Linear(
                cnn_units_list[-1] * input_horizon, output_horizon)
        else:
            fc_layers = [torch.nn.Linear(
                cnn_units_list[-1] * input_horizon, dense_units), activation_dir[activation]()]
            for _ in range(n_fc_layers-1):
                fc_layers.append(torch.nn.Linear(dense_units, dense_units))
                fc_layers.append(activation_dir[activation]())
            fc_layers.append(torch.nn.Linear(dense_units, output_horizon))
            self.fc = torch.nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = torch.reshape(x, (-1, 1, x.shape[-1]))
        return x
