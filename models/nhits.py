from torch.nn.utils import weight_norm
from api_module.model.model_config import ModelParam
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.models.base_model import BaseModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NHITS(BaseModel, PytorchService):

    model_name = "NHITS"

    def __init__(
        self,
        n_blocks: list = None,
        mlp_units_list: list = None,
        n_pool_kernel_size: list = None,
        n_freq_downsample: list = None,
        pooling_mode: str = "maxpool1d",
        interpolation_mode: str = "linear",
        dropout_ratio: float = 0.0,
        activation: str="relu",

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

        callbacks = {"EarlyStopping": {"patience": 5}, "ReduceLROnPlateau":{}
                     } if callbacks is None else callbacks
        n_blocks = [1, 1, 1] if n_blocks is None else n_blocks
        mlp_units_list = 2 * [128] if mlp_units_list is None else mlp_units_list
        n_pool_kernel_size = [4, 2, 1] if n_pool_kernel_size is None else n_pool_kernel_size
        n_freq_downsample = [4, 2, 1] if n_freq_downsample is None else n_freq_downsample
        self.model_class = NHITSImplementation
        assert len(n_blocks) == len(n_pool_kernel_size), f"unmatching lengths n_blocks:{n_blocks} and n_pool_kernel_size:{n_pool_kernel_size}"
        assert len(n_freq_downsample) == len(n_pool_kernel_size), f"unmatching lengths n_freq_downsample:{n_freq_downsample} and n_pool_kernel_size{n_pool_kernel_size}"
        self.model_params = dict(
            n_blocks=n_blocks,
            mlp_units_list=mlp_units_list,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_ratio=dropout_ratio,
            activation=activation,
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
            "dropout_ratio": {"type": "float", "params": {
                "lower": 0.0,
                "upper": 0.3
            }},
            "activation": {"type": "cate", "params": ['relu', 'leakyrelu', 'silu', 'sigmoid', 'tanh']},
            "interpolation_mode": {"type": "cate", "params": ['avgpool1d', 'maxpool1d']},

            "mlp_units_list": {"type": "cate", "params": [64, 128, 256]},
            "n_blocks": {"type": "int", "params": {
                "lower": 1,
                "upper": 5}},
            "n_freq_downsample": {"type": "int", "params": {
                "lower": 1,
                "upper": 5}},
            "n_pool_kernel_size": {"type": "int", "params": {
                "lower": 1,
                "upper": 5}},


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
class NHITSBlock(nn.Module):

    def __init__(
        self,
        N_cols: int,
        input_horizon: int,
        output_horizon: int,
        mlp_units_list: list,
        n_pool_kernel_size: int,
        n_freq_downsample: int,
        pooling_mode: str,
        dropout_ratio: float,
        activation: str,
        interpolation_mode: str
    ):
        super().__init__()
        activation_dir = {
            "relu": torch.nn.ReLU,
            "leakyrelu": torch.nn.LeakyReLU,
            "silu": torch.nn.SiLU,
            "selu": torch.nn.SELU,
            "prelu": torch.nn.PReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid
        }
        pooling_dir = {
            "maxpool1d": torch.nn.MaxPool1d,
            "avgpool1d": torch.nn.AvgPool1d,
        }
        activation = activation_dir[activation]()
        # pooling
        self.pooling_layer = pooling_dir[pooling_mode](
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLPs
        input_layers = int(np.ceil(input_horizon / n_pool_kernel_size)*N_cols)
        n_theta = input_horizon + max(output_horizon // n_freq_downsample, 1)
        hidden_layers = [
            nn.Linear(in_features=input_layers, out_features=mlp_units_list[0])
        ]
        if len(mlp_units_list)>1:
            for i in range(len(mlp_units_list)-1):
                hidden_layers.append(nn.Linear(in_features=mlp_units_list[i], out_features=mlp_units_list[i+1]))
                hidden_layers.append(activation)
                hidden_layers.append(nn.Dropout(p=dropout_ratio))
        output_layer = [nn.Linear(in_features=mlp_units_list[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        

        self.output_horizon = output_horizon
        self.input_horizon = input_horizon
        self.interpolation_mode = interpolation_mode
        self.N_cols = N_cols

    def forward(
        self,
        insample_y):
        # Pooling
        batch_size = insample_y.shape[0]

        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.reshape(batch_size,-1)
        
        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast = theta[:, : self.input_horizon]
        backcast = backcast.reshape(batch_size,1,-1)

        # Interpolation is performed on default dim=-1 := output_horizon
        knots = theta[:, self.input_horizon : ]
        knots = knots.reshape(batch_size, 1, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            forecast = F.interpolate(
                input=knots, size=self.output_horizon, mode=self.interpolation_mode
            )
        else: raise ValueError("""interpolation mode must be either "nearest" or "linear" """)
        return backcast, forecast

class NHITSImplementation(nn.Module):

    def __init__(
        self,
        N_cols: int,
        output_horizon: int,
        input_horizon: int,
        n_blocks: list,
        mlp_units_list: list,
        n_pool_kernel_size: list,
        n_freq_downsample: list,
        pooling_mode: str,
        interpolation_mode: str,
        dropout_ratio: float,
        activation: str,
    ):
        super().__init__()
        self.output_horizon = output_horizon
        self.N_cols = N_cols
        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):

                nbeats_block = NHITSBlock(
                    N_cols= N_cols,
                    output_horizon=output_horizon,
                    input_horizon=input_horizon,
                    mlp_units_list=mlp_units_list,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    n_freq_downsample=n_freq_downsample[i],
                    pooling_mode=pooling_mode,
                    dropout_ratio=dropout_ratio,
                    activation=activation,
                    interpolation_mode=interpolation_mode,
                )
                block_list.append(nbeats_block)

        self.blocks = torch.nn.ModuleList(block_list)

    def forward(self, insample_y):

        # backcast init
        residuals = insample_y
        batch_size = insample_y.shape[0]
        forecast = torch.zeros(batch_size, 1, self.output_horizon, device=insample_y.device)
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals
            )
            residuals = residuals.clone()
            residuals[:,0:1,:] = (residuals[:,0:1,:] - backcast)
            forecast = forecast + block_forecast

        return forecast