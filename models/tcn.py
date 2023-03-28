import torch
from torch.nn.utils import weight_norm
from api_module.model.model_config import ModelParam
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.models.base_model import BaseModel


class tcn(BaseModel, PytorchService):

    model_name = "TCN"

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
        callbacks = {"EarlyStopping": {"patience": 5}, "ReduceLROnPlateau":{}
                     } if callbacks is None else callbacks
        self.model_class = TCNImplementation
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


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, activation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = activation()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(torch.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = activation()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                       self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = torch.nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = activation()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNImplementation(torch.nn.Module):

    def __init__(
        self,
        n_fc_layers: int,
        dense_units: int,
        cnn_units_list,
        kernel_size,
        dropout_ratio,
        input_horizon: int,
        output_horizon: int,
        N_cols: int,
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
        super(TCNImplementation, self).__init__()
        layers = []
        num_levels = len(cnn_units_list)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = N_cols if i == 0 else cnn_units_list[i-1]
            out_channels = cnn_units_list[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, activation=activation_dir[activation],  dropout=dropout_ratio)]
        self.flatten = torch.nn.Flatten()
        self.network = torch.nn.Sequential(*layers)

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
        x = self.network(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = torch.reshape(x, (-1, 1, x.shape[-1]))
        return x
