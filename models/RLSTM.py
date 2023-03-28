import torch
import torch.nn as nn
import torch.jit as jit
import io
import onnxruntime
from api_module.model.model_config import ModelParam
from forecasting_module.models.framework_utils.wrapper import PytorchService
from forecasting_module.models.base_model import BaseModel

class rlstm(BaseModel, PytorchService):
    model_name = "RLSTM"

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 3,
        n_fc_layers: int = 2,
        dense_units: int = 128,
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
        self.model_class = RLSTMImplementation
        self.model_params = dict(
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_fc_layers=n_fc_layers,
            dense_units=dense_units,
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

    def get_model(self, **kwargs):
        model =  super().get_model(**kwargs)
        return jit.script(model)

    def save_model(self, output_path):
        inputs = self.args
        print("CUSTOM EXPORT LSTM")
        # traced = jit.trace(self.model,inputs)
        # print(type(self.model))
        torch.onnx.export(
            self.model,
            inputs,
            output_path, 
            input_names = ["x"], 
            output_names = ["output"],
            dynamic_axes={"x":{0:"batch"},"output":{0:"batch"}},
        )

class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        #self.dropout_layer = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden[0], hidden[1]
        # print(torch.mm(input, self.weight_ii.t()).size(), self.bias_ii.size(),hx.size())
        ifo_gates = (torch.mm(input, self.weight_ii.t()) + self.bias_ii +
                     torch.mm(hx, self.weight_ih.t()) + self.bias_ih +
                     torch.mm(cx, self.weight_ic.t()) + self.bias_ic)
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)
        
        cellgate = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)


        # if self.input_size == self.hidden_size:
        #   hy = outgate * (ry + input)
        # else:
        #   hy = outgate * (ry + torch.mm(input, self.weight_ir.t()))

        hy = outgate * (ry + torch.mm(input, self.weight_ir.t()))
        
        return hy, (hy, cy)

class ResLSTMLayers(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.):
        super(ResLSTMLayers, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.cell = LSTMCell(input_size, hidden_size, dropout=0.)
        self.RLSTM_layers = torch.nn.ModuleList(
            [ResLSTMCell(input_size,hidden_size,dropout)]
            +[ResLSTMCell(hidden_size,hidden_size,dropout)
            for i in range(num_layers-1)]
        )
        self.extra_num_layers = num_layers-1

    def forward(self, input, hidden):

        inputs = input.unbind(0)
        h, c = torch.chunk(hidden,2,0)
        outputs=[]
        for i in range(len(inputs)):
            # h,c = h.detach(), c.detach()
            # print(h.requires_grad,c.requires_grad)
            h = h.squeeze(0)
            c = c.squeeze(0)
            out = inputs[i]
            new_h =[]
            new_c =[]
            for j, lay in enumerate(self.RLSTM_layers):

                out, (a, b) = lay(out, (h[j], c[j]))
                new_h.append(a)
                new_c.append(b)
            h = torch.stack(new_h)
            c = torch.stack(new_c)


            outputs +=[out]
        outputs = torch.stack(outputs)
        

        return outputs, (h, c)
    
class RLSTMImplementation(nn.Module):
    def __init__(
            self,
            input_horizon: int,
            output_horizon: int,
            N_cols: int,
            hidden_size: int,
            num_layers: int,
            dropout_ratio: float,
            dense_units: int,
            activation: str,
            n_fc_layers: int,   
        ):
        super().__init__()
        N_cols
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        activation_dir = {
            "relu": torch.nn.ReLU,
            "leakyrelu": torch.nn.LeakyReLU,
            "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh,
            "prelu": torch.nn.PReLU,
            "selu": torch.nn.SELU
        }

        self.net = ResLSTMLayers(N_cols, hidden_size, num_layers)

        
        if n_fc_layers == 0:
            self.fc = torch.nn.Linear(hidden_size,output_horizon)
        else:
            fc_layers = [torch.nn.Linear(hidden_size, dense_units),activation_dir[activation]()]
            for _ in range(n_fc_layers-1):
                fc_layers.append(torch.nn.Linear(dense_units, dense_units))
                fc_layers.append(activation_dir[activation]())
            fc_layers.append(torch.nn.Linear(dense_units, output_horizon))
            self.fc = torch.nn.Sequential(*fc_layers)

    # @jit.script_method
    def forward(self, inputs):
        batches = inputs.size()[0]
        # hidden_h = torch.randn((self.hidden_size*self.num_layers*batch,self.hidden_size))
        # hidden_c = torch.randn((self.hidden_size*self.num_layers*batch,self.hidden_size))
        # self.hidden_init = hidden.h.chunk(self.num_layers,0), hidden_c.chunk(self.num_layers,0)
        h = torch.zeros((self.num_layers, batches, self.hidden_size),device=inputs.device)
        c = torch.zeros((self.num_layers, batches, self.hidden_size),device=inputs.device)
        hidden=torch.stack((h,c))
        inputs = torch.permute(inputs,(2,0,1))
        outputs, _ = self.net(inputs, hidden)
        output = self.fc(outputs[-1])
        return output.unsqueeze(1)


if __name__ == "__main__":
    inp=3
    batches=1
    seq=100
    hid = 32
    # RLSTM_layer = ResLSTMLayer(inp,hid,dropout=0.2)

    # inputs = torch.rand((seq,batches,inp)) #(100,13,20)

    # hidden = (torch.rand((batches,hid)), torch.rand((batches,hid)))

    # outputs, hidden = RLSTM_layer(inputs, hidden)
    buf = io.BytesIO()
    model = RLSTMImplementation(
        input_horizon=seq,
        output_horizon=50,
        N_cols=inp,
        hidden_size=hid,
        num_layers=5,
        dropout_ratio=0.3,
        dense_units=16,
        activation="relu",
        n_fc_layers=3
    )
    inputs = torch.randn((batches,inp,2))
    # outputs = model(inputs)
    traced = jit.script(model)
    

    torch.onnx.export(
        traced,
        inputs,
        buf, 
        input_names = ["x"], 
        output_names = ["output"],
        dynamic_axes={"x":{0:"batch"},"output":{0:"batch"}})

    ort_session = onnxruntime.InferenceSession(buf.getvalue())
    outputs = ort_session.run(
            None, {ort_session.get_inputs()[0].name: inputs.numpy()})
    print(outputs)