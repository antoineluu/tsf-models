import onnx
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import io
import onnxruntime
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
   
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = self.sig(ingate)
        forgetgate = self.sig(forgetgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sig(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.tanh(cy)
        return hy, (hy, cy)

class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)
        self.hid = cell_args[1]

    
    def forward(self, input: Tensor) -> Tensor:
        inputs = input.unbind(0)
        batch = input[0].size()[0]
        state = (torch.zeros((batch,self.hid)),torch.zeros((batch,self.hid)))
        outputs = torch.jit.annotate(Tensor, torch.empty((state[0].size())))
        for i in range(len(inputs)):
            outputs, state = self.cell(inputs[i], state)
        return outputs

class LSTMImplementation(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_horizon: int,
        N_cols: int,
    ):

        super(LSTMImplementation, self).__init__()
        self.pred_len = output_horizon
        self.hid = hidden_size

        self.lstm = torch.nn.LSTM(
            input_size=N_cols,
            hidden_size=hidden_size,
            # num_layers=num_layers,
            # bidirectional=bidirectional,
            # dropout=dropout_ratio
        )
        self.fc_layers = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.permute((2, 0, 1))
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        # x = self.flatten(hidden[-1])
        x = x.reshape((x.shape[0], 1, self.hid))
        return x
if __name__ == "__main__":
    inp=2
    batches=1
    seq=1
    hid = 2
    num_layers = 2
    buf = io.BytesIO()
    
    lstm = LSTMImplementation(
        hid,4,inp
    )
    inputs = torch.randn((batches, inp, seq))
    torch.onnx.export(
        model = lstm,
        args = inputs,
        f = buf,
        # verbose=True,
        # opset_version=14,
        # input_names=["input"],
        # output_names=["output"],
        # dynamic_axes={"input":[0,1],"output":[0,1]},
    )

    ort_session = onnxruntime.InferenceSession(buf.getvalue())
    outputs = ort_session.run(
            None, {ort_session.get_inputs()[0].name: inputs.numpy()})
    print(outputs)