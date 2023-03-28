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


if __name__ == "__main__":
    inp=2
    batches=2
    seq=3
    hid = 2
    num_layers = 2
    buf = io.BytesIO()
    
    lstm = LSTMLayer(LSTMCell,inp,hid)
    inputs = torch.randn((seq,batches, inp))
    lstm = jit.script(lstm,inputs)
    # print(trace.code)
    torch.onnx.export(
        model = lstm,
        args = inputs,
        f = buf,
        # verbose=True,
        opset_version=15,
        # input_names=["input","hidden"],
        # output_names=["output","hidden_n"]
    )

    ort_session = onnxruntime.InferenceSession(buf.getvalue())
    outputs = ort_session.run(
            None, {ort_session.get_inputs()[0].name: inputs.numpy()})
    print(outputs)