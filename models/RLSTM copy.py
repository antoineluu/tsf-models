import torch
import torch.nn as nn
import torch.jit as jit
import io
import onnxruntime
from RLSTMcopy2 import *

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
    inp=9
    batches=5
    seq=100
    hid = 10
    num_layers = 1
    # RLSTM_layer = ResLSTMLayer(inp,hid,dropout=0.2)

    # inputs = torch.rand((seq,batches,inp)) #(100,13,20)

    # hidden = (torch.rand((batches,hid)), torch.rand((batches,hid)))

    # outputs, hidden = RLSTM_layer(inputs, hidden)
    buf = io.BytesIO()
    # model = RLSTMImplementation(
    #     input_horizon=seq,
    #     output_horizon=25,
    #     N_cols=inp,
    #     hidden_size=hid,
    #     num_layers=2,
    #     dropout_ratio=0.0,
    #     dense_units=16,
    #     activation="relu",
    #     n_fc_layers=0
    # )

    # inputs = torch.randn((batches,inp,2))
    # print("inputs",inputs.size())
    # # outputs = model(inputs)
    # traced = jit.script(model)
    # output = traced(inputs)
    # print("output", output.size())
    # print(traced.code)

    # torch.onnx.export(
    #     traced,
    #     inputs,
    #     buf,
    #     input_names = ["x"], 
    #     output_names = ["output"],
    #     dynamic_axes={"x":{0:"batch"},"output":{0:"batch"}},
    #     opset_version=15)

    # ort_session = onnxruntime.InferenceSession(buf.getvalue())
    # outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: inputs.numpy()})

    # lstm = LSTMLayer(
    #     cell = 
    #     input_size=inp,
    #     hidden_size=hid,
    #     num_layers=num_layers
    # )
    inputs = torch.randn((seq, batches, inp))
    # states = LSTMState(torch.randn(batches, hid),torch.randn(batches, hid))
    states = [LSTMState(torch.randn(batches, hid),
                        torch.randn(batches, hid))
              for _ in range(num_layers)]
    out = lstm(inputs, states)
    torch.onnx.export(
        model = lstm,
        args = inputs,
        f = buf,
        input_names=["input","hidden","cell"],
        output_names=["output","hidden_n","cell_n"]
    )