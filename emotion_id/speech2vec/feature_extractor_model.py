import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, input_tensor, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.input_tensor = input_tensor
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                            self.input_tensor,
                            self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True
                            )


    def forward(self, input_tensor):
        output, hidden = self.lstm(input_tensor)
        
        return output, hidden


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_ch = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_ch = torch.nn.Parameter(torch.randn(4 * hidden_size))

        self.init_weights()

    def forward(self, input, state, encoder_hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # input: batch_size * input_size
        # state: hx -> batch_size * projection_size 
        #        cx -> batch_size * hidden_size 
        # state cannot be None
        '''
        if state is not None:
            hx, cx = state
        else:
            hx = input.new_zeros(input.size(0), self.projection_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        '''
        hx, cx = state
        
        input = input.squeeze(1)
        hx = hx.squeeze(0)
        cx = cx.squeeze(0)
        enh = encoder_hidden[0].squeeze(0)

        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh + torch.mm(enh, self.weight_ch.t()) + self.bias_ch)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)
    

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.uniform_(self.weight_ih, -stdv, stdv)
        torch.nn.init.uniform_(self.weight_hh, -stdv, stdv)
        torch.nn.init.uniform_(self.weight_ch, -stdv, stdv)
        torch.nn.init.uniform_(self.bias_ih)
        torch.nn.init.uniform_(self.bias_hh)
        torch.nn.init.uniform_(self.bias_ch)
    

class Decoder(nn.Module):
    def __init__(self, decoder_hidden_size, batch_size):
        super(Decoder, self).__init__()

        self.decoder_hidden_size = decoder_hidden_size
        self.batch_size = batch_size

        self.lstm = CustomLSTM(13, self.decoder_hidden_size)
        self.out = nn.Linear(self.decoder_hidden_size, 13)


    def forward(self, encoder_hidden, input_tensor, decoder_hidden):
        input_tensor = input_tensor.permute(1, 0, 2)
        decoder_output, decoder_hidden = self.lstm(input_tensor, decoder_hidden, encoder_hidden)

        return decoder_output, decoder_hidden
