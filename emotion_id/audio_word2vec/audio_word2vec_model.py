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
        #input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths)
        output, hidden = self.lstm(input_tensor)
        #output = pad_packed_sequence(output)[0]
        
        return output, hidden


#class CustomLSTMEncoder(nn.Module):
#    def __init__(self, input_sz, hidden_sz, peephole, batch_size):
#        super(CustomLSTMEncoder, self).__init__()
#        self.peephole = peephole
#
#        self.input_sz = input_sz
#        self.hidden_size = hidden_sz
#        
#        self.W_i = nn.Parameter(torch.Tensor(batch_size, input_sz, hidden_sz))
#        self.W_f = nn.Parameter(torch.Tensor(batch_size, input_sz, hidden_sz))
#        self.W_g = nn.Parameter(torch.Tensor(batch_size, input_sz, hidden_sz))
#        self.W_o = nn.Parameter(torch.Tensor(batch_size, input_sz, hidden_sz))
#        
#        self.U_i = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        self.U_f = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        self.U_g = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        self.U_o = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#
#        self.C_i = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        self.C_f = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        self.C_g = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        self.C_o = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#        
#        self.bias_i = nn.Parameter(torch.Tensor(batch_size, 1, hidden_sz))
#        self.bias_f = nn.Parameter(torch.Tensor(batch_size, 1, hidden_sz))
#        self.bias_g = nn.Parameter(torch.Tensor(batch_size, 1, hidden_sz))
#        self.bias_o = nn.Parameter(torch.Tensor(batch_size, 1, hidden_sz))
#
#        self.init_weights()
#
#        if self.peephole == True:
#            self.C_i = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#            self.C_f = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#            self.C_o = nn.Parameter(torch.Tensor(batch_size, hidden_sz, hidden_sz))
#
#        self.init_weights()
#        
#
#    def init_weights(self):
#        stdv = 1.0 / math.sqrt(self.hidden_size)
#        for weight in self.parameters():
#            weight.data.uniform_(-stdv, stdv)
#        
#
#    def forward(self, x, init_states=None):
#        """x = (batch, sequence, feature)"""
#        x = x.permute(1, 0, 2)
#        bs, seq_sz, _ = x.size()
#        hidden_seq = []
#
#        if init_states is None:
#            h_t, c_t = (torch.zeros(bs, 1, self.hidden_size).to(x.device), torch.zeros(bs, 1, self.hidden_size).to(x.device))
#        else:
#            h_t, c_t = init_states
#        
#        for t in range(seq_sz):
#            x_t = x[:, t, :].unsqueeze(1)
#
#            if self.peephole:
#                i_t = torch.sigmoid(torch.bmm(x_t, self.W_i) + torch.bmm(h_t, self.U_i) + torch.bmm(c_t, self.C_i) + self.bias_i)
#                f_t = torch.sigmoid(torch.bmm(x_t, self.W_f) + torch.bmm(h_t, self.U_f) + torch.bmm(c_t, self.C_f) + self.bias_f)
#                o_t = torch.sigmoid(torch.bmm(x_t, self.W_o) + torch.bmm(h_t, self.U_o) + torch.bmm(c_t, self.C_o) + self.bias_o)
#                g_t = torch.tanh(torch.bmm(x_t, self.W_g) + torch.bmm(h_t, self.U_g) + self.bias_g)
#            else:
#                i_t = torch.sigmoid(torch.bmm(x_t, self.W_i) + torch.bmm(h_t, self.U_i) + self.bias_i)
#                f_t = torch.sigmoid(torch.bmm(x_t, self.W_f) + torch.bmm(h_t, self.U_f) + self.bias_f)
#                o_t = torch.sigmoid(torch.bmm(x_t, self.W_o) + torch.bmm(h_t, self.U_o) + self.bias_o)
#                g_t = torch.tanh(torch.bmm(x_t, self.W_g) + torch.bmm(h_t, self.U_g) + self.bias_g)
#
#            c_t = f_t * c_t + i_t * g_t
#            h_t = o_t * torch.tanh(c_t)
#
#            hidden_seq.append(h_t.unsqueeze(0))
#        hidden_seq = torch.cat(hidden_seq, dim=0)
#        
#        return hidden_seq.squeeze(2), (h_t.permute(1, 0, 2), c_t.permute(1, 0, 2))
#
#
#
#class Decoder(nn.Module):
#    def __init__(self, decoder_hidden_size, batch_size, peephole):
#        super(Decoder, self).__init__()
#
#        self.decoder_hidden_size = decoder_hidden_size
#        self.batch_size = batch_size
#        self.peephole = peephole
#
#        self.lstm = CustomLSTMEncoder(13, self.decoder_hidden_size, self.peephole, self.batch_size)
#        self.out = nn.Linear(self.decoder_hidden_size, 13)
#
#    def forward(self, input_tensor, decoder_hidden):
#        input_tensor = input_tensor.permute(1, 0, 2)
#        decoder_hidden = (decoder_hidden[0].permute(1, 0, 2), decoder_hidden[1].permute(1, 0, 2))
#        decoder_output, decoder_hidden = self.lstm(input_tensor, init_states=decoder_hidden)
#
#        return decoder_output, decoder_hidden




# LSTM with peephole connection
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = torch.nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ch = torch.nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_gh = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_ghh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.bias_ih = torch.nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ch = torch.nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_gh = torch.nn.Parameter(torch.randn(hidden_size))
        self.bias_ghh = torch.nn.Parameter(torch.randn(hidden_size))

        self.init_weights()

    def forward(self, input, state):
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
       
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh + torch.mm(cx, self.weight_ch.t()) + self.bias_ch)
        ingate, forgetgate, outgate = gates.chunk(3, 1)
        cellgate = torch.mm(input, self.weight_gh.t() + self.bias_gh) + torch.mm(hx, self.weight_ghh.t() + self.bias_ghh)
        
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
        torch.nn.init.uniform_(self.weight_gh, -stdv, stdv)
        torch.nn.init.uniform_(self.weight_ghh, -stdv, stdv)
        torch.nn.init.uniform_(self.bias_ih)
        torch.nn.init.uniform_(self.bias_hh)
        torch.nn.init.uniform_(self.bias_ch)
        torch.nn.init.uniform_(self.bias_gh)
        torch.nn.init.uniform_(self.bias_ghh)


class Decoder(nn.Module):
    def __init__(self, decoder_hidden_size, batch_size):
        super(Decoder, self).__init__()

        self.decoder_hidden_size = decoder_hidden_size
        self.batch_size = batch_size

        self.lstm = CustomLSTM(13, self.decoder_hidden_size)
        self.out = nn.Linear(self.decoder_hidden_size, 13)


    def forward(self, input_tensor, decoder_hidden):
        #input_tensor = input_tensor.permute(1, 0, 2)
        decoder_output, decoder_hidden = self.lstm(input_tensor, decoder_hidden)

        return decoder_output, decoder_hidden
