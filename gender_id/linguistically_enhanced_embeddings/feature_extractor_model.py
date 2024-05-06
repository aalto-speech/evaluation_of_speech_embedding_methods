import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np



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


class AcousticDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, num_layers, encoder_num_layers, batch_size):
        super(AcousticDecoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.encoder_hidden_size+13,
                            self.decoder_hidden_size,
                            num_layers=self.num_layers,
			    bidirectional=False)
   
        
        self.fc_hidden = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.weight = nn.Parameter(torch.rand((self.batch_size, self.encoder_hidden_size, 1), dtype=torch.float))
        
        self.out = nn.Linear(self.decoder_hidden_size, 13)


    def forward(self, encoder_output, input_tensor, decoder_hidden):
        scores = torch.tanh(self.fc_hidden(decoder_hidden[0]) + self.fc_encoder(encoder_output))
        scores = scores.permute(1, 0, 2)
        scores = scores.bmm(self.weight)
        attn_weights = F.softmax(scores, dim=1)

        context_vector = torch.bmm(attn_weights.permute(0, 2, 1), encoder_output.permute(1, 0, 2))
        output = torch.cat((input_tensor.permute(1, 0, 2), context_vector), 2)
        output = output.permute(1, 0, 2)
        decoder_output, decoder_hidden = self.lstm(output, decoder_hidden)
        
        return decoder_output, decoder_hidden


class LinguisticDecoder(nn.Module):
    def __init__(self, encoder_hidden_size):
        super(LinguisticDecoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.dropout = nn.Dropout(0.3)
        
        self.lin_1 = nn.Linear(encoder_hidden_size, encoder_hidden_size*2)
        self.lin_2 = nn.Linear(encoder_hidden_size*2, encoder_hidden_size*4)
        self.out = nn.Linear(self.encoder_hidden_size*4, 768)


    def forward(self, encoder_hidden):
        output = self.dropout(encoder_hidden)
        output = F.relu(self.lin_1(output))
        output = F.relu(self.lin_2(output))
        output = self.out(output)

        return output
