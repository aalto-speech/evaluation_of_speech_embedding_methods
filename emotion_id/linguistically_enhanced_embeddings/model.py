import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np


class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(self.input_size,
                            512,
                            num_layers=4,
                            bidirectional=True
                            )        
       
        self.lin_1 = nn.Linear(1024, 1024)
        self.lin_2 = nn.Linear(1024, 1024)
        self.lin_3 = nn.Linear(1024, 1024)

        self.out = nn.Linear(1024, output_size)


    def forward(self, embedded_seq, input_seq_lengths):
        embedded_seq = pack_padded_sequence(embedded_seq, input_seq_lengths, enforce_sorted=False)
        output, hidden = self.lstm(embedded_seq)
        output = pad_packed_sequence(output)[0]
        output = torch.sum(output, dim=0)
        output = self.dropout(output)

        output = F.relu(self.lin_1(output))
        output = F.relu(self.lin_2(output))
        output = F.relu(self.lin_3(output))
        output = self.out(output) 
        
        return output
