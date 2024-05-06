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
        
        self.out = nn.Linear(128, output_size)


    def forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0)
        output = self.out(input_tensor)

        return output
