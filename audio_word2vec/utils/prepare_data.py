import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

# load features combined in one file
def load_features_combined(features_path):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)

    for feature in features:
            feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


# load segmented features combined in one file
def load_features_segmented_combined(features_path, max_len=80):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)

    for feature in features:
        for segment in feature:
            segment = segment.astype(np.float)
            if segment.shape[0] > 80:
                segment = segment[:max_len, :]
            feature_array.append(autograd.Variable(torch.FloatTensor(segment)))

    return feature_array


def combine_data(features):
    res = []
    for i in range(len(features)):
        res.append((features[i], features[i]))

    return res


def zero_masking(pad_input_seqs, prob=0.3):
    for i in range(pad_input_seqs.size(1)):
        mask = torch.FloatTensor(pad_input_seqs.size(0)).uniform_() > prob
        mask = mask.int()
        mask_idx = torch.where(mask == 0)
        temp = pad_input_seqs[:, i, :]
        temp[mask_idx] = torch.zeros(13)
        pad_input_seqs[:, i, :] = temp

    return pad_input_seqs


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, label_seqs = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]
    label_seq_lengths = [len(seq) for seq in label_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    # apply zero masking
    pad_input_seqs = zero_masking(pad_input_seqs)
    
    # pad label seqeucens
    pad_label_seqs = pad_sequence(label_seqs, padding_value=padding_value)

    return pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths
