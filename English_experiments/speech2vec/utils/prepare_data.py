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


# load segmented features combined in one file and group them with the neighbors for skipgram
def load_features_segmented_combined(features_path, form_triplets=True, max_len=80):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    if form_triplets == True:
        for feature in features:
            if len(feature) > 1:
                for i in range(len(feature)):
                    segment = feature[i].astype(np.float)
                    if i == 0:
                        pass
                    elif i == len(feature) - 1:
                        pass
                    else:
                        prev_segment = feature[i-1].astype(np.float)
                        next_segment = feature[i+1].astype(np.float)
                        
                        # pad if necessary
                        length = len(segment)
                        if length > max_len:
                            segment = segment[:max_len, :]
                        elif length < max_len:
                            diff = max_len - length
                            zeros = np.zeros((diff, 13))
                            segment = np.concatenate((segment, zeros), axis=0)

                        length = len(prev_segment)
                        if length > max_len:
                            prev_segment = prev_segment[:max_len, :]
                        elif length < max_len:
                            diff = max_len - length
                            zeros = np.zeros((diff, 13))
                            prev_segment = np.concatenate((prev_segment, zeros), axis=0)

                        length = len(next_segment)
                        if length > max_len:
                            next_segment = next_segment[:max_len, :]
                        elif length < max_len:
                            diff = max_len - length
                            zeros = np.zeros((diff, 13))
                            next_segment = np.concatenate((next_segment, zeros), axis=0)

                        pair = (torch.FloatTensor(segment), torch.FloatTensor(prev_segment), torch.FloatTensor(next_segment))
                        feature_array.append(pair)
    else:
        for feature in features:
            for segment in feature:
                length = len(segment)
                if length > max_len:
                    segment = segment[:max_len, :]
                elif length < max_len:
                    diff = max_len - length
                    zeros = np.zeros((diff, 13))
                    segment = np.concatenate((segment, zeros), axis=0)

                segment = segment.astype(np.float)
                feature_array.append(autograd.Variable(torch.FloatTensor(segment)))

    return feature_array



def combine_data(features):
    res = []
    for i in range(len(features)):
        res.append((features[i][0], features[i][1], features[i][2]))
    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, prev_label_seqs, next_label_seqs = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]
    prev_label_seq_lengths = [len(seq) for seq in prev_label_seqs]
    next_label_seq_lengths = [len(seq) for seq in next_label_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    # pad prev_label seqeucens
    pad_prev_label_seqs = pad_sequence(prev_label_seqs, padding_value=padding_value)
    
    # pad next_label seqeucens
    pad_next_label_seqs = pad_sequence(next_label_seqs, padding_value=padding_value)

    return pad_input_seqs, input_seq_lengths, pad_prev_label_seqs, prev_label_seq_lengths, pad_next_label_seqs, next_label_seq_lengths
