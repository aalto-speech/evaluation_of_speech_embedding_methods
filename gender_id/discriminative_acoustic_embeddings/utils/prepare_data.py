import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence


# load features combined in one file
def load_features_segmented_combined(features_path):
    feature_array = []
    temp = []
    features = np.load(features_path, allow_pickle=True)

    for feature in features:
        for segment in feature:
            segment = segment.astype(np.float)
            temp.append(segment)
        
        temp = np.array(temp)
        feature_array.append(temp)
        temp = []
    
    feature_array = np.array(feature_array)
    return feature_array



def extract_embeddings(feature_extractor, features, device):
    embedded_seqs = []
    for feature in features:
        for word in feature:
            word = torch.FloatTensor(word)
            word = word.unsqueeze(1)
            hidden = feature_extractor(word.to(device)).squeeze(0)
            embedded_seqs.append(hidden)
    
    embedded_seqs = torch.cat(embedded_seqs, dim=0)
    embedded_seqs = embedded_seqs.cpu()
    embedded_seqs = autograd.Variable(torch.FloatTensor(embedded_seqs))
    
    return embedded_seqs


# load transcripts
def load_labels_combined(labels_path):
    labels_array = []
    with open(labels_path, "r") as f:
        data = f.readlines()
    for line in data:
        line = line.split()
        for elem in line:
            labels_array.append(elem.rstrip())

    return labels_array



# generate label mapping
def label_mapping(labels):
    label2idx = {}
    for label in labels:
        label = label.rstrip()
        if label not in label2idx.keys():
            label2idx[label] = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label


def label_to_idx(labels, label2idx):
    labels_array = []
    for label in labels:
        labels_array.append(label2idx[label])
    labels_array = torch.LongTensor(labels_array)
    return labels_array



def combine_data(features, labels):
    res = []
    for i in range(len(features)):
        res.append((features[i], labels[i]))
    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, label_seqs = zip(*list_of_samples)
    
    padding_value = 0
    
    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    label_seqs = torch.LongTensor(label_seqs)

    return pad_input_seqs, label_seqs
