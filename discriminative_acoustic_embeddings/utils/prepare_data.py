import numpy as np
import os
from collections import defaultdict, Counter
import itertools
import pickle

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

            # pad if necessary
            length = len(segment)
            if length > max_len:
                segment = segment[:max_len, :]
            elif length < max_len:
                diff = max_len - length
                zeros = np.zeros((diff, 13))
                segment = np.concatenate((segment, zeros), axis=0)

            feature_array.append(autograd.Variable(torch.FloatTensor(segment)))
    
    return feature_array


# load transcripts segmented
def load_transcripts_segmented_combined(transcripts_path):
    transcripts_array = []
    with open(transcripts_path, "r") as f:
        data = f.readlines()

    for line in data:
        line = line.split()
        for word in line:
            transcripts_array.append(word.rstrip())

    return transcripts_array


def clean_data(features, labels):
    clean_features = []
    clean_labels = []

    word_loc = defaultdict(list)
    for i, item in enumerate(labels):
        word_loc[item].append(i)
    
    for key, locs in word_loc.items():
        # if a word appears too many times, make few pairs
        if len(locs) > 10:
            locs = locs[:10] 
        
        for loc in locs:
            clean_features.append(features[loc])
            clean_labels.append(labels[loc])
    
    return clean_features, clean_labels


def form_pairs(features, labels, idx2word):
    feature_pairs = []
    label_pairs = []
    word2pos = {} 
    
    word_loc = defaultdict(list)
    for i, item in enumerate(labels):
        item = idx2word[item.item()]
        word_loc[item].append(i)
    
    for key, locs in word_loc.items():
        word2pos[key] = locs
        try: 
            if len(locs) > 1:
                for i in range(len(locs)-1):
                    for j in range(i+1, len(locs)):
                        label_pairs.append((labels[locs[i]], labels[locs[j]]))
                        feature_pairs.append((features[locs[i]], features[locs[j]]))
        except:
            pass
    
    return feature_pairs, label_pairs, word2pos


def get_idx_word_dict(words):
    idx2word = {}
    for word in words:
        if word not in idx2word.values():
            idx2word[len(idx2word)] = word

    return idx2word


def label_to_idx(words, word2idx):
    res_words = []
    res_word_pairs = []
    for word in words:
        res_words.append(torch.tensor(word2idx[word]))
    
    return res_words



def combine_data(features, labels):
    res = []
    for i in range(len(features)):
        res.append((features[i][0], features[i][1], labels[i][0], labels[i][1]))

    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs_1, input_seqs_2, label_seqs_1, label_seqs_2 = zip(*list_of_samples) 
    
    padding_value = 0

    # pad input sequences
    pad_input_seqs_1 = pad_sequence(input_seqs_1, padding_value=padding_value)
    pad_input_seqs_2 = pad_sequence(input_seqs_2, padding_value=padding_value)
    
    label_seqs_1 = torch.vstack(label_seqs_1)
    label_seqs_2 = torch.vstack(label_seqs_2)

    return pad_input_seqs_1, pad_input_seqs_2, label_seqs_1, label_seqs_2
