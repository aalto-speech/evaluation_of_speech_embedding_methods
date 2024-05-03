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


# load features combined in one file
def load_features_segmented_combined(features_path):
    feature_array = []
    temp = []
    lengths = []
    features = np.load(features_path, allow_pickle=True)

    for feature in features:
        for segment in feature:
            segment = segment.astype(np.float)
            temp.append(segment)
        
        temp = np.array(temp)
        lengths.append(len(temp))
        feature_array.append(temp)
        temp = []
    
    feature_array = np.array(feature_array)
    return feature_array, lengths


def extract_embeddings(feature_extractor, features, device):
    embedded_seqs = []
    embedded_sentence = []
    for feature in features:
        for word in feature:
            word = torch.FloatTensor(word)
            word = word.unsqueeze(1)
            output, hidden = feature_extractor(word.to(device))
            hidden = hidden[0].sum(0, keepdim=True).squeeze(0)
            embedded_sentence.append(hidden)

        embedded_sentence = torch.cat(embedded_sentence, dim=0)
        embedded_sentence = embedded_sentence.cpu()
        embedded_seqs.append(autograd.Variable(torch.FloatTensor(embedded_sentence)))
        embedded_sentence = []
    
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


def extract_bert_embeddings(model, tokenizer, data, device):
    embedding_array = []
    for sample in data:
        sentence = []
        words = []
        sent = '[CLS] ' + sample + ' [SEP]'

        # tokenize it
        tokenized_sent = tokenizer.tokenize(sent)
        if len(tokenized_sent) > 512:
            tokenized_sent = tokenized_sent[:511]
            tokenized_sent.append('[SEP]')
        sent_idx = tokenizer.convert_tokens_to_ids(tokenized_sent)
    
        # add segment ID
        segments_ids = [1] * len(sent_idx)
    
        # convert data to tensors
        sent_idx = torch.tensor([sent_idx]).to(device)
        segments_ids = torch.tensor([segments_ids]).to(device)
        
        #get embeddings
        with torch.no_grad():
            outputs = model(sent_idx, segments_ids)
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=0)
            embeddings = torch.sum(hidden_states[-4:], dim=0)
            embeddings = embeddings.squeeze()
            embedding_array.append(torch.mean(embeddings, dim=0))

    return embedding_array


def combine_data(features, lengths, labels):
    res = []
    for i in range(len(features)):
        res.append((features[i], lengths[i], labels[i]))
    return res


def collate(list_of_samples):
    #list_of_samples.sort(key=lambda x: x[1], reverse=True)
    input_seqs, input_seq_lengths, label_seqs = zip(*list_of_samples)
    padding_value = 0
    
    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)

    label_seqs = torch.LongTensor(label_seqs)

    return pad_input_seqs, input_seq_lengths, label_seqs
