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
def load_features_segmented_combined(features_path, max_len=80):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    for feature in features:
        for segment in feature:
            segment = segment.astype(np.float)
            if segment.shape[0] > max_len:
                segment = segment[:max_len, :]
            feature_array.append(autograd.Variable(torch.FloatTensor(segment)))
    
    return feature_array


# load transcripts
def load_transcripts_combined(transcripts_path):
    transcripts_array = []
    with open(transcripts_path, "r") as f:
        data = f.readlines()

    for line in data:
        transcripts_array.append(line.rstrip())

    return transcripts_array


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


def combine_data(features, transcripts):
    res = []
    for i in range(len(features)):
        res.append((features[i], transcripts[i]))
    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, label_seqs = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]
    label_seq_lengths = [len(seq) for seq in label_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    # pad label seqeucens
    pad_label_seqs = pad_sequence(label_seqs, padding_value=padding_value)

    return pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths
