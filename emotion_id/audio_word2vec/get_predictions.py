import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

import pickle
import operator


def get_predictions(classifier, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Evaluating...")

    with torch.no_grad():
        predicted_labels = []
        true_labels = []

        for l, batch in enumerate(test_data):
            pad_input_seqs, input_seq_lengths, label_seqs = batch
            pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)
            
            for elem in label_seqs:
                true_labels.append(elem.item())
            
            output = classifier(pad_input_seqs, input_seq_lengths)
            output = F.softmax(output, dim=-1)
            topk, topi = output.topk(1)

            for elem in topi:
                predicted_labels.append(elem.item())
    
    print("F1: ", f1_score(true_labels, predicted_labels, average="micro"))
    print("UAR: ", balanced_accuracy_score(true_labels, predicted_labels))

    #with open("../statistical_significance/predictions/audio_word2vec.txt", "w") as f:
    #    for elem in predicted_labels:
    #        f.write(str(elem))
    #        f.write("\n")
