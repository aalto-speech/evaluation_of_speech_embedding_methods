import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

import pickle
import operator


def get_predictions(classifier, idx2label, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    with torch.no_grad():
        predicted_labels = []
        true_labels = []

        for l, batch in enumerate(test_data):
            pad_input_seqs, label_seqs = batch
            pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)           
            
            for elem in label_seqs:
                true_labels.append(elem.item())
            
            output = classifier(pad_input_seqs)
            output = output.squeeze()
            output = torch.sigmoid(output)
            predictions = output >= 0.5
            predictions = predictions.long()

            for elem in predictions:
                predicted_labels.append(elem.item())

        print("F1: ", f1_score(true_labels, predicted_labels, average="micro"))
        print("UAR: ", balanced_accuracy_score(true_labels, predicted_labels)) 
