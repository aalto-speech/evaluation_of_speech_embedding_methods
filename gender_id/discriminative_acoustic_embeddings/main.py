import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle5 as pickle

import utils.prepare_data as prepare_data
from model import Classifier
from feature_extractor_model import Encoder
from config.config import *
from train import train
from get_predictions import get_predictions


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Gender ID")
print(device)


# initialize and load the embedding extractor
feature_extractor = Encoder(13, encoder_hidden_size, encoder_layers).to(device)
checkpoint = torch.load("weights/feature_extractor/state_dict_28.pt", map_location=torch.device("cpu"))
feature_extractor.load_state_dict(checkpoint["encoder"])
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()


# load features and labels
print("Loading data..")
features_train  = prepare_data.load_features_segmented_combined("path/to/data.npy")
labels_train = prepare_data.load_labels_combined("path/to/data.txt")

features_dev = prepare_data.load_features_segmented_combined("path/to/data.npy")
labels_dev = prepare_data.load_labels_combined("path/to/data.txt")

print("Done...")

print("Extracting audio embeddings...")
features_train = prepare_data.extract_embeddings(feature_extractor, features_train, device)
features_dev = prepare_data.extract_embeddings(feature_extractor, features_dev, device)
print("Done...")

with open('output/embeddings_train.pickle', 'wb') as handle:
   pickle.dump(features_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('output/embeddings_dev.pickle', 'wb') as handle:
   pickle.dump(features_dev, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('output/embeddings_train.pickle', 'rb') as handle:
    features_train = pickle.load(handle)
with open('output/embeddings_dev.pickle', 'rb') as handle:
    features_dev = pickle.load(handle)



label2idx = {"m": 0, "f": 1}
idx2label = {0: "m", 1: "f"}

# convert labels to indices
labels_train = prepare_data.label_to_idx(labels_train, label2idx)
labels_dev = prepare_data.label_to_idx(labels_dev, label2idx)

# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train, labels_train)
dev_data = prepare_data.combine_data(features_dev, labels_dev)


pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    drop_last=True,
                    pin_memory=True)

pairs_batch_dev = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    drop_last=True,
                    pin_memory=True)



# initialize the Classifier
classifier = Classifier(input_size, output_size).to(device)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr)

total_trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

print("The number of trainable parameters is: %d" % (total_trainable_params))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load("weights/state_dict_58.pt")
    #classifier.load_state_dict(checkpoint["classifier"])
    #classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    train(pairs_batch_train, 
            pairs_batch_dev,
            classifier,
            classifier_optimizer,
            criterion,
            batch_size, 
            num_epochs, 
            device) 
else:
    checkpoint = torch.load("weights/gender_model/state_dict_2.pt", map_location=torch.device("cpu"))
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()


pairs_batch_test = DataLoader(dataset=dev_data,
                    batch_size=128,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)



print("Evaluating...")
get_predictions(classifier, idx2label, pairs_batch_test) 
print("Done...")
