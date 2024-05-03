import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel

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

print(device)


# initialize and load the embedding extractor
feature_extractor = Encoder(13, encoder_hidden_size, encoder_layers).to(device)
checkpoint = torch.load("weights/feature_extractor/state_dict_62.pt", map_location=torch.device("cpu"))
feature_extractor.load_state_dict(checkpoint["encoder"])
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()


# load features and labels
print("Loading data..")
'''
The data is in a form on Numpy array, where each audio file is segmented.
If the sentence has for example 5 words, each array with have MFCC features for each word.
Then, all the arrays are combined into one big array containing all the data.
'''
features_train, lengths_train = prepare_data.load_features_segmented_combined("path/to/data.npy")
labels_train = prepare_data.load_labels_combined("path/to/data.txt")

features_dev, lengths_dev = prepare_data.load_features_segmented_combined("path/to/data.npy")
labels_dev = prepare_data.load_labels_combined("path/to/data.txt")
print("Done...")


print("Extracting audio embeddings...")
features_train = prepare_data.extract_embeddings(feature_extractor, features_train, device)
features_dev = prepare_data.extract_embeddings(feature_extractor, features_dev, device)
print("Done...")

with open('output/embeddings_train.pickle', 'wb') as handle:
   pickle.dump(features_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('output/embeddings_test.pickle', 'wb') as handle:
   pickle.dump(features_dev, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('output/embeddings_train.pickle', 'rb') as handle:
    features_train = pickle.load(handle)
with open('output/embeddings_test.pickle', 'rb') as handle:
    features_dev = pickle.load(handle)


label2idx = {"neu": 0, "ang": 1, "sad": 2, "hap": 3}
idx2label = {0: "neu", 1: "ang", 2: "sad", 3: "hap"}

# convert labels to indices
labels_train = prepare_data.label_to_idx(labels_train, label2idx)
labels_dev = prepare_data.label_to_idx(labels_dev, label2idx)

# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train, lengths_train, labels_train)
dev_data = prepare_data.combine_data(features_dev, lengths_dev, labels_dev)


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
    criterion = nn.CrossEntropyLoss(reduction="mean")
    train(pairs_batch_train, 
            pairs_batch_dev,
            feature_extractor,
            classifier, 
            classifier_optimizer, 
            criterion, 
            batch_size, 
            num_epochs, 
            device) 
else:
    checkpoint = torch.load("weights/emo_id_model/state_dict_8.pt", map_location=torch.device("cpu"))
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()


pairs_batch_test = DataLoader(dataset=dev_data,
                    batch_size=128,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

import time
start_time = time.time()
get_predictions(classifier, pairs_batch_test) 
print("--- %s seconds ---" % (time.time() - start_time))
