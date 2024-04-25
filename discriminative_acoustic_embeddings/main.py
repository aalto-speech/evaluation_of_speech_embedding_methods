import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
#import pickle
import pickle5 as pickle

import utils.prepare_data as prepare_data
from model import Encoder
from config.config import *
from train import train
from utils.cosine_similarity import get_cosine_similarity
from utils.clustering import get_cluster_accuracy
from utils.word_discrimination import calculate_discrimination_score
from utils.plot_embeddings import plot_embeddings


torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load features and labels
print("Loading data..")
'''
The data is in a form on Numpy array, where each audio file is segmented.
If the sentence has for example 5 words, each array with have MFCC features for each word.
Then, all the arrays are combined into one big array containing all the data.

The transcripts are combined in one file, where each line corresponds to the transcript of the corresponding audio file.
'''
features_train = prepare_data.load_features_segmented_combined("path/to/data.npy")
labels_train = prepare_data.load_transcripts_segmented_combined("path/to/data.txt")

features_dev = prepare_data.load_features_segmented_combined("path/to/data.npy")
labels_dev = prepare_data.load_transcripts_segmented_combined("path/to/data.txt")

print("Done...")


print("Cleaning the data...")
features_train, labels_train = prepare_data.clean_data(features_train, labels_train)
features_dev, labels_dev = prepare_data.clean_data(features_dev, labels_dev)
print("Done...")


print("Creating dictionaries...")
# generate dictionary mapping
idx2word_train = prepare_data.get_idx_word_dict(labels_train)
idx2word_dev = prepare_data.get_idx_word_dict(labels_dev)
# save the dictionary
with open('indices/idx2word_train.pkl', 'wb') as f:
   pickle.dump(idx2word_train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('indices/idx2word_dev.pkl', 'wb') as f:
   pickle.dump(idx2word_dev, f, protocol=pickle.HIGHEST_PROTOCOL)

# load the dictionary
with open('indices/idx2word_dev.pkl', 'rb') as f:
    idx2word_train = pickle.load(f)
word2idx_train = {v: k for k, v in idx2word_train.items()}

with open('indices/idx2word_dev.pkl', 'rb') as f:
    idx2word_dev = pickle.load(f)
word2idx_dev = {v: k for k, v in idx2word_dev.items()}
print("Done...")


# convert labels to indices and remove samples that have words that appear only once
print("Indexing the labels...")
labels_train  = prepare_data.label_to_idx(labels_train, word2idx_train)
labels_dev = prepare_data.label_to_idx(labels_dev, word2idx_dev)
print("Done...")


print("Forming pairs...")
features_train_pairs, labels_train_pairs, word2pos_train = prepare_data.form_pairs(features_train, labels_train, idx2word_train)
features_dev_pairs, labels_dev_pairs, word2pos_dev = prepare_data.form_pairs(features_dev, labels_dev, idx2word_dev)
print("Done...")


# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train_pairs, labels_train_pairs)
dev_data = prepare_data.combine_data(features_dev_pairs, labels_dev_pairs)


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


# initialize the Encoder
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)
scheduler_enc = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.5)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

print("The number of trainable parameters is: %d" % (total_trainable_params_encoder))

# train
if skip_training == False:
    print("Training...")
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load("weights/state_dict_24.pt")
    #encoder.load_state_dict(checkpoint["encoder"])
    #encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
    #scheduler_enc.load_state_dict(checkpoint["scheduler_enc"])
   

    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=-1), margin=0.4, reduction="mean")
    train(
          pairs_batch_train, 
          pairs_batch_dev, 
          encoder, 
          encoder_optimizer, 
          scheduler_enc,
          criterion,
          idx2word_train,
          idx2word_dev,
          word2pos_train,
          word2pos_dev,
          features_train,
          features_dev,
          labels_train,
          labels_dev,
          device) 
else:
    checkpoint = torch.load("weights/state_dict_28.pt", map_location=torch.device("cpu"))
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()


##################################################################################################3

# features_dev = prepare_data.load_features_segmented_combined("path/to/data.npy")
# with open("path/to/data.txt", "r") as f:
#     text_data = f.readlines()

#print("Clustering...")
#get_cluster_accuracy(encoder, features_dev, text_data, device)
#print("Done...")

#print("Plotting embeddings...")
#plot_embeddings(encoder, text_data, features_dev, device)
#print("Done...")

#print("Calculating cosine similarity...")
#get_cosine_similarity(encoder, text_data, features_dev, "ever", "never", device)
#print("Done...")

# print("Calculating word discrimination score...")
# calculate_discrimination_score(encoder, features_dev, text_data, device)
# print("Done")
