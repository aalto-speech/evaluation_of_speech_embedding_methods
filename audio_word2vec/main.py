import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle

import utils.prepare_data as prepare_data
from model import Encoder, Decoder
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
'''
features_train = prepare_data.load_features_segmented_combined("path/to/data")
features_dev = prepare_data.load_features_segmented_combined("path/to/data")
print("Done...")

#features_train = features_train[:256]
#features_dev = features_dev[:256]


# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train)
dev_data = prepare_data.combine_data(features_dev)


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

# initialize the Decoder
decoder = Decoder(decoder_hidden_size, batch_size).to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)
scheduler_dec = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=10, gamma=0.5)


#print(encoder)
#print(decoder)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

print("The number of trainable parameters is: %d" % (total_trainable_params_encoder + total_trainable_params_decoder))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load("weights/state_dict_10.pt")
    #encoder.load_state_dict(checkpoint["encoder"])
    #decoder.load_state_dict(checkpoint["decoder"])
    #encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
    #decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
    #scheduler_enc.load_state_dict(checkpoint["scheduler_enc"])
    #scheduler_dec.load_state_dict(checkpoint["scheduler_dec"])

    criterion = nn.MSELoss(reduction="mean")
    train(pairs_batch_train, 
            pairs_batch_dev, 
            encoder, 
            decoder,
            encoder_optimizer, 
            decoder_optimizer,
            scheduler_enc,
            scheduler_dec,
            criterion,
            num_epochs, 
            device) 
else:
    checkpoint = torch.load("weights/optimised/state_dict_59.pt", map_location=torch.device("cpu"))
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()


##################################################################################################3

# features_dev = prepare_data.load_features_segmented_combined("path/to/data")
# with open("path/to/data", "r") as f:
#     text_data = f.readlines()

#features_dev = prepare_data.load_features_segmented_combined("path/to/data")
#with open("path/to/data", "r") as f:
#    text_data = f.readlines()


#print("Clustering...")
#get_cluster_accuracy(encoder, features_dev, text_data, device)
#print("Done...")

#print("Plotting embeddings...")
#plot_embeddings(encoder, text_data, features_dev, device)
#print("Done...")

#print("Calculating cosine similarity...")
#get_cosine_similarity(encoder, text_data, features_dev, "mother", "school", device)
#print("Done...")

# print("Calculating word discrimination score...")
# calculate_discrimination_score(encoder, features_dev, text_data, device)
# print("Done")
