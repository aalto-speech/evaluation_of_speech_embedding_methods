import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel

import numpy as np
import pickle

import utils.prepare_data as prepare_data

from model import Encoder, AcousticDecoder, LinguisticDecoder
from config.config import *
from train import train
from utils.cosine_similarity import get_cosine_similarity
from utils.extract_embeddings import extract_audio_embeddings
from utils.word_discrimination import calculate_discrimination_score
from utils.plot_embeddings import plot_embeddings


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize BERT model
bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device)
bert_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
transcripts_train = prepare_data.load_transcripts_segmented_combined("path/to/data.txt")

features_dev = prepare_data.load_features_segmented_combined("path/to/data.npy")
transcripts_dev = prepare_data.load_transcripts_segmented_combined("path/to/data.txt")

print("Done...")



if skip_training == False:
    # extract BERT embeddings
    print("Extracting BERT embeddings...")
    transcripts_train = prepare_data.extract_bert_embeddings(bert_model, bert_tokenizer, transcripts_train, device)
    transcripts_dev = prepare_data.extract_bert_embeddings(bert_model, bert_tokenizer, transcripts_dev, device)
    print("Done...")

    # combine features and labels in a tuple
    train_data = prepare_data.combine_data(features_train, transcripts_train)
    dev_data = prepare_data.combine_data(features_dev, transcripts_dev)
    
    
    pairs_batch_train = DataLoader(dataset=train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=prepare_data.collate,
                        drop_last=True,
                        pin_memory=False)
    
    pairs_batch_dev = DataLoader(dataset=dev_data,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=prepare_data.collate,
                        drop_last=True,
                        pin_memory=False)



# initialize the Encoder
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)
scheduler_enc = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.5)

# initialize the AcousticDecoder
acoustic_decoder = AcousticDecoder(encoder_hidden_size, decoder_hidden_size, decoder_layers, encoder_layers, batch_size).to(device)
acoustic_decoder_optimizer = optim.Adam(acoustic_decoder.parameters(), lr=decoder_lr)
scheduler_ac_dec = optim.lr_scheduler.StepLR(acoustic_decoder_optimizer, step_size=10, gamma=0.5)

# initialize the LinguisticDecoder
linguistic_decoder = LinguisticDecoder(encoder_hidden_size).to(device)
linguistic_decoder_optimizer = optim.Adam(linguistic_decoder.parameters(), lr=decoder_lr)
scheduler_lin_dec = optim.lr_scheduler.StepLR(linguistic_decoder_optimizer, step_size=10, gamma=0.5)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_acoustic_decoder = sum(p.numel() for p in acoustic_decoder.parameters() if p.requires_grad)
total_trainable_params_linguistic_decoder = sum(p.numel() for p in linguistic_decoder.parameters() if p.requires_grad)

print("The number of trainable parameters is: %d" % (total_trainable_params_encoder + total_trainable_params_acoustic_decoder + total_trainable_params_linguistic_decoder))


# train
if skip_training == False:
    print("Training...")
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load("weights/state_dict_2.pt", map_location=torch.device("cpu"))
    #encoder.load_state_dict(checkpoint["encoder"])
    #acoustic_decoder.load_state_dict(checkpoint["acoustic_decoder"])
    #linguistic_decoder.load_state_dict(checkpoint["linguistic_decoder"])
    #encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
    #acoustic_decoder_optimizer.load_state_dict(checkpoint["acoustic_decoder_optimizer"])
    #linguistic_decoder_optimizer.load_state_dict(checkpoint["linguistic_decoder_optimizer"])
    #scheduler_enc.load_state_dict(checkpoint["scheduler_enc"])
    #scheduler_ac_dec.load_state_dict(checkpoint["scheduler_ac_dec"])
    #scheduler_lin_dec.load_state_dict(checkpoint["scheduler_lin_dec"])

    criterion = nn.MSELoss(reduction="mean")

    train(pairs_batch_train, 
            pairs_batch_dev, 
            encoder,
            acoustic_decoder,
            linguistic_decoder,
            encoder_optimizer,
            acoustic_decoder_optimizer,
            linguistic_decoder_optimizer,
            scheduler_enc,
            scheduler_ac_dec,
            scheduler_lin_dec,
            criterion,
            device) 
else:
    checkpoint = torch.load("weights/state_dict_62.pt", map_location=torch.device("cpu"))
    encoder.load_state_dict(checkpoint["encoder"])
    acoustic_decoder.load_state_dict(checkpoint["acoustic_decoder"])
    linguistic_decoder.load_state_dict(checkpoint["linguistic_decoder"])
    encoder.eval()
    acoustic_decoder.eval()
    linguistic_decoder.eval()



#################################################################################################


# features_dev = prepare_data.load_features_segmented_combined("path/to/data")
# with open(".path/to/data", "r") as f:
#     text_data = f.readlines()


# extract audio embeddings
#print("Extracting embeddings...")
#extract_audio_embeddings(encoder, features_dev, device, "path/to/data")
#print("Done...")

#print("Plotting embeddings...")
#plot_embeddings(encoder, text_data, features_dev, device)
#print("Done...")

#print("Calculating cosine similarity...")
#get_cosine_similarity(encoder, text_data, features_dev, "ever", "ever", device)
#print("Done...")

# print("Calculating word discrimination score...")
# calculate_discrimination_score(encoder, features_dev, text_data, device)
# print("Done")
