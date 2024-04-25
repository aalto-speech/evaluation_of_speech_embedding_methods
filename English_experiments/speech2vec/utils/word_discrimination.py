from collections import defaultdict
import itertools
import random
import pickle
import torch.nn.functional as F
import difflib


def split_sent_to_words(labels):
    words = []
    for sent in labels:
        sent = sent.split()
        for word in sent:
            word = word.rstrip()
            words.append(word)
    return words


def find_nearest_neighbor(anchor, labels, number_neighbors):
    unique_labels = list(set(labels))
    neighbors = difflib.get_close_matches(anchor, unique_labels, number_neighbors)
    # the first neighbor is the same word
    return neighbors[-1]


def form_triplets(features, labels):
    feature_pairs = []
    label_pairs = []
    word2pos = {} 
    
    labels = split_sent_to_words(labels) 

    # create word-loc dict
    all_word_loc = defaultdict(list)
    for i, sent in enumerate(labels):
        words = sent.split()
        for word in words:
            word = word.rstrip()
            if len(all_word_loc[word]) < 2:
                all_word_loc[word].append(i)
    
    # form triplets
    triplets = []
    for word, indices in all_word_loc.items():
        if len(indices) == 2:
            nearest_neighbor = find_nearest_neighbor(word, labels, 2)
            # get location of the neighbor
            j = random.randrange(len(all_word_loc[nearest_neighbor]))
            neighbor_loc = all_word_loc[nearest_neighbor][j]
            triplets.append((indices[0], indices[1], neighbor_loc))

    
    with open("indices/word_discrimination_triplets.pickle", "wb") as handle:
        pickle.dump(triplets, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_embeddings(triplet, encoder, features, device):
    sample_1 = features[triplet[0]]
    sample_1 = sample_1.unsqueeze(1).to(device)

    sample_2 = features[triplet[1]]
    sample_2 = sample_2.unsqueeze(1).to(device)

    sample_3 = features[triplet[2]]
    sample_3 = sample_3.unsqueeze(1).to(device)
    

    encoder_output, encoder_hidden = encoder(sample_1, [int(sample_1.size(0))])
    sample_1_emb = encoder_hidden[0].sum(0).squeeze()
    
    encoder_output, encoder_hidden = encoder(sample_2, [int(sample_2.size(0))])
    sample_2_emb = encoder_hidden[0].sum(0).squeeze()
    
    encoder_output, encoder_hidden = encoder(sample_3, [int(sample_3.size(0))])
    sample_3_emb = encoder_hidden[0].sum(0).squeeze()

    return sample_1_emb, sample_2_emb, sample_3_emb


def cosine_similarity(emb_1, emb_2):    
    similarity = F.cosine_similarity(emb_1, emb_2, dim=0).item()

    return similarity


def calculate_discrimination_score(encoder, features, labels, device):
    #form_triplets(features, labels)
    with open("indices/word_discrimination_triplets.pickle", "rb") as handle:
        triplets = pickle.load(handle)
    
    total_samples = 0
    correct = 0
    incorrect = 0
    for triplet in triplets:
        emb_1, emb_2, emb_3 = get_embeddings(triplet, encoder, features, device)
        sim_same = cosine_similarity(emb_1, emb_2)
        sim_dif = cosine_similarity(emb_1, emb_3)

        total_samples += 1
        if sim_same > sim_dif:
            correct += 1
        else:
            incorrect += 1

    print("Accuracy: ", str((correct/total_samples)*100))
