import torch
import torch.nn.functional as F
import numpy as np


def cosine_similarity(encoder, sample_1, sample_2):
    encoder_output, encoder_hidden = encoder(sample_1, [int(sample_1.size(0))])
    sample_1_emb = encoder_hidden[0].sum(0, keepdim=True).squeeze()

    encoder_output, encoder_hidden = encoder(sample_2, [int(sample_2.size(0))])
    sample_2_emb = encoder_hidden[0].sum(0, keepdim=True).squeeze()

    similarity = F.cosine_similarity(sample_1_emb, sample_2_emb, dim=0).item()

    return similarity


def get_cosine_similarity(encoder, text_data, features, word_1, word_2, device):
    words = []
    for line in text_data:
        line = line.split()
        for i in line:
            words.append(i.rstrip())
    
    indices = [i for i, x in enumerate(words) if x == word_1]
    indices_2 = [i for i, x in enumerate(words) if x == word_2]
    
    similarities = []
    num_pairs = 0

    for i in range(len(indices)):
        if word_1 == word_2:
            for j in range(i+1, len(indices)):
                num_pairs += 1

                sample_1 = features[indices[i]]
                sample_1 = sample_1.unsqueeze(1).to(device)
                
                sample_2 = features[indices[j]]
                sample_2 = sample_2.unsqueeze(1).to(device)
                similarity = cosine_similarity(encoder, sample_1, sample_2)
                similarities.append(similarity)
        else:
            for j in range(len(indices_2)):
                num_pairs += 1

                sample_1 = features[indices[i]]
                sample_1 = sample_1.unsqueeze(1).to(device)
                
                sample_2 = features[indices_2[j]]
                sample_2 = sample_2.unsqueeze(1).to(device)
                similarity = cosine_similarity(encoder, sample_1, sample_2)
                similarities.append(similarity)
    
    similarities = np.array(similarities)
    print("Number of pairs: ", num_pairs)
    print(np.mean(similarities))



