import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def get_embeddings(encoder, features, device):
    embeddings = []
    for feature in features:
        feature = feature.unsqueeze(1).to(device)
        encoder_output, encoder_hidden = encoder(feature, [int(feature.size(0))])
        embedding = encoder_hidden[0].sum(0, keepdim=True).squeeze()
        embeddings.append(embedding.detach().cpu().numpy())

    embeddings = np.array(embeddings)
    embeddings = np.vstack(embeddings)
    
    return embeddings


def get_index_dictionary(transcripts):
    word2idx = {}
    for sent in transcripts:
        sent = sent.split()
        for word in sent:
            word = word.rstrip()
            if word not in word2idx.keys():
                word2idx[word] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    
    return word2idx, idx2word


def generate_labels(transcripts, word2idx):
    labels = []
    for sent in transcripts:
        sent = sent.split()
        for word in sent:
            word = word.rstrip()
            labels.append(word2idx[word])
    labels = np.array(labels)

    return labels


def get_frequent_samples(embeddings, labels, frequency):
    unique, counts = np.unique(labels, return_counts=True)
    label_count = dict(zip(unique, counts))
    label_count = {k: v for k, v in sorted(label_count.items(), key=lambda item: item[1])}
    
    final_emb = []
    final_labels = []
    for i in range(len(embeddings)):
        if label_count[labels[i]] >= frequency:
            final_emb.append(embeddings[i])
            final_labels.append(labels[i])
    final_emb = np.array(final_emb)
    final_labels = np.array(final_labels)

    return final_emb, final_labels

 
def cluster_data(model, embeddings):
    model.fit(embeddings)
    predictions = model.labels_
    return predictions


def get_cluster_accuracy(encoder, features, transcripts, device):
    print("Generating embeddings...")
    embeddings = get_embeddings(encoder, features, device)
    word2idx, idx2word = get_index_dictionary(transcripts)
    labels = generate_labels(transcripts, word2idx)
    embeddings, labels = get_frequent_samples(embeddings, labels, 150)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Number of clusters: %.d" % (len(unique_labels)))

    kmeans = KMeans(n_clusters=len(unique_labels), random_state=0)
    print("Learning clustering...")
    predictions = cluster_data(kmeans, embeddings)

    print("Accuracy: %.2f" % (accuracy_score(labels, predictions)*100))
