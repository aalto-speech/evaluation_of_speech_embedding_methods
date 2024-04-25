import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP


def collect_embeddings(encoder, text, features, word, device):
    words = []
    embeddings = []
    for line in text:
        line = line.split()
        for i in line:
            words.append(i.rstrip())

    indices = [i for i, x in enumerate(words) if x == word]

    for i in range(len(indices)):
        word_feat = features[indices[i]]
        word_feat = word_feat.unsqueeze(1).to(device)

        encoder_output, encoder_hidden = encoder(word_feat, [int(word_feat.size(0))])
        embedding = encoder_hidden[0].sum(0, keepdim=True).squeeze().detach().cpu().numpy()
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    embeddings = np.vstack(embeddings)
    embeddings = np.mean(embeddings, axis=0)
    return embeddings
    

def plot_embeddings(encoder, text, features, device):
    embeddings_mother = collect_embeddings(encoder, text, features, "mother", device)
    embeddings_father = collect_embeddings(encoder, text, features, "father", device)
    embeddings_another = collect_embeddings(encoder, text, features, "another", device)
    embeddings_other = collect_embeddings(encoder, text, features, "other", device)
    embeddings_rather = collect_embeddings(encoder, text, features, "rather", device)

    embeddings_fair = collect_embeddings(encoder, text, features, "fair", device)
    embeddings_chair = collect_embeddings(encoder, text, features, "chair", device)
    embeddings_air = collect_embeddings(encoder, text, features, "air", device)
    embeddings_hair = collect_embeddings(encoder, text, features, "hair", device)
    embeddings_pair = collect_embeddings(encoder, text, features, "pair", device)
   
    embeddings_shine = collect_embeddings(encoder, text, features, "shine", device)
    embeddings_wine = collect_embeddings(encoder, text, features, "wine", device)
    embeddings_fine = collect_embeddings(encoder, text, features, "fine", device)
    embeddings_line = collect_embeddings(encoder, text, features, "line", device)
    embeddings_nine = collect_embeddings(encoder, text, features, "nine", device)

    embeddings = np.vstack((embeddings_mother, embeddings_father, embeddings_another, embeddings_other, embeddings_rather, embeddings_fair, embeddings_chair, embeddings_air, embeddings_hair, embeddings_pair, embeddings_shine, embeddings_wine, embeddings_fine, embeddings_line, embeddings_nine))
    
    embeddings = UMAP(n_neighbors=50, min_dist=0.1, random_state=0).fit_transform(embeddings)

    df = pd.DataFrame(data=embeddings, columns=["dim1", "dim2"])
    
    df["Word"] = ["mother", "father", "another", "other", "rather", "fair", "chair", "air", "hair", "pair", "shine", "wine", "fine", "line", "nine"]
    df["Category"] = ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2", "3", "3", "3", "3", "3"]

    plot = sns.scatterplot(data=df, x="dim1", y="dim2", hue="Category")
    plot.legend([],[], frameon=False)
    plot.set_xlim(1, 4.5)

    for line in range(0, len(df["Word"])):
        plot.text(df.loc[line]["dim1"]+0.01, df.loc[line]["dim2"], df.loc[line]["Word"], horizontalalignment='left', color='black', weight='semibold', fontsize=12)

    # plot
    plt.savefig("output/embeddings.png", dpi=300)
