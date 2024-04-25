import numpy as np
import os

def extract_audio_embeddings(encoder, features, device, save_path):
    embeddings = []
    counter = 1
    for sample in features:
        sample = sample.unsqueeze(1).to(device)
        _, _, output, hidden = encoder(sample, [int(sample.size(0))])
        
        output = output.squeeze(1)
        output = output.detach().cpu()
        # extract the training set separately and then combine the files
        #np.save(os.path.join("../data/SLURP/embeddings/linguistically_enhanced_embeddings/train", str(counter) + ".npy"), output)
        #counter += 1
        embeddings.append(output)
    embeddings = np.array(embeddings, dtype=object)
    np.save(save_path, embeddings)

