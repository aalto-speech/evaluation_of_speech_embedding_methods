import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

from captum.attr import IntegratedGradients, TokenReferenceBase

from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm
import librosa
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


class SER(sb.Brain):
    def compute_forward(self, embeddings, lens, stage):
        # pass through LSTM
        output, hidden = self.modules.lstm(embeddings)
        output = self.hparams.avg_pool(output, lens)
        output = output.view(output.shape[0], -1)
        # apply dropout
        output = self.hparams.dropout(output)
        # apply linear
        output = F.relu(self.modules.lin_1(output))
        output = F.relu(self.modules.lin_2(output))
        output = F.relu(self.modules.lin_3(output))

        output = self.modules.label_lin(output)
        output = self.hparams.log_softmax(output)

        return output

    
    def compute_ig(
            self,
            dataset, # Must be obtained from the dataio_function
            min_key, # We load the model with the lowest error rate
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)
        
        ig = IntegratedGradients(self.compute_forward, multiply_by_inputs=True)

        # Now we iterate over the dataset and we simply compute_forward and decode
        attributions = []
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                batch = batch.to(device)
                sig, lens = batch.sig

                embeddings = self.modules.wav2vec2(sig)
                # take a specific layer
                embeddings = embeddings[1]
    
                label, _ = batch.labels_encoded
                reference_indices = torch.zeros(embeddings.size()).to(device)

                # compute attributions and approximation delta using integrated gradients
                attributions_ig, delta = ig.attribute(embeddings, reference_indices, target=label.item(), n_steps=50, return_convergence_delta=True, additional_forward_args=(lens, sb.Stage.TEST))
                attributions_ig = attributions_ig.squeeze(0)

                attributions_ig = torch.mean(attributions_ig, dim=0)
                attributions_ig = attributions_ig.cpu().detach().numpy()
                attributions.append(attributions_ig)
                                  
        attributions = np.array(attributions)
        attributions = np.stack(attributions, axis=0).squeeze()
        attributions = np.mean(attributions, axis=0)
        attributions = np.expand_dims(attributions, axis=0)
        attributions = attributions / np.linalg.norm(attributions)
        

        # get 10% of the most important dimensions
        n_dims = int(attributions.shape[1] * 0.1)
        # get the important dimensions n dims
        important_dims = np.argsort(np.abs(attributions[0]))
        important_dims = important_dims[-n_dims:]
        print(important_dims)
        # save the dimensions in a text file
        with open("important_dims.txt", "w") as f:
            for dim in important_dims:
                f.write(str(dim) + ", ")


        # # plot heat map
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue", "white", "darkred"])
        # ax = sns.heatmap(attributions, cmap="Blues", center=0, linewidths=0.2, linecolor="black")
        # ax.tick_params(left=False, bottom=True)
        # ax.set_xticklabels('')
        # ax.set_xticks(np.arange(0, 6, 1), minor=False)
        # ax.set_xticklabels(["0-3", "3-6", "6-9", "9-12", "12-15", "15-18"], minor=False, fontsize=9)
        # ax.set(xlabel="Dimension (in scale of thousand)")
        # plt.savefig("plots/ig.png", dpi=300)


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        #sig = librosa.load(file_path, sr=16000)[0]

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("labels_encoded")
    def text_pipeline(emo):
        label = emo.split()[0]
        labels_encoded = hparams["label_encoder"].encode_sequence_torch([emo])
        yield labels_encoded

    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["label_encoder"].update_from_didataset(train_data, output_key="emo")

    # save the encoder
    hparams["label_encoder"].save(hparams["label_encoder_file"])
    
    # load the encoder
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "labels_encoded"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    # CuDNN with RNN doesn't support gradient computation in eval mode that's why we need to disable cudnn for RNN in eval mode
    torch.backends.cudnn.enabled=False
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Trainer initialization
    ser_brain = SER(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

    # Dataset creation
    train_data, valid_data, test_data = data_prep("/path/to/data", hparams)

    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ser_brain.fit(
            ser_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # Evaluate
        print("Evaluating")
        ser_brain.compute_ig(test_data, "error_rate", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
