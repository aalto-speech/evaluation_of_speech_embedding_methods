import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm
import librosa


class SER(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        sig, lens = batch.sig
        
        outputs = self.modules.wav2vec2.forward_encoder(sig)
        # take a specific layer
        outputs = outputs[7]

        # take the most important dimensions
        important_dims = [590, 180, 128, 6, 107, 426, 436, 453, 713, 39, 457, 648, 7, 71, 557, 
                          197, 472, 60, 43, 468, 418, 131, 391, 715, 553, 430, 685, 387, 93, 
                          704, 67, 460, 12, 679, 133, 442, 571, 469, 482, 413, 52, 37, 333, 
                          40, 92, 404, 323, 4, 677, 14, 528, 172, 30, 139, 452, 691, 151, 
                          399, 38, 114, 166, 41, 325, 77, 23, 720, 408, 116, 389, 90, 44, 
                          34, 397, 578, 35, 463]
        outputs = outputs[:, :, important_dims]

        # last dim will be used for AdaptativeAVG pool
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        
        outputs = self.modules.label_lin(outputs)
        outputs = self.hparams.log_softmax(outputs)

        return outputs 


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        emoid, label_lens = batch.labels_encoded

        """to meet the input form of nll loss"""
        emoid = emoid.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

        
    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            # self.wav2vec2_optimizer.step()
            self.optimizer.step()

        # self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()

        return loss.detach()


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            #self.error_metrics = self.hparams.accuracy_computer()
    

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
                #"ACC": self.error_metrics.summarize(),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )
            

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    
    def run_inference(
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

        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            true_labels = []
            pred_labels = []
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                output = self.compute_forward(batch, stage=sb.Stage.TEST) 
                
                # SER prediction
                topi, topk = output.topk(1)
                topk = topk.squeeze()

                labels, label_lens = batch.labels_encoded
                labels = labels.squeeze()

                topk = topk.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                
                try: 
                    for elem in labels:
                        true_labels.append(elem)

                    for elem in topk:
                        pred_labels.append(elem)
                except:
                    #print(labels)
                    true_labels.append(labels)
                    pred_labels.append(topk)
                

            true_labels = np.array(true_labels)
            pred_labels = np.array(pred_labels)
 
            print("F1 score: ", round(f1_score(true_labels, pred_labels, average="micro"), 4) * 100)
            print("UAR: ", round(balanced_accuracy_score(true_labels, pred_labels), 4) * 100)

       

def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        # sig = sb.dataio.dataio.read_audio(file_path)
        sig = librosa.load(file_path, sr=16000)[0]
        # limit the audio to 5 seconds
        sig = sig[:5*16000]
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("labels_encoded")
    def text_pipeline(label):
        label = label.split()[0]
        labels_encoded = hparams["label_encoder"].encode_sequence_torch([label])
        yield labels_encoded

    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    # hparams["label_encoder"].update_from_didataset(train_data, output_key="label")

    # save the encoder
    # hparams["label_encoder"].save(hparams["label_encoder_file"])
    
    # load the encoder
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "labels_encoded"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data




def main(device="cuda"):
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
    train_data, valid_data, test_data = data_prep("path/to/data", hparams)

    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ###ser_brain.checkpointer.delete_checkpoints(num_to_keep=0)
        ser_brain.fit(
            ser_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # evaluate
        print("Evaluating")
        ser_brain.run_inference(test_data, "error_rate", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
