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
from speechbrain.utils.profiling import profile, profile_analyst, report_time, schedule, profile_optimiser

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
        a_sig, a_lens = batch.a_sig
        p_sig, p_lens = batch.p_sig
        n_sig, n_lens = batch.n_sig

        a_outputs = self.modules.wav2vec2.forward_encoder(a_sig)
        p_outputs = self.modules.wav2vec2.forward_encoder(p_sig)
        n_outputs = self.modules.wav2vec2.forward_encoder(n_sig)
        # take a specific layer
        a_outputs = a_outputs[1]
        p_outputs = p_outputs[1]
        n_outputs = n_outputs[1]

        a_outputs = self.hparams.avg_pool(a_outputs, a_lens)
        a_outputs = a_outputs.view(a_outputs.shape[0], -1)
        p_outputs = self.hparams.avg_pool(p_outputs, p_lens)
        p_outputs = p_outputs.view(p_outputs.shape[0], -1)
        n_outputs = self.hparams.avg_pool(n_outputs, n_lens)
        n_outputs = n_outputs.view(n_outputs.shape[0], -1)

        return a_outputs, p_outputs, n_outputs


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label"""
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
            
            ## early stopping
            #if self.hparams.epoch_counter.should_stop(current=epoch, current_metric=stats["ACC"]):
            #    self.hparams.epoch_counter.current = self.hparams.epoch_counter.limit
        

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
            n_correct = 0
            n_incorrect = 0
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward().
                a_output, p_output, n_output = self.compute_forward(batch, stage=sb.Stage.TEST) 
                # get cosine similarity
                a_p_sim = F.cosine_similarity(a_output, p_output, dim=-1)
                a_n_sim = F.cosine_similarity(a_output, n_output, dim=-1)
                if a_p_sim > a_n_sim:
                    n_correct += 1
                else:
                    n_incorrect += 1

            # Get the accuracy
            accuracy = n_correct / (n_correct + n_incorrect)
            print("Accuracy: ", accuracy*100)
                
               

def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test_triplets.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test_triplets.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test_triplets.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("a_wav", "p_wav", "n_wav")
    @sb.utils.data_pipeline.provides("a_sig", "p_sig", "n_sig")
    def audio_pipeline(a_wav, p_wav, n_wav):
        a_sig = sb.dataio.dataio.read_audio(a_wav)
        p_sig = sb.dataio.dataio.read_audio(p_wav)
        n_sig = sb.dataio.dataio.read_audio(n_wav)

        return a_sig, p_sig, n_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("a_emo", "p_emo", "n_emo")
    @sb.utils.data_pipeline.provides("a_labels_encoded", "p_labels_encoded")
    def text_pipeline(a_emo, p_emo, n_emo):
        a_label = a_emo.split()[0]
        p_label = p_emo.split()[0]
        n_label = n_emo.split()[0]
        a_emo = hparams["label_encoder"].encode_sequence_torch([a_emo])
        p_emo = hparams["label_encoder"].encode_sequence_torch([p_emo])
        n_emo = hparams["label_encoder"].encode_sequence_torch([n_emo])

        yield a_emo, p_emo, n_emo

    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    #hparams["label_encoder"].update_from_didataset(train_data, output_key="a_emo")

    # save the encoder
    #hparams["label_encoder"].save(hparams["label_encoder_file"])
    
    # load the encoder
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "a_sig", "p_sig", "n_sig", "a_emo", "p_emo", "n_emo"])
    
    #train_data = train_data.filtered_sorted(sort_key="a_length", reverse=False)
    
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
        print("Evaluating...")
        ser_brain.run_inference(test_data, "error_rate", hparams["test_dataloader_options"])

if __name__ == "__main__":
    main()
