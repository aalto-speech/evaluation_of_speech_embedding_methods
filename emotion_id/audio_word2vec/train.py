import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


def train(pairs_batch_train, pairs_batch_dev, feature_extractor, classifier, classifier_optimizer, criterion, batch_size, num_epochs, device):
    early_stopping = False
    stopping_patience = 0
    prev_best = 0
    for epoch in range(100):
        if early_stopping == False:
            classifier.train()
            
            batch_loss_train = 0
            batch_loss_dev = 0

            predicted_labels = []
            true_labels = []

            for iteration, batch in enumerate(pairs_batch_train):
                pad_input_seqs, input_seq_lengths, label_seqs = batch
                pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)
       
                train_loss = 0

                classifier.zero_grad()
                
                output = classifier(pad_input_seqs, input_seq_lengths)
                train_loss = criterion(output, label_seqs)

                # backward step
                train_loss.backward()
                classifier_optimizer.step()
                batch_loss_train += train_loss.detach()


            # CALCULATE EVALUATION
            with torch.no_grad():
                classifier.eval()
                true_labels = []
                predicted_labels = []

                for iteration, batch in enumerate(pairs_batch_dev):
                    pad_input_seqs, input_seq_lengths, label_seqs = batch
                    pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)
       
                    dev_loss = 0

                    output = classifier(pad_input_seqs, input_seq_lengths) 
                    dev_loss = criterion(output, label_seqs)

                    batch_loss_dev += dev_loss.detach()
                    
                    # calculate F1
                    for elem in label_seqs:
                        true_labels.append(elem.item())
                    
                    output = classifier(pad_input_seqs, input_seq_lengths)
                    output = F.softmax(output, dim=-1)
                    _, topi = output.topk(1)
                
                    for elem in topi:
                        predicted_labels.append(elem.item())


            f1 = f1_score(true_labels, predicted_labels, average="micro")

            if f1 >= prev_best:
                prev_best = f1
                stopping_patience = 0
                
                print("saving the models...")
                torch.save({
                  "classifier": classifier.state_dict(),
                  "classifier_optimizer": classifier_optimizer.state_dict(),
                }, "weights/emo_id_model/state_dict_" + str(epoch+1) + ".pt")
            else:
                stopping_patience += 1
                if stopping_patience == 10:
                    early_stopping = True

            print("[Epoch: %d] train_loss: %.4f    val_loss: %.4f   micro F1: %.4f" % (epoch+1, batch_loss_train.item(), batch_loss_dev.item(), f1))

            with open("loss/loss.txt", "a") as f:
               f.write(str(epoch + 1) + "	" + str(batch_loss_train.item()) + " " + str(batch_loss_dev.item()) + " " + str(f1) + "\n")
            
        else:
            break
