import random
import torch
import torch.nn.functional as F
import numpy as np
import random


def train(pairs_batch_train, pairs_batch_dev, encoder, encoder_optimizer, scheduler_enc, criterion, idx2word_train, idx2word_dev, word2pos_train, word2pos_dev, features_train, features_dev, labels_train, labels_dev, device):
    clip = 1.0
    early_stopping = False
    prev_best = 100000

    for epoch in range(100):
        if early_stopping == False:
            encoder.train()
            
            batch_loss_train = 0
            batch_loss_dev = 0 

            for iteration, batch in enumerate(pairs_batch_train):
                pad_input_seqs_1, pad_input_seqs_2, label_seqs_1, label_seqs_2  = batch
                pad_input_seqs_1, pad_input_seqs_2, label_seqs_1, label_seqs_2 = pad_input_seqs_1.to(device), pad_input_seqs_2.to(device), label_seqs_1.to(device), label_seqs_2.to(device)
               
                ################################################################
                # CREATE NEGATIVE SAMPLES
                batch_size = pad_input_seqs_1.size(1)  
                # sample
                negative_samples = []
                for sample in range(batch_size):
                    # pick a label
                    different_label = False
                    # pick until the label is different from the anchor
                    while different_label == False:
                        anchor_word = idx2word_train[label_seqs_1[sample].item()]
                        sample_idx = random.sample(list(idx2word_train), 1)
                        negative_word = idx2word_train[sample_idx[0]]
                        if anchor_word != negative_word:
                            different_label = True
                            negative_word_pos = word2pos_train[negative_word]
                            negative_sample = random.sample(negative_word_pos, 1)[0]
                            negative_samples.append(features_train[negative_sample].unsqueeze(0).cpu().detach().numpy())

                negative_samples = np.array(negative_samples)
                pad_input_seqs_3 = np.vstack(negative_samples)
                pad_input_seqs_3 = torch.from_numpy(pad_input_seqs_3)
                pad_input_seqs_3 = pad_input_seqs_3.permute(1, 0, 2).to(device) 
                ###################################################################

                train_loss = 0

                encoder_optimizer.zero_grad()

                embedding_1 = encoder(pad_input_seqs_1)
                embedding_2 = encoder(pad_input_seqs_2)
                embedding_3 = encoder(pad_input_seqs_3)
                
                train_loss_1 = criterion(embedding_1, embedding_2, embedding_3)
                train_loss_2 = criterion(embedding_2, embedding_1, embedding_3)
                train_loss = train_loss_1 + train_loss_2
                batch_loss_train += train_loss.detach()
               
                
                # backward step
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
                encoder_optimizer.step()
             

            
            # CALCULATE EVALUATION
            with torch.no_grad():
                encoder.eval()
                for _, batch in enumerate(pairs_batch_dev):
                    pad_input_seqs_1, pad_input_seqs_2, label_seqs_1, label_seqs_2  = batch
                    pad_input_seqs_1, pad_input_seqs_2, label_seqs_1, label_seqs_2 = pad_input_seqs_1.to(device), pad_input_seqs_2.to(device), label_seqs_1.to(device), label_seqs_2.to(device)    

                    ################################################################
                    # CREATE NEGATIVE SAMPLES
                    batch_size = pad_input_seqs_1.size(1)  
                    # sample
                    negative_samples = []
                    for sample in range(batch_size):
                        # pick a label
                        different_label = False
                        # pick until the label is different from the anchor
                        while different_label == False:
                            anchor_word = idx2word_dev[label_seqs_1[sample].item()]
                            sample_idx = random.sample(list(idx2word_dev), 1)
                            negative_word = idx2word_dev[sample_idx[0]]
                            if anchor_word != negative_word:
                                different_label = True
                                negative_word_pos = word2pos_dev[negative_word]
                                negative_sample = random.sample(negative_word_pos, 1)[0]
                                negative_samples.append(features_dev[negative_sample].unsqueeze(0).cpu().detach().numpy())

                    negative_samples = np.array(negative_samples)
                    pad_input_seqs_3 = np.vstack(negative_samples)
                    pad_input_seqs_3 = torch.from_numpy(pad_input_seqs_3)
                    pad_input_seqs_3 = pad_input_seqs_3.permute(1, 0, 2).to(device) 
                    ###################################################################

                    dev_loss = 0

                    encoder_optimizer.zero_grad()
                    
                    embedding_1 = encoder(pad_input_seqs_1)
                    embedding_2 = encoder(pad_input_seqs_2)
                    embedding_3 = encoder(pad_input_seqs_3)
                    
                    dev_loss_1 = criterion(embedding_1, embedding_2, embedding_3)
                    dev_loss_2 = criterion(embedding_2, embedding_1, embedding_3)
                    dev_loss = dev_loss_1 + dev_loss_2
                    batch_loss_dev += dev_loss.detach()


            scheduler_enc.step()
            
            
            if batch_loss_dev.item() <= prev_best:
                prev_best = batch_loss_dev.item()
                stopping_patience = 0
             
                print("saving the models...")
                torch.save({
                   "encoder": encoder.state_dict(),
                   "encoder_optimizer": encoder_optimizer.state_dict(),
                   "scheduler_enc": scheduler_enc.state_dict(),
                }, 'weights/state_dict_' + str(epoch+1) + '.pt')
                
            else:
                stopping_patience += 1
                if stopping_patience == 10:
                    early_stopping = True
            
            with open("loss/loss.txt", 'a') as f:
               f.write(str(epoch + 1) + "  " + str(batch_loss_train.item()) + "    " + str(batch_loss_dev.item()) + "\n")

            
            print("[Epoch: %d] train_loss: %.4f    val_loss: %.4f" % (epoch+1, batch_loss_train.item(), batch_loss_dev.item())) 
        else:
            break
