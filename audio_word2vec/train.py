import random
import torch
import torch.nn.functional as F
import numpy as np


def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, scheduler_enc, scheduler_dec, criterion, num_epochs, device):
    clip = 1.0
    early_stopping = False
    stopping_patience = 0
    prev_best = 100000
    for epoch in range(100):
        if early_stopping == False:
            encoder.train()
            decoder.train()
            
            batch_loss_train = 0
            batch_loss_dev = 0

            for iteration, batch in enumerate(pairs_batch_train):
                pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths  = batch
                pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
                train_loss = 0

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                decoder_input = torch.zeros((decoder_hidden[0].size(1), 1, 13)).to(device)

                out_seqs = torch.zeros(pad_label_seqs.size(0), pad_label_seqs.size(1), pad_label_seqs.size(2)).to(device)

                for i in range(0, pad_label_seqs.size(0)):
                    output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    #decoder_input = pad_label_seqs[i].unsqueeze(0)
                    decoder_input = pad_label_seqs[i].unsqueeze(1)
                    out_seqs[i] = decoder.out(output)
                           
                train_loss = criterion(out_seqs, pad_label_seqs)
                batch_loss_train += train_loss.detach()
               
                
                # backward step
                train_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

                encoder_optimizer.step()
                decoder_optimizer.step()
             

            # CALCULATE EVALUATION
            with torch.no_grad():
                encoder.eval()
                decoder.eval()

                for _, batch in enumerate(pairs_batch_dev):
                    pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths  = batch
                    pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
                    dev_loss = 0

                    encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
                    decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                    decoder_input = torch.zeros((decoder_hidden[0].size(1), 1, 13)).to(device)

                    out_seqs = torch.zeros(pad_label_seqs.size(0), pad_label_seqs.size(1), pad_label_seqs.size(2)).to(device)

                    for i in range(0, pad_label_seqs.size(0)):
                        output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        #decoder_input = pad_label_seqs[i].unsqueeze(0)
                        decoder_input = pad_label_seqs[i].unsqueeze(1)
                        out_seqs[i] = decoder.out(output)
                               
                    dev_loss = criterion(out_seqs, pad_label_seqs)

                    batch_loss_dev += dev_loss.detach()
                    

            scheduler_enc.step()
            scheduler_dec.step()
        
            if batch_loss_dev.item() <= prev_best:
                prev_best = batch_loss_dev.item()
                stopping_patience = 0
                
                print("saving the models...")
                torch.save({
                   "encoder": encoder.state_dict(),
                   "decoder": decoder.state_dict(),
                   "encoder_optimizer": encoder_optimizer.state_dict(),
                   "decoder_optimizer": decoder_optimizer.state_dict(),
                   "scheduler_enc": scheduler_enc.state_dict(),
                   "scheduler_dec": scheduler_dec.state_dict(),
                }, "weights/optimised/state_dict_" + str(epoch+1) + ".pt")
            else:
                stopping_patience += 1
                if stopping_patience == 10:
                    early_stopping = True
            
            with open("loss/loss_optimised.txt", 'a') as f:
               f.write(str(epoch + 1) + "  " + str(batch_loss_train.item()) + "    " + str(batch_loss_dev.item()) + "\n")
            
            print("[Epoch: %d] train_loss: %.4f    val_loss: %.4f" % (epoch+1, batch_loss_train.item(), batch_loss_dev.item()))
        
        else:
            break
