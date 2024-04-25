import random
import torch
import torch.nn.functional as F
import numpy as np


def train(
        pairs_batch_train, 
        pairs_batch_dev, 
        encoder, 
        acoustic_decoder, 
        linguistic_decoder, 
        encoder_optimizer, 
        acoustic_decoder_optimizer, 
        linguistic_decoder_optimizer, 
        scheduler_enc, 
        scheduler_ac_dec, 
        scheduler_lin_dec, 
        criterion, 
        device):

    clip = 1.0

    for epoch in range(100):
        encoder.train()
        acoustic_decoder.train()
        linguistic_decoder.train()
        
        batch_loss_train = 0
        batch_loss_dev = 0

        for iteration, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths = batch
            pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
            train_loss = 0
            
            encoder_optimizer.zero_grad()
            acoustic_decoder_optimizer.zero_grad()
            linguistic_decoder_optimizer.zero_grad()

            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)

            # reconstruct the audio
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 13)).to(device)
            out_seqs = torch.zeros(pad_input_seqs.size(0), pad_input_seqs.size(1), pad_input_seqs.size(2)).to(device)
            
            for i in range(0, pad_input_seqs.size(0)):
                output, decoder_hidden = acoustic_decoder(encoder_output, decoder_input, decoder_hidden)
                decoder_input = pad_input_seqs[i].unsqueeze(0)
                out_seqs[i] = acoustic_decoder.out(output)
             
            # reconstruct the word embeddings
            encoder_hidden = encoder_hidden[0].sum(0)
            dec_out = linguistic_decoder(encoder_hidden)
            
            train_loss = criterion(out_seqs, pad_input_seqs) + criterion(dec_out, pad_label_seqs.permute(1, 0))
            batch_loss_train += train_loss.detach()
           
            # backward step
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(acoustic_decoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(linguistic_decoder.parameters(), clip)

            encoder_optimizer.step()
            acoustic_decoder_optimizer.step()
            linguistic_decoder_optimizer.step()


        # CALCULATE EVALUATION
        with torch.no_grad():
            encoder.eval()
            acoustic_decoder.eval()
            linguistic_decoder.eval()

            for _, batch in enumerate(pairs_batch_dev):
                pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths = batch
                pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
                dev_loss = 0
                
                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
                
                # reconstruct the audio
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 13)).to(device)
                out_seqs = torch.zeros(pad_input_seqs.size(0), pad_input_seqs.size(1), pad_input_seqs.size(2)).to(device)
                
                for i in range(0, pad_input_seqs.size(0)):
                    output, decoder_hidden= acoustic_decoder(encoder_output, decoder_input, decoder_hidden)
                    decoder_input = pad_input_seqs[i].unsqueeze(0)
                    out_seqs[i] = acoustic_decoder.out(output)
                 
                # reconstruct the word embeddings
                encoder_hidden = encoder_hidden[0].sum(0)
                dec_out = linguistic_decoder(encoder_hidden)
                
                dev_loss = criterion(out_seqs, pad_input_seqs) + criterion(dec_out, pad_label_seqs.permute(1, 0))
                batch_loss_dev += dev_loss.detach()


        scheduler_enc.step()
        scheduler_ac_dec.step()
        scheduler_lin_dec.step()

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, batch_loss_train.item(), batch_loss_dev.item()))


        with open('loss/loss.txt', 'a') as f:
           f.write(str(epoch + 1) + '  ' + str(batch_loss_train.item()) + ' ' + str(batch_loss_dev.item()) + '\n')

        print("saving the models...")
        torch.save({
           "encoder": encoder.state_dict(),
           "acoustic_decoder": acoustic_decoder.state_dict(),
           "linguistic_decoder": linguistic_decoder.state_dict(),
           "encoder_optimizer": encoder_optimizer.state_dict(),
           "acoustic_decoder_optimizer": acoustic_decoder_optimizer.state_dict(),
           "linguistic_decoder_optimizer": linguistic_decoder_optimizer.state_dict(),
           "scheduler_enc": scheduler_enc.state_dict(),
           "scheduler_ac_dec": scheduler_ac_dec.state_dict(),
           "scheduler_lin_dec": scheduler_lin_dec.state_dict(),
        }, "weights/state_dict_" + str(epoch+1) + ".pt")
