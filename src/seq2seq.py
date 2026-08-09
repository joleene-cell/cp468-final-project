import random
import torch
import torch.nn as nn
from attention import make_src_mask

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ):
        batch_size = src.size(0)
        max_tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_tgt_len, tgt_vocab_size, device=self.device)

        # Encode the source sequences
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # Create source mask for attention padding
        mask = make_src_mask(src_lengths, max_len=src.size(1), device=self.device)

        # First input to decoder is the <sos> token
        decoder_input = tgt[:, 0]

        for t in range(1, max_tgt_len):
            # Pass through single decoder step
            logits, hidden, cell, _ = self.decoder.forward_step(
                decoder_input, hidden, cell, encoder_outputs, mask
            )
            
            # Store predictions
            outputs[:, t, :] = logits

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)

            # Decide if next input is true target or the model's highest probability prediction
            decoder_input = tgt[:, t] if teacher_force else top1

        return outputs