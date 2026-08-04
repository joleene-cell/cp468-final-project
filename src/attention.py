"""
attention.py
Bahdanau (additive) and Luong (multiplicative) attention mechanisms for
use with encoder.py's LSTMEncoder outputs 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Build a boolean padding mask from sequence lengths
def make_src_mask(src_lengths: torch.Tensor, max_len: int, device=None) -> torch.Tensor:
    batch_size = src_lengths.size(0)
    device = device or src_lengths.device
    arange = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
    return arange < src_lengths.unsqueeze(1)

# Additive attention
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor):
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)  # (batch, 1, attn_dim)
        enc_proj = self.W_enc(encoder_outputs)  # (batch, src_len, attn_dim)
        scores = self.v(torch.tanh(dec_proj + enc_proj)).squeeze(-1)  # (batch, src_len)

        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)  # (batch, src_len)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_dim)
        return context, attn_weights

# Multiplicative attention
class LuongAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.W = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor):
        dec_proj = self.W(decoder_hidden).unsqueeze(2)  # (batch, enc_dim, 1)
        scores = torch.bmm(encoder_outputs, dec_proj).squeeze(2)  # (batch, src_len)

        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
