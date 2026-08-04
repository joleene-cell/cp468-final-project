"""
encoder.py
Embedding layer + Bidirectional LSTM Encoder for the seq2seq model
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Thin wrapper around nn.Embedding with dropout, padding-aware."""

    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int = 0, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, seq_len, embed_dim)
        return self.dropout(self.embedding(x))

# Bidirectional multi-layer LTSM encoder
class LSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.embedding = Embedding(vocab_size, embed_dim, pad_idx, dropout)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bridge: combine the final layer's forward+backward hidden/cell
        # states into a single hidden_dim vector for decoder initialization.
        self.bridge_h = nn.Linear(hidden_dim * num_directions, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * num_directions, hidden_dim)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        """
        Args:
            src: (batch, src_len) token-id LongTensor
            src_lengths: (batch,) LongTensor, true sequence lengths

        Returns:
            outputs, hidden, cell  (see class docstring)
        """
        embedded = self.embedding(src)  # (batch, src_len, embed_dim)

        # pack_padded_sequence requires lengths on CPU; enforce_sorted=False
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        # total_length pins the output back to src's original padded width
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        # outputs: (batch, src_len, hidden_dim * num_directions)

        if self.bidirectional:
            # hidden/cell shape: (num_layers * 2, batch, hidden_dim).
            # Last two "layers" are the final layer's forward/backward states.
            h_fwd, h_bwd = hidden[-2], hidden[-1]
            c_fwd, c_bwd = cell[-2], cell[-1]
            hidden_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, hidden_dim*2)
            cell_cat = torch.cat([c_fwd, c_bwd], dim=1)
        else:
            hidden_cat = hidden[-1]
            cell_cat = cell[-1]

        bridged_hidden = torch.tanh(self.bridge_h(hidden_cat))  # (batch, hidden_dim)
        bridged_cell = torch.tanh(self.bridge_c(cell_cat))

        return outputs, bridged_hidden, bridged_cell
