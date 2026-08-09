import torch
import torch.nn as nn
from encoder import Embedding

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_dim: int,
        dec_dim: int,
        attention: nn.Module,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.pad_idx = pad_idx
        
        self.embedding = Embedding(vocab_size, embed_dim, pad_idx=pad_idx, dropout=dropout)
        
        # LSTMCell instead of LSTM to compute step-by-step for attention
        self.lstm_cell = nn.LSTMCell(embed_dim, dec_dim)
        
        # Linear layer to combine context vector and decoder hidden state
        self.fc_concat = nn.Linear(enc_dim + dec_dim, dec_dim)
        
        # Linear layer to project to the target vocabulary
        self.fc_out = nn.Linear(dec_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor, 
                     encoder_outputs: torch.Tensor, mask: torch.Tensor):
        """
        Executes a single step of the decoder.
        Args:
            input_token: (batch_size,) - The token ID for the current time step.
            hidden: (batch_size, dec_dim) - Decoder hidden state.
            cell: (batch_size, dec_dim) - Decoder cell state.
            encoder_outputs: (batch_size, src_len, enc_dim) - Outputs from the encoder.
            mask: (batch_size, src_len) - Boolean mask for padded source tokens.
        """
        # 1. Embed the input token
        # input_token is 1D (batch_size). Unsqueeze to (batch_size, 1) for embedding, then squeeze back
        embedded = self.embedding(input_token.unsqueeze(1)).squeeze(1)  # (batch_size, embed_dim)
        
        # 2. LSTM step update
        hidden_next, cell_next = self.lstm_cell(embedded, (hidden, cell))  # (batch_size, dec_dim)
        
        # 3. Calculate attention context
        # attention forward returns context vector and attention weights
        context, attn_weights = self.attention(hidden_next, encoder_outputs, mask)  # context: (batch_size, enc_dim)
        
        # 4. Concatenate and pass through a linear + tanh layer
        concat_input = torch.cat([hidden_next, context], dim=1)  # (batch_size, enc_dim + dec_dim)
        concat_output = torch.tanh(self.fc_concat(concat_input))  # (batch_size, dec_dim)
        concat_output = self.dropout(concat_output)
        
        # 5. Project to vocabulary size to get logits
        logits = self.fc_out(concat_output)  # (batch_size, vocab_size)
        
        return logits, hidden_next, cell_next, attn_weights