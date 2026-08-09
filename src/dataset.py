"""
dataset.py
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
class Seq2SeqDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = self.src_sentences[idx]
        tgt_tokens = self.tgt_sentences[idx]

        src_numerical = (
            [self.src_vocab.stoi[self.src_vocab.sos_token]]
            + self.src_vocab.numercalize(src_tokens)
            + [self.src_vocab.stoi[self.src_vocab.eos_token]]
        )

        tgt_numerical = (
            [self.tgt_vocab.stoi[self.tgt_vocab.sos_token]]
            + self.tgt_vocab.numercalize(tgt_tokens)
            + [self.tgt_vocab.stoi[self.tgt_vocab.eos_token]]
        )

        return torch.tensor(src_numerical, dtype=torch.long), torch.tensor(tgt_numerical, dtype=torch.long)

class CollatePad:
    def __init__(self, pad_idx_src, pad_idx_tgt):
        self.pad_idx_src = pad_idx_src
        self.pad_idx_tgt = pad_idx_tgt

    def __call__(self, batch):
        src_list = [item[0] for item in batch]
        tgt_list = [item[1] for item in batch]

        src_padded = pad_sequence(src_list, batch_first=True, padding_value=self.pad_idx_src)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=self.pad_idx_tgt)

        return src_padded, tgt_padded