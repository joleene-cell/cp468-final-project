"""
data_loader.py
"""

import torch
from torch.utils.data import DataLoader
from vocab import Vocabulary
from dataset import Seq2SeqDataset, CollatePad

def get_data_loaders(
        train_src, train_tgt,
        val_src, val_tgt,
        test_src, test_tgt, 
        batch_size=32,
        min_freq=2
):
    src_vocab = Vocabulary()
    src_vocab.build_vocabulary(train_src, min_freq=min_freq)

    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocabulary(train_tgt, min_freq=min_freq)

    train_dataset = Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = Seq2SeqDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    test_dataset = Seq2SeqDataset(test_src, test_tgt, src_vocab, tgt_vocab)

    collate_fn = CollatePad(
        pad_idx_src=src_vocab.stoi["<pad>"],
        pad_idx_tgt=tgt_vocab.stoi["<pad>"]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab