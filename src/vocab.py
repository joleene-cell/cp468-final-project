import torch
from collections import Counter

class Vocabulary:
    def __init__(
            self,
            pad_token="<pad>",
            unk_token="<unk>",
            sos_token="<sos>",
            eos_token="<eos>"
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.itos = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.sos_token,
            3: self.eos_token
        }

        self.stoi = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sos_token: 2,
            self.eos_token: 3
        }

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list, min_freq=2):
        frequencies = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            for token in sentence:
                frequencies[token] += 1
                if frequencies[token] == min_freq:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1

    def numercalize(self, token_list):
        return [ 
            self.stoi.get(token, self.stoi[self.unk_token])
            for token in token_list
        ]

    def decode(self, index_list):
        if isinstance(index_list, torch.Tensor):
            index_list = index_list.tolist()
        return [self.itos.get(idx, self.unk_token) for idx in index_list]