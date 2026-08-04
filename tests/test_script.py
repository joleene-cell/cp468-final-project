import torch
from torch.utils.data import DataLoader
from vocab import Vocabulary
from dataset import Seq2SeqDataset, CollatePad
from data_loader import get_data_loaders

src_data = [
    ["hello", "world"],
    ["this", "is", "a", "longer", "sentence", "test"]
]

train_src = [
    ["how", "are", "you"],
    ["i", "am", "doing", "well"],
    ["this", "is", "a", "sample", "sentence"],
    ["learning", "nlp", "is", "fun"]
]

val_src = [["how", "are", "you"]]

test_src = [["learning", "nlp"]]

tgt_data = [
    ["bonjour", "monde"],
    ["c'est", "une", "phrase", "plus", "longue"] 
]

train_tgt = [
    ["comment", "allez", "vous"],
    ["je", "vais", "bien"],
    ["c'est", "un", "exemple", "de", "phrase"],
    ["l'apprentissage", "du", "taln", "est", "amusant"]
]

val_tgt =[["comment", "allez", "vous"]]

test_tgt = [["l'apprentissage", "du", "taln"]]

train_sentences = [
    ["hello", "world", "this", "is", "a", "test"],
    ["hello", "again", "world"],
    ["this", "is", "another", "test"]
]

print("Building DataLoaders...")
train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data_loaders(
    train_src, train_tgt,
    val_src, val_tgt,
    test_src, test_tgt,
    batch_size=2,
    min_freq=1
)

print("\n--- Check 1: Vocabularies ---")
print(f"Source Vocab Size: {len(src_vocab)}")
print(f"Target Vocab Size: {len(tgt_vocab)}")
assert len(src_vocab) > 4, "Source vocabulary failed to add tokens!"
assert len(tgt_vocab) > 4, "Target vocabulary failed to add tokens!"

print("\n--- Check 2: Train Dataloader Batch Iteration ---")
for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f" Source Tensor Shape: {src_batch.shape} --> (batch_size, max_seq_len)")
    print(f" Target Tensor Shape: {tgt_batch.shape} --> (batch_size, max_seq_len)")

    assert (src_batch[:, 0] == src_vocab.stoi["<sos>"]).all(), "Missing <sos> at sequence start!"
    assert src_batch.dtype == torch.long, "Tensor must be torch.long (integer indices)!"
print("\nSuccess: data_loader.py is working as expected!")

src_vocab = Vocabulary()
src_vocab.build_vocabulary(src_data, min_freq=1)

tgt_vocab = Vocabulary()
tgt_vocab.build_vocabulary(tgt_data, min_freq=1)

dataset = Seq2SeqDataset(src_data, tgt_data, src_vocab, tgt_vocab)
collate_fn = CollatePad(src_vocab.stoi["<pad>"], tgt_vocab.stoi["<pad>"])

loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

for src_batch, tgt_batch in loader:
    print ("Source Batch Shape:", src_batch.shape)
    print("Target Batch Shape:", tgt_batch.shape)
    print("\nPadded Source Tensors:\n", src_batch)
    break

vocab = Vocabulary()
vocab.build_vocabulary(train_sentences, min_freq=2)

print("Vocab size:", len(vocab))

sample_sentence = ["hello", "world", "unknownword"]
numerical = vocab.numercalize(sample_sentence)
print("Numerical representation:", numerical)

decoded = vocab.decode(numerical)
print("Decoded back:", decoded)