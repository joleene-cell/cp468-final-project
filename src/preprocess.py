import nltk
from datasets import load_dataset
from data_loader import get_data_loaders

nltk.download('punkt')
nltk.download('punkt_tab')

def prepare_multi30k_loaders(batch_size=32, min_freq=2):
    print("1. Downloading Multi30k dataset...")
    dataset = load_dataset("bentrevett/multi30k")

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print("2. Tokenizing sentences...")
    def tokenize_pairs(split):
        src_sentences = [nltk.word_tokenize(item["en"].lower()) for item in split]
        tgt_sentences = [nltk.word_tokenize(item["de"].lower()) for item in split]
        return src_sentences, tgt_sentences

    train_src, train_tgt = tokenize_pairs(train_data)
    val_src, val_tgt = tokenize_pairs(val_data)
    test_src, test_tgt = tokenize_pairs(test_data)
    print("3. Building Vocabularies and DataLoaders...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data_loaders(
        train_src, train_tgt,
        val_src, val_tgt,
        test_src, test_tgt,
        batch_size=batch_size,
        min_freq=min_freq
    )

    print(f"Data ready! Loaded {len(train_src)} training samples.")
    print(f"  Source Vocab Size: {len(src_vocab)}")
    print(f"  Target Vocab Size: {len(tgt_vocab)}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, test_src, test_tgt

if __name__ == "__main__":
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, raw_test_src, raw_test_tgt = prepare_multi30k_loaders()