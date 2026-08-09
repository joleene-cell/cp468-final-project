import time
import json
import torch

from preprocess import prepare_multi30k_loaders
from encoder import LSTMEncoder
from decoder import LSTMDecoder
from attention import LuongAttention, make_src_mask
from seq2seq import Seq2Seq
from batch_utils import lengths_from_padded
from seed_utils import set_seed

MAX_GEN_LEN = 50  # generation cap, independent of any reference length


# Generates translations token-by-token with no teacher forcing
@torch.no_grad()
def greedy_decode(model, src, src_lengths, sos_idx, eos_idx, max_len=MAX_GEN_LEN):
    model.eval()
    batch_size = src.size(0)
    device = model.device

    encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
    mask = make_src_mask(src_lengths, max_len=src.size(1), device=device)

    decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
    generated = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for t in range(max_len):
        logits, hidden, cell, _ = model.decoder.forward_step(
            decoder_input, hidden, cell, encoder_outputs, mask
        )
        top1 = logits.argmax(dim=1)  # greedy: always pick the highest-probability token
        generated[:, t] = top1
        decoder_input = top1  # feed the model's own prediction back in, not the true answer

    return generated

# Converts a 1D tensor of token ids to a string
def ids_to_text(id_tensor, vocab, eos_idx, pad_idx):
    tokens = []
    for idx in id_tensor.tolist():
        if idx == eos_idx or idx == pad_idx:
            break
        tokens.append(vocab.itos.get(idx, vocab.unk_token))
    return " ".join(tokens)


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Must match train.py's hyperparameters
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    print("Loading test data and vocabularies...")
    _, _, test_loader, src_vocab, tgt_vocab, raw_test_src, raw_test_tgt = prepare_multi30k_loaders()

    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    PAD_IDX = tgt_vocab.stoi["<pad>"]
    SOS_IDX = tgt_vocab.stoi["<sos>"]
    EOS_IDX = tgt_vocab.stoi["<eos>"]

    encoder = LSTMEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, pad_idx=PAD_IDX, dropout=ENC_DROPOUT, bidirectional=True)
    attention = LuongAttention(enc_dim=HID_DIM * 2, dec_dim=HID_DIM)
    decoder = LSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM * 2, HID_DIM, attention, pad_idx=PAD_IDX, dropout=DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    print("Loading trained weights from lstm_seq2seq_best.pt ...")
    model.load_state_dict(torch.load('lstm_seq2seq_best.pt', map_location=device))
    model.eval()

    print(f"Generating translations for {len(raw_test_src)} test samples (no teacher forcing)...")
    predictions = []
    start_time = time.time()

    idx = 0
    for src_batch, tgt_batch in test_loader:
        src_batch = src_batch.to(device)
        src_lengths = lengths_from_padded(src_batch, pad_idx=0)

        generated_ids = greedy_decode(model, src_batch, src_lengths, SOS_IDX, EOS_IDX)

        for i in range(generated_ids.size(0)):
            source_text = " ".join(raw_test_src[idx])
            reference_text = " ".join(raw_test_tgt[idx])
            prediction_text = ids_to_text(generated_ids[i], tgt_vocab, EOS_IDX, PAD_IDX)

            predictions.append({
                "id": idx,
                "source": source_text,
                "reference": reference_text,
                "prediction": prediction_text,
            })
            idx += 1

    total_time = time.time() - start_time

    summary = {
        "configuration": "LSTM_BiEncoder_LuongAttention",
        "total_requests": len(predictions),
        "total_time_seconds": total_time,
        "avg_latency_seconds": total_time / len(predictions) if predictions else 0,
        "predictions": predictions,
    }

    with open("lstm_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\nDone. Generated {len(predictions)} translations in {total_time:.1f}s.")
    print("Saved to lstm_results.json -- same shape as llm_results_*.json, ready for eval.py.")
    print("\nSample outputs:")
    for p in predictions[:3]:
        print(f"  source:     {p['source']}")
        print(f"  reference:  {p['reference']}")
        print(f"  prediction: {p['prediction']}")
        print()


if __name__ == "__main__":
    main()
