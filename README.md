# Member 2: Encoder & Attention Architecture

## Files
| File | Purpose |
|---|---|
| `encoder.py` | `Embedding` + `LSTMEncoder` (bidirectional, from scratch on top of `nn.LSTM`/`nn.Embedding`) |
| `attention.py` | `BahdanauAttention`, `LuongAttention`, and `make_src_mask` helper |
| `seed_utils.py` | `set_seed()` — locks Python `random`, NumPy, and PyTorch (CPU+CUDA) RNGs |
| `test_encoder_attention.py` | Unit tests: shape checks, NaN checks, mask correctness, attention weight sums (synthetic tensors) |
| `sanity_check.py` | Forward pass on a mock batch + gradient clipping demo (synthetic tensors) |
| `batch_utils.py` | `lengths_from_padded()` — derives lengths from Member 1's padded-only batches |
| `integration_test.py` | **Runs against Member 1's real `data_loader.py`/`dataset.py`/`vocab.py`** — proves actual compatibility, not just mocks |
| `requirements.txt` | Dependencies for this module (merge into the team-wide file) |

## How to run

```bash
pip install -r requirements.txt

# unit tests
pytest test_encoder_attention.py -v

# forward-pass + gradient clipping sanity check
python sanity_check.py
```

Both commands should complete with all tests passing / a final "Sanity
check PASSED" message, with no NaNs and gradient norm correctly bounded
after clipping.

## Integration contract (confirmed against Member 1's actual code)

Verified directly against the real `data_loader.py` / `dataset.py` / `vocab.py`
(see `integration_test.py`, which runs this module against their real files,
not mocks):

- `vocab.py`: `<pad>=0, <unk>=1, <sos>=2, <eos>=3` — matches this module's `pad_idx=0` default.
- `dataset.py`'s `Seq2SeqDataset` already wraps every sequence in `<sos>...<eos>`.
- **Gap found:** `data_loader.py`'s `get_data_loaders()` / `CollatePad` returns
  only `(src_padded, tgt_padded)` — **no lengths tensor**. `LSTMEncoder`
  needs true lengths for `pack_padded_sequence`, so rather than asking
  Member 1 to change their collate function, `batch_utils.py` derives
  lengths directly from the padded batch (counting non-`pad_idx` positions,
  safe since `<pad>` is reserved index 0 and never a real token).

Example call (this is literally what `integration_test.py` does):

```python
from data_loader import get_data_loaders          # Member 1
from encoder import LSTMEncoder                    # Member 2
from attention import BahdanauAttention, make_src_mask  # Member 2
from batch_utils import lengths_from_padded         # glue helper

train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data_loaders(
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, batch_size=32
)

encoder = LSTMEncoder(vocab_size=len(src_vocab), embed_dim=256, hidden_dim=512,
                       pad_idx=src_vocab.stoi["<pad>"], bidirectional=True)
attn = BahdanauAttention(enc_dim=512 * 2, dec_dim=512, attn_dim=256)

src_batch, tgt_batch = next(iter(train_loader))
src_lengths = lengths_from_padded(src_batch, pad_idx=src_vocab.stoi["<pad>"])

encoder_outputs, hidden, cell = encoder(src_batch, src_lengths)
mask = make_src_mask(src_lengths, max_len=src_batch.size(1))

# inside the decoder's per-timestep loop:
context, attn_weights = attn(decoder_hidden, encoder_outputs, mask)
```

**Worth raising with Member 1 anyway:** it would be cleaner long-term if
`CollatePad` returned lengths directly (it already has the un-padded
`src_list`/`tgt_list` before padding, so `[len(s) for s in src_list]` is
free) rather than every downstream consumer recomputing them from padded
tensors. Not blocking — `batch_utils.py` handles it either way — but flag
it for the team.

`hidden`/`cell` are already bridged down to `(batch, hidden_dim)` and can
be used directly as the decoder LSTM's initial `(h_0, c_0)` (unsqueeze to
add a `num_layers` dim of 1 if the decoder LSTM expects `(1, batch, hidden_dim)`).

## Design notes for the report

- **Bidirectional encoder + unidirectional decoder bridge**: rather than
  feeding the decoder a `2*hidden_dim` state directly, a learned linear
  layer (`bridge_h`/`bridge_c`) projects the concatenated
  forward+backward final states down to `hidden_dim`, followed by `tanh`.
  This keeps the decoder architecture simple (matches `hidden_dim`) while
  still letting the decoder's initial state benefit from both directions
  of context.
- **Masking**: `pad_packed_sequence` is called with `total_length=src.size(1)`
  to guarantee `encoder_outputs` always matches the original padded width,
  so `make_src_mask` (built from the same `src_lengths`) lines up correctly
  regardless of which sequence in a batch happens to be longest.
- **Two attention variants implemented** (Bahdanau additive, Luong
  multiplicative) sharing an identical interface — this makes attention
  choice a one-line ablation (`config.attention_type`) if the team wants
  to report it as a comparison point in section 4.1.
