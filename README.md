# CP468 Final Project: LSTM vs. LLM

Comparing a classical attention-based LSTM encoder-decoder (trained from scratch in PyTorch)
against a locally-run open-weights LLM baseline (Llama 3.1, 8B parameters) on English→German
translation, using the Multi30k dataset.

## Team & Status

| Member | Component | Status |
|---|---|---|
| 1 | Data pipeline & preprocessing | ✅ Done |
| 2 | Encoder & attention architecture | ✅ Done, tested, integrated |
| 3 | Decoder & training loop | ✅ Done — model trained, checkpoint saved |
| 4 | LLM baseline | ✅ Done — full evaluation run complete |
| 5 | Inference, evaluation & qualitative analysis | ⏳ In progress |

## Setup

```bash
git clone https://github.com/joleene-cell/cp468-final-project.git
cd cp468-final-project
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing Results

All commands assume the virtual environment is active and are run from the project root.
`PYTHONPATH=src` lets modules in `tests/` and the root import from `src/`.

**1. Run the encoder/attention test suite:**
```bash
PYTHONPATH=src pytest tests/test_encoder_attention.py -v
```
Expected: 13 passed.

**2. Prepare the dataset** (downloads Multi30k, tokenizes, builds vocab — requires internet):
```bash
PYTHONPATH=src python3 src/preprocess.py
```

**3. Train the LSTM model:**
```bash
PYTHONPATH=src python3 src/train.py
```
Trains for 10 epochs, saves the best checkpoint to `lstm_seq2seq_best.pt`, and prints a
training summary (parameter count, total time, hardware) at the end.

**4. Run the LLM baseline** (requires Ollama installed with `llama3.1:8b` pulled):
```bash
PYTHONPATH=src python3 src/llm_baseline.py
```
Runs 4 configurations (2 prompt variants × zero-shot/3-shot) on the test set, saves results
to `results/llm_results_v{variant}_{k}shot.json`.

**5. Generate LSTM translations on the test set:**
```bash
PYTHONPATH=src python3 src/inference.py
```
Loads the trained checkpoint, generates translations with no teacher forcing, saves to
`results/lstm_results.json`.

**6. Run evaluation** (BLEU/ROUGE scoring, comparing LSTM vs. each LLM configuration):
```bash
PYTHONPATH=src python3 src/eval.py
```

## Reproducibility

- Random seeds are locked (seed = 42) across Python `random`, NumPy, and PyTorch (CPU + CUDA/MPS)
  via `src/seed_utils.py::set_seed()`.
- Train/val/test splits come pre-fixed from the Multi30k dataset itself — no random re-splitting,
  so no leakage risk from re-running preprocessing.

## Model Summary

**LSTM (this team's implementation):**
- Bidirectional LSTM encoder + LSTMCell-based decoder, bridged via a learned projection
- Luong (multiplicative) attention
- 14,651,059 trainable parameters
- Trained 10 epochs, ~20 minutes on Apple M2 (MPS), final validation perplexity 19.2

**LLM baseline:**
- Llama 3.1, 8B parameters, run locally via Ollama (no API cost)
- 2 prompt variants (direct instruction; expert-persona), zero-shot and 3-shot
- ~3.5 hours total compute across all 4 configurations, ~1000 test samples each

## Known Integration Notes

- `data_loader.py`'s `CollatePad` returns only padded tensors, no lengths —
  `src/batch_utils.py::lengths_from_padded()` derives them from the padding index (`<pad>=0`).
- Vocabulary special tokens: `<pad>=0, <unk>=1, <sos>=2, <eos>=3`.
- The LSTM decoder generates with no teacher forcing during inference/evaluation — predictions
  are the model's own greedy output, not conditioned on the reference translation.
