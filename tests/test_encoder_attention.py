"""
test_encoder_attention.py
Unit tests for LSTMEncoder and attention modules
"""

import torch
import pytest

from encoder import Embedding, LSTMEncoder
from attention import BahdanauAttention, LuongAttention, make_src_mask
from seed_utils import set_seed

set_seed(42)

VOCAB_SIZE = 50
EMBED_DIM = 16
HIDDEN_DIM = 32
BATCH_SIZE = 4
MAX_SRC_LEN = 10
PAD_IDX = 0

#Random-length token batches, padded with PAD_IDX, lengths left unsorted on purpose
def make_mock_batch():
    lengths = torch.randint(3, MAX_SRC_LEN + 1, (BATCH_SIZE,))
    src = torch.zeros(BATCH_SIZE, MAX_SRC_LEN, dtype=torch.long)
    for i, length in enumerate(lengths):
        src[i, :length] = torch.randint(1, VOCAB_SIZE, (length.item(),))
    return src, lengths


class TestEmbedding:
    def test_output_shape(self):
        emb = Embedding(VOCAB_SIZE, EMBED_DIM, pad_idx=PAD_IDX)
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SRC_LEN))
        out = emb(x)
        assert out.shape == (BATCH_SIZE, MAX_SRC_LEN, EMBED_DIM)

    def test_pad_idx_vector_is_zero_at_init(self):
        emb = Embedding(VOCAB_SIZE, EMBED_DIM, pad_idx=PAD_IDX)
        pad_vec = emb.embedding.weight[PAD_IDX]
        assert torch.allclose(pad_vec, torch.zeros_like(pad_vec))


class TestLSTMEncoder:
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_output_shapes(self, bidirectional):
        encoder = LSTMEncoder(
            VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, pad_idx=PAD_IDX, bidirectional=bidirectional
        )
        src, lengths = make_mock_batch()
        outputs, hidden, cell = encoder(src, lengths)

        num_directions = 2 if bidirectional else 1
        assert outputs.shape == (BATCH_SIZE, MAX_SRC_LEN, HIDDEN_DIM * num_directions)
        assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)
        assert cell.shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_no_nans_in_forward_pass(self):
        encoder = LSTMEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, pad_idx=PAD_IDX)
        src, lengths = make_mock_batch()
        outputs, hidden, cell = encoder(src, lengths)
        assert not torch.isnan(outputs).any()
        assert not torch.isnan(hidden).any()
        assert not torch.isnan(cell).any()

    def test_handles_unsorted_lengths(self):
        # enforce_sorted=False must let pack_padded_sequence handle batches
        # where lengths are NOT sorted descending 
        encoder = LSTMEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, pad_idx=PAD_IDX)
        src = torch.randint(1, VOCAB_SIZE, (3, MAX_SRC_LEN))
        lengths = torch.tensor([5, 10, 3])  # deliberately unsorted
        outputs, hidden, cell = encoder(src, lengths)
        assert outputs.shape[0] == 3

    def test_multilayer_encoder_shapes(self):
        encoder = LSTMEncoder(
            VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, num_layers=2, pad_idx=PAD_IDX, dropout=0.1
        )
        src, lengths = make_mock_batch()
        outputs, hidden, cell = encoder(src, lengths)
        assert outputs.shape == (BATCH_SIZE, MAX_SRC_LEN, HIDDEN_DIM * 2)
        assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)


class TestAttention:
    def _mock_encoder_outputs(self, enc_dim):
        outputs = torch.randn(BATCH_SIZE, MAX_SRC_LEN, enc_dim)
        lengths = torch.randint(3, MAX_SRC_LEN + 1, (BATCH_SIZE,))
        mask = make_src_mask(lengths, MAX_SRC_LEN)
        return outputs, mask, lengths

    def test_mask_shape_and_values(self):
        lengths = torch.tensor([3, 10, 5, 1])
        mask = make_src_mask(lengths, MAX_SRC_LEN)
        assert mask.shape == (4, MAX_SRC_LEN)
        assert mask[0].sum().item() == 3
        assert mask[3].sum().item() == 1

    def test_bahdanau_shapes_and_weight_sums(self):
        enc_dim = HIDDEN_DIM * 2  # bidirectional encoder output size
        dec_dim = HIDDEN_DIM
        attn = BahdanauAttention(enc_dim, dec_dim, attn_dim=24)
        encoder_outputs, mask, _ = self._mock_encoder_outputs(enc_dim)
        decoder_hidden = torch.randn(BATCH_SIZE, dec_dim)

        context, weights = attn(decoder_hidden, encoder_outputs, mask)
        assert context.shape == (BATCH_SIZE, enc_dim)
        assert weights.shape == (BATCH_SIZE, MAX_SRC_LEN)
        assert torch.allclose(weights.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)

    def test_bahdanau_zero_weight_on_padding(self):
        enc_dim = HIDDEN_DIM * 2
        attn = BahdanauAttention(enc_dim, HIDDEN_DIM, attn_dim=24)
        encoder_outputs, mask, lengths = self._mock_encoder_outputs(enc_dim)
        decoder_hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        _, weights = attn(decoder_hidden, encoder_outputs, mask)
        for i in range(BATCH_SIZE):
            pad_weights = weights[i, lengths[i]:]
            assert torch.allclose(pad_weights, torch.zeros_like(pad_weights), atol=1e-6)

    def test_luong_shapes_and_weight_sums(self):
        enc_dim = HIDDEN_DIM * 2
        attn = LuongAttention(enc_dim, HIDDEN_DIM)
        encoder_outputs, mask, _ = self._mock_encoder_outputs(enc_dim)
        decoder_hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        context, weights = attn(decoder_hidden, encoder_outputs, mask)
        assert context.shape == (BATCH_SIZE, enc_dim)
        assert weights.shape == (BATCH_SIZE, MAX_SRC_LEN)
        assert torch.allclose(weights.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)

# Confirms a full forward + backward pass through encoder
# And attention produces finite gradients     
class TestGradientFlow:
    def test_backward_pass_and_gradient_clipping(self):
        encoder = LSTMEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, pad_idx=PAD_IDX, bidirectional=True)
        attn = BahdanauAttention(enc_dim=HIDDEN_DIM * 2, dec_dim=HIDDEN_DIM, attn_dim=24)
        params = list(encoder.parameters()) + list(attn.parameters())

        src, lengths = make_mock_batch()
        encoder_outputs, hidden, cell = encoder(src, lengths)
        mask = make_src_mask(lengths, MAX_SRC_LEN)
        context, attn_weights = attn(hidden, encoder_outputs, mask)

        # include cell (not just hidden) so bridge_c's parameters are exercised too
        loss = context.sum() + attn_weights.sum() + cell.sum()
        loss.backward()

        assert not torch.isnan(context).any()
        assert not torch.isnan(attn_weights).any()
        assert all(p.grad is not None for p in params), "Some parameters received no gradient"
        assert not any(torch.isnan(p.grad).any() for p in params), "NaN gradients detected"

        pre_clip_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        assert pre_clip_norm > 1.0, "Test setup issue: gradient norm too small to demonstrate clipping"

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        post_clip_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
        assert post_clip_norm <= 1.0 + 1e-4, "Gradient clipping did not bound the norm as expected"

# runs data_loader.py / dataset.py / vocab.py through LSTMencoder + attention with no mocking
class TestRealDataIntegration:
    def _toy_data(self):
        train_src = [
            ["how", "are", "you"],
            ["i", "am", "doing", "well"],
            ["this", "is", "a", "sample", "sentence"],
            ["learning", "nlp", "is", "fun"],
        ]
        train_tgt = [
            ["comment", "allez", "vous"],
            ["je", "vais", "bien"],
            ["c'est", "un", "exemple", "de", "phrase"],
            ["l'apprentissage", "du", "taln", "est", "amusant"],
        ]
        val_src = [["how", "are", "you"]]
        val_tgt = [["comment", "allez", "vous"]]
        test_src = [["learning", "nlp"]]
        test_tgt = [["l'apprentissage", "du", "taln"]]
        return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt

    def test_real_data_loader_to_encoder_attention(self):
        from data_loader import get_data_loaders  # Member 1
        from batch_utils import lengths_from_padded  # glue helper

        train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = self._toy_data()
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data_loaders(
            train_src, train_tgt, val_src, val_tgt, test_src, test_tgt,
            batch_size=2, min_freq=1,
        )

        src_batch, tgt_batch = next(iter(train_loader))
        assert src_batch.dtype == torch.long

        src_lengths = lengths_from_padded(src_batch, pad_idx=src_vocab.stoi["<pad>"])
        assert (src_lengths > 0).all()
        assert (src_lengths <= src_batch.size(1)).all()

        embed_dim, hidden_dim = 32, 64
        encoder = LSTMEncoder(
            vocab_size=len(src_vocab), embed_dim=embed_dim, hidden_dim=hidden_dim,
            pad_idx=src_vocab.stoi["<pad>"], bidirectional=True,
        )
        attn = BahdanauAttention(enc_dim=hidden_dim * 2, dec_dim=hidden_dim, attn_dim=48)

        encoder_outputs, hidden, cell = encoder(src_batch, src_lengths)
        mask = make_src_mask(src_lengths, max_len=src_batch.size(1))
        context, attn_weights = attn(hidden, encoder_outputs, mask)

        assert not torch.isnan(encoder_outputs).any()
        assert not torch.isnan(context).any()
        assert torch.allclose(attn_weights.sum(dim=1), torch.ones(src_batch.size(0)), atol=1e-5)

        # confirm backward pass also works on real data end to end
        params = list(encoder.parameters()) + list(attn.parameters())
        loss = context.sum() + attn_weights.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)