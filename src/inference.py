import torch

from attention import make_src_mask
from batch_utils import lengths_from_padded


@torch.no_grad()
def greedy_decode(
    model,
    src,
    src_lengths,
    tgt_vocab,
    max_len=50,
    device="cpu"
):
    model.eval()

    src = src.to(device)
    src_lengths = src_lengths.to(device)

    encoder_outputs, hidden, cell = model.encoder(
        src,
        src_lengths
    )

    mask = make_src_mask(
        src_lengths,
        max_len=src.size(1),
        device=device
    )

    sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
    eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]

    batch_size = src.size(0)

    input_token = torch.full(
        (batch_size,),
        sos_idx,
        dtype=torch.long,
        device=device
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device
    )

    output_tokens = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        logits, hidden, cell, _ = model.decoder.forward_step(
            input_token,
            hidden,
            cell,
            encoder_outputs,
            mask
        )

        next_token = logits.argmax(dim=1)

        for i in range(batch_size):
            if not finished[i]:
                token_id = next_token[i].item()

                if token_id == eos_idx:
                    finished[i] = True
                else:
                    output_tokens[i].append(token_id)

        input_token = next_token

        if finished.all():
            break

    return output_tokens


@torch.no_grad()
def beam_search_decode(
    model,
    src,
    src_lengths,
    tgt_vocab,
    beam_width=5,
    max_len=50,
    length_penalty=0.7,
    device="cpu"
):
    # Beam search is done one sentence at a time.
    model.eval()

    src = src.to(device)
    src_lengths = src_lengths.to(device)

    sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
    eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]

    results = []

    for i in range(src.size(0)):
        src_i = src[i:i + 1]
        length_i = src_lengths[i:i + 1]

        encoder_outputs, hidden, cell = model.encoder(
            src_i,
            length_i
        )

        mask = make_src_mask(
            length_i,
            max_len=src_i.size(1),
            device=device
        )

        beams = [
            ([sos_idx], hidden, cell, 0.0, False)
        ]

        for _ in range(max_len):
            if all(beam[4] for beam in beams):
                break

            candidates = []

            for tokens, h, c, score, finished in beams:
                if finished:
                    candidates.append(
                        (tokens, h, c, score, True)
                    )
                    continue

                input_token = torch.tensor(
                    [tokens[-1]],
                    dtype=torch.long,
                    device=device
                )

                logits, new_hidden, new_cell, _ = (
                    model.decoder.forward_step(
                        input_token,
                        h,
                        c,
                        encoder_outputs,
                        mask
                    )
                )

                log_probs = torch.log_softmax(
                    logits.squeeze(0),
                    dim=-1
                )

                top_scores, top_tokens = log_probs.topk(
                    beam_width
                )

                for new_score, token_id in zip(
                    top_scores.tolist(),
                    top_tokens.tolist()
                ):
                    candidates.append(
                        (
                            tokens + [token_id],
                            new_hidden,
                            new_cell,
                            score + new_score,
                            token_id == eos_idx
                        )
                    )

            def normalized_score(beam):
                tokens, _, _, score, _ = beam
                length = max(len(tokens) - 1, 1)

                return score / (
                    length ** length_penalty
                )

            candidates.sort(
                key=normalized_score,
                reverse=True
            )

            beams = candidates[:beam_width]

        best_tokens = max(
            beams,
            key=normalized_score
        )[0]

        # Remove <sos>.
        best_tokens = best_tokens[1:]

        # Remove <eos>.
        if best_tokens and best_tokens[-1] == eos_idx:
            best_tokens = best_tokens[:-1]

        results.append(best_tokens)

    return results


def strip_special(tokens, vocab):
    special_tokens = {
        vocab.pad_token,
        vocab.sos_token,
        vocab.eos_token
    }

    return [
        token
        for token in tokens
        if token not in special_tokens
    ]


@torch.no_grad()
def run_inference(
    model,
    test_loader,
    src_vocab,
    tgt_vocab,
    decode_fn=greedy_decode,
    max_len=50,
    device="cpu",
    **decode_kwargs
):
    sources = []
    references = []
    predictions = []

    src_pad_idx = src_vocab.stoi[
        src_vocab.pad_token
    ]

    for src_batch, tgt_batch in test_loader:
        src_lengths = lengths_from_padded(
            src_batch,
            pad_idx=src_pad_idx
        )

        predicted_ids = decode_fn(
            model,
            src_batch,
            src_lengths,
            tgt_vocab,
            max_len=max_len,
            device=device,
            **decode_kwargs
        )

        for i in range(src_batch.size(0)):
            source_tokens = strip_special(
                src_vocab.decode(src_batch[i]),
                src_vocab
            )

            reference_tokens = strip_special(
                tgt_vocab.decode(tgt_batch[i]),
                tgt_vocab
            )

            prediction_tokens = [
                tgt_vocab.itos.get(
                    token_id,
                    tgt_vocab.unk_token
                )
                for token_id in predicted_ids[i]
            ]

            sources.append(
                " ".join(source_tokens)
            )

            references.append(
                " ".join(reference_tokens)
            )

            predictions.append(
                " ".join(prediction_tokens)
            )

    return sources, references, predictions


def load_model(
    checkpoint_path,
    src_vocab,
    tgt_vocab,
    device="cpu"
):
    from encoder import LSTMEncoder
    from decoder import LSTMDecoder
    from attention import LuongAttention
    from seq2seq import Seq2Seq

    enc_emb_dim = 256
    dec_emb_dim = 256
    hidden_dim = 512
    num_layers = 1
    dropout = 0.5

    src_pad_idx = src_vocab.stoi[
        src_vocab.pad_token
    ]

    tgt_pad_idx = tgt_vocab.stoi[
        tgt_vocab.pad_token
    ]

    encoder = LSTMEncoder(
        len(src_vocab),
        enc_emb_dim,
        hidden_dim,
        num_layers,
        pad_idx=src_pad_idx,
        dropout=dropout,
        bidirectional=True
    )

    attention = LuongAttention(
        enc_dim=hidden_dim * 2,
        dec_dim=hidden_dim
    )

    decoder = LSTMDecoder(
        len(tgt_vocab),
        dec_emb_dim,
        hidden_dim * 2,
        hidden_dim,
        attention,
        pad_idx=tgt_pad_idx,
        dropout=dropout
    )

    model = Seq2Seq(
        encoder,
        decoder,
        torch.device(device)
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.eval()

    return model
