import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import prepare_multi30k_loaders
from encoder import LSTMEncoder
from decoder import LSTMDecoder
from attention import LuongAttention
from seq2seq import Seq2Seq
from batch_utils import lengths_from_padded
from seed_utils import set_seed

def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):

    model.train()
    epoch_loss = 0
    
    for _, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(model.device), tgt.to(model.device)
        src_lengths = lengths_from_padded(src, pad_idx=0)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_lengths, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Flatten outputs for criterion. Ignore the first token (<sos>)
        # output shape: (batch_size, tgt_len, vocab_size) -> (batch_size * (tgt_len - 1), vocab_size)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

@torch.no_grad()
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    for _, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(model.device), tgt.to(model.device)
        src_lengths = lengths_from_padded(src, pad_idx=0)
        
        # Turn off teacher forcing during evaluation
        output = model(src, src_lengths, tgt, teacher_forcing_ratio=0) 
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 10
    CLIP = 1.0

    print("Preparing Dataloaders...")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, _, _ = prepare_multi30k_loaders(batch_size=BATCH_SIZE)
    
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    PAD_IDX = tgt_vocab.stoi["<pad>"]

    # Assemble Model Components
    encoder = LSTMEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, pad_idx=PAD_IDX, dropout=ENC_DROPOUT, bidirectional=True)
    attention = LuongAttention(enc_dim=HID_DIM * 2, dec_dim=HID_DIM)
    decoder = LSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM * 2, HID_DIM, attention, pad_idx=PAD_IDX, dropout=DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    print(f'Model initialized. The model has {count_parameters(model):,} trainable parameters.')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Testing Teacher Forcing Decay (starts at 0.5, decays linearly)
    best_valid_loss = float('inf')

    print("Starting Training Loop...")

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        # Dynamic teacher forcing decay (Example Strategy)
        teacher_forcing_ratio = max(0.2, 0.5 - (0.05 * epoch)) 
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, teacher_forcing_ratio)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'lstm_seq2seq_best.pt')
            
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s | TF Ratio: {teacher_forcing_ratio:.2f}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == "__main__":
    main()