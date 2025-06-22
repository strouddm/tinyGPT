"""CPU‑only training script for PocketGPT."""
import os, pickle, math, argparse, time, pathlib

import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
import tiktoken

from model import TinyGPT

BLOCK_SIZE = 256
BATCH_SIZE = 64  # Increased from 32 for 32GB RAM
VAL_SPLIT  = 0.02
EPOCHS     = 5
LR         = 3e-3  # Slightly increased for larger batch size

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', default='corpus/corpus_clean.txt')
args = parser.parse_args()

class TokenDataset(Dataset):
    def __init__(self, tokens, split='train'):
        split_idx = int(len(tokens) * (1 - VAL_SPLIT))
        self.tokens = np.array(tokens[:split_idx] if split=='train' else tokens[split_idx:], dtype=np.int64)
        print(f"   {split.capitalize()} dataset: {len(self.tokens):,} tokens")
    def __len__(self): return len(self.tokens) - BLOCK_SIZE
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+BLOCK_SIZE], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+BLOCK_SIZE+1], dtype=torch.long)
        return x, y

def main():
    print("🚀 Starting TinyGPT Training Setup...")

    # --- Load & tokenize corpus --------------------------------------------------
    print("📚 Loading and tokenizing corpus...")
    tok_cache = 'corpus/tokens.pkl'
    tokenizer = tiktoken.get_encoding('gpt2')
    if os.path.exists(tok_cache):
        print(f"✅ Loading cached tokens from {tok_cache}")
        tokens = pickle.load(open(tok_cache, 'rb'))
        print(f"   Loaded {len(tokens):,} tokens")
    else:
        print(f"🔄 Tokenizing corpus from {args.corpus}...")
        text = pathlib.Path(args.corpus).read_text()
        print(f"   Corpus size: {len(text):,} characters")
        tokens = tokenizer.encode(text)
        print(f"   Tokenized to {len(tokens):,} tokens")
        pathlib.Path(tok_cache).parent.mkdir(exist_ok=True)
        pickle.dump(tokens, open(tok_cache, 'wb'))
        print(f"✅ Saved tokens to {tok_cache}")

    # --- Dataset -----------------------------------------------------------------
    print("📊 Creating datasets...")
    train_ds = TokenDataset(tokens, 'train')
    val_ds   = TokenDataset(tokens, 'val')
    # Multi-threaded for 8-core VM
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=4, pin_memory=False)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   DataLoader workers: 4 (optimized for 8-core VM)")

    # --- Model & optim -----------------------------------------------------------
    print("🧠 Initializing model...")
    model = TinyGPT(tokenizer.n_vocab, BLOCK_SIZE)
    device = torch.device('cpu')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocabulary size: {tokenizer.n_vocab:,}")
    print(f"   Block size: {BLOCK_SIZE}")
    print(f"   Learning rate: {LR}")

    # Memory usage estimation
    estimated_memory = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # MB
    batch_memory = BATCH_SIZE * BLOCK_SIZE * 4 / 1024 / 1024  # MB for batch data
    total_estimated = estimated_memory + batch_memory
    print(f"   Estimated model memory: {estimated_memory:.1f} MB")
    print(f"   Estimated batch memory: {batch_memory:.1f} MB")
    print(f"   Total estimated memory: {total_estimated:.1f} MB")
    print(f"   Available RAM: 32GB - plenty of headroom!")

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, steps = 0, 0
        mode = "Training" if train else "Validation"
        print(f"   🔄 {mode} epoch...")
        
        # Time tracking
        epoch_start = time.time()
        last_update = epoch_start
        
        for batch_idx, (x, y) in enumerate(loader):
            batch_start = time.time()
            
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item()
            steps += 1
            
            # Progress update every 10 batches with timing info
            if batch_idx % 10 == 0 and batch_idx > 0:
                avg_loss = total_loss / steps
                progress = (batch_idx / len(loader)) * 100
                elapsed = time.time() - epoch_start
                batch_time = time.time() - batch_start
                eta = (len(loader) - batch_idx) * batch_time
                print(f"      Batch {batch_idx}/{len(loader)} ({progress:.1f}%) - "
                      f"Loss: {avg_loss:.4f} - "
                      f"Time: {elapsed/60:.1f}m - "
                      f"ETA: {eta/60:.1f}m")
            
            # Emergency stop if taking too long
            if batch_idx > 0 and batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                if elapsed > 3600:  # More than 1 hour
                    print(f"⚠️  Emergency stop: Training taking too long ({elapsed/60:.1f} minutes for {batch_idx} batches)")
                    print(f"   This suggests a performance issue. Consider reducing model size or batch size.")
                    return total_loss / steps
        
        avg_loss = total_loss / steps
        epoch_time = time.time() - epoch_start
        print(f"   ✅ {mode} complete - Avg Loss: {avg_loss:.4f} - Time: {epoch_time/60:.1f} minutes")
        return avg_loss

    print("🎯 Starting training loop...")
    print("⏱️  Expected time: ~10-20 minutes per epoch (optimized for 8-core VM)")
    print("🚨 If training takes >1 hour per epoch, there's a performance issue!")
    
    for epoch in range(EPOCHS):
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        print("=" * 50)
        
        t0 = time.time()
        tr_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        tr_ppl = math.exp(tr_loss)
        val_ppl = math.exp(val_loss)
        dt = time.time() - t0
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {tr_loss:.4f} | Train PPL: {tr_ppl:.2f}")
        print(f"   Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        print(f"   Time: {dt/60:.1f} minutes")
        
        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

    print("\n🎉 Training complete!")
    print("📁 Checkpoints saved:")
    for i in range(1, EPOCHS + 1):
        print(f"   - checkpoint_epoch{i}.pt")
    print(f"\n💡 Performance optimizations for 8-core, 32GB RAM VM:")
    print(f"   - Batch size: 64 (2x larger than before)")
    print(f"   - Multi-threaded mode (4 workers for 8-core CPU)")
    print(f"   - Learning rate: 3e-3 (optimized for larger batch size)")
    print(f"   - Emergency stop if training takes >1 hour per epoch")
    print(f"   - Detailed timing information")

if __name__ == '__main__':
    main()
