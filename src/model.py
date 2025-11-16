import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import numpy as np

# =============================
# CONFIG
# =============================

VOCAB_SIZE = 2000
BLOCK_SIZE = 8
BATCH_SIZE = 8
NUM_WORKERS = 0
MAX_ITERS = 5000
EVAL_INTERVAL = 1000
LEARNING_RATE = 3e-4


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device defined to : {device}")
torch.manual_seed(2077)

# =============================
# TOKENIZATION
# =============================

def create_encode_corpus():
    corpus_path = Path("./data/corpus/corpus.txt")
    tokens_path = Path("./data/corpus/corpus.npy")
    model_path = Path("./data/tokenizer/tokenizer.model")
    
    if tokens_path.exists():
        print("Les tokens ont déjà été générés et sont présent dans `./data/corpus/`")
        return
    

    if not model_path.exists():
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix="./data/tokenizer/tokenizer",
            vocab_size=VOCAB_SIZE,
            character_coverage=0.9995,
            num_threads=4
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load('./data/tokenizer/tokenizer.model')
    
    with open(corpus_path, "r") as file:
        data = file.read()

    tokens = sp.encode(data, out_type=int)
    np.save(tokens_path, np.array(tokens, dtype=np.int32))
    
    
create_encode_corpus()

# =============================
# DATASET FOR PYTORCH MODEL
# =============================

class FinanceDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.block_size = block_size
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.block_size]
        y = self.tokens[idx+1:idx+self.block_size+1]
        return x, y

def load_data():
    tokens = np.load('./data/corpus/corpus.npy')
    n = len(tokens)
    train_tokens = tokens[:int(n * 0.9)]
    val_tokens = tokens[int(n * 0.9):]
    
    train_dataset = FinanceDataset(tokens=train_tokens, block_size=BLOCK_SIZE)
    val_dataset = FinanceDataset(tokens=val_tokens, block_size=BLOCK_SIZE)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'mps' else False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'mps' else False
    )

    return train_loader, val_loader

# =============================
# MODEL
# =============================
class BigramLanguagueModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx = B, T
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =============================
# TRAINING LOOP
# =============================
@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters):
    model.eval()
    losses = []
    
    for i, (xb, yb) in enumerate(data_loader):
        if i >= eval_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)

def train():
    train_loader, val_loader = load_data()
    
    model = BigramLanguagueModel(VOCAB_SIZE)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    sp = spm.SentencePieceProcessor()
    sp.load('tokenizer.model')
    
    print(f"training launched on {device}")
    
    step = 0
    running_loss = 0.0
    
    for _ in range(100):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            _, loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            step += 1

            if step % EVAL_INTERVAL == 0:
                train_loss = running_loss / EVAL_INTERVAL
                val_loss = estimate_loss(model, val_loader, EVAL_INTERVAL)
                
                print(f"Step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                running_loss = 0.0
                
                                
                # Test génération
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=50)
                text = sp.decode(generated[0].tolist())
                print(f"Génération: {text}\n")
            
            if step >= MAX_ITERS:
                break
        
        if step >= MAX_ITERS:
            break
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, 'bigram_model.pt')
    
    print("Training terminé!")
    
def generate_text(prompt="", max_tokens=100):
    sp = spm.SentencePieceProcessor()
    sp.load('tokenizer.model')
    
    model = BigramLanguagueModel(VOCAB_SIZE)
    checkpoint = torch.load('bigram_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if prompt:
        context = torch.tensor(sp.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated = model.generate(context, max_new_tokens=max_tokens)
    return sp.decode(generated[0].tolist())

create_encode_corpus()
train()
print(generate_text("The company", max_tokens=50))