import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import numpy as np
import math
import config

# =============================
# CONFIG
# =============================

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
            vocab_size=config.VOCAB_SIZE,
            character_coverage=0.9995,
            num_threads=4
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load('./data/tokenizer/tokenizer.model')
    
    with open(corpus_path, "r") as file:
        data = file.read()

    tokens = sp.encode(data, out_type=int)
    print(len(tokens))
    np.save(tokens_path, np.array(tokens, dtype=np.int32))
    

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
    
    train_dataset = FinanceDataset(tokens=train_tokens, block_size=config.BLOCK_SIZE)
    val_dataset = FinanceDataset(tokens=val_tokens, block_size=config.BLOCK_SIZE)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'mps' else False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'mps' else False
    )

    return train_loader, val_loader

# =============================
# ATTENTION BLOCK
# =============================
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        
        self.W_Q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_K = nn.Linear(n_embd, n_embd, bias=False)
        self.W_V = nn.Linear(n_embd, n_embd, bias=False)
        self.W_O = nn.Linear(n_embd, n_embd, bias=False)
        
        self.dropout = dropout
    
    def forward(self, x):
        B, T, C = x.shape
        
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        
        p_drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(Q, K, V, dropout_p=p_drop, is_causal=True)
        
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.W_O(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1) # Evite le sur-apprentissage
        )

    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# =============================
# MODEL
# =============================
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, max_seq_len, n_head, dropout, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_encoding = PositionalEncoding(n_embd, max_seq_len)    
        
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        x = self.position_encoding(tok_emb)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            max_len = self.position_encoding.pe.size(1)
            idx_cond = idx if idx.size(1) <= max_len else idx[:, -max_len:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # devient (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
