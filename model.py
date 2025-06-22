import torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                      .unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_mult*dim),
            nn.GELU(),
            nn.Linear(hidden_mult*dim, dim)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.ff  = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=256, n_layers=6, n_heads=6, dim=384):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks  = nn.ModuleList([Block(dim, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(dim)
        self.head    = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence longer than block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
