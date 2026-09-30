import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader

class GPTEmbedding(nn.Module):
    
    def __init__(self, vocab_size, block_size, embedding_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

    def forward(self, x):

        B, T = x.shape

        token_emb = self.token_embedding(x)

        positions = torch.arange(T, device=x.device)
        position_emb = self.position_embedding(positions)

        return token_emb + position_emb

class Head(nn.Module):
    
    def __init__(self, head_size, block_size, embedding_dim, dropout = 0.0):
        super().__init__()

        # Query, Key & Value
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        # Causal Mask
        # register_buffer("name", tensor) is a method of class nn.Module
        # torch.tril - keep the lower triange, change upper triangle to 0

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # 1. Create Q, K, V metrics
        q = self.query(x) # Shape - (B, T, head_size)
        k = self.key(x)   # Shape - (B, T, head_size)
        v = self.value(x) # Shape - (B, T, head_size)

        # 2. Attention score & scaling
        weights = q @ k.transpose(-2,-1) # (B, T, head_size) x (B, head_size, T) = (B, T, T)
        weights = weights / (k.shape[-1] ** 0.5)

        # 3. Causal Mask
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        return output


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, block_size, embedding_dim, dropout = 0.0):
        super().__init__()

        assert embedding_dim % n_head == 0

        head_size = embedding_dim // n_head

        # nn.ModuleList = a list of neural-network layers that PyTorch knows about.
        self.head = nn.ModuleList([
            Head(embedding_dim=embedding_dim,
                 head_size=head_size,
                 block_size=block_size,
                 dropout=dropout)

                 for _ in range(n_head)
        ])

        # Mix information from all heads
        self.projection = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Concat all attention head
        output = torch.cat(
            [head(x) for head in self.heads],
            dim = -1
        )

        # Project concatenated heads
        output = self.projection(output)

        output = self.dropout(output)

        return output


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout=0.0):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed_forward(x)


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, n_head, block_size, dropout=0.0):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x):

        # Attention + Residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward + Residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x


class MiniGPT(nn.Module):

    def __init__(self, vocab_size, block_size, 
                 embedding_dim, n_head, n_layer, dropout=0.0):
        super().__init__()

        # Embeddings
        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim
        )

        # Transformer Block
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim=embedding_dim, n_head=n_head, block_size=block_size, dropout=dropout)
            for _ in range(n_layer)
        ])

        # Final Layer Normalization
        self.ln_final = nn.LayerNorm(embedding_dim)

        # Final projection (Language Modeling Head)
        self.projection_final = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, index):

        # index : Shape : (B,T)
        x = self.embedding(index) # Shape : (B, T, C)

        x = self.blocks(x) # Shape : (B, T, C)
        x = self.ln_final(x) # Shape : (B, T, C)

        logits = self.projection_final(x) # Shape : # (B, T, vocab_size)

        return logits