# %%

import torch as t
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from dataclasses import dataclass

# %%

@dataclass
class ModelConfig:
    num_layers: int
    num_heads: int
    embedding_dim: int
    num_tokens: int
    seq_len: int


class AttnLayer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim // self.num_heads

        self.query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)

    def attn_pattern(self, x: t.Tensor) -> t.Tensor:
        '''
        input: shape (batch, seq, embedding_dim)
        '''
        queries = rearrange(self.query(x), 'b s (n h) -> b s n h', n=self.num_heads)
        keys = rearrange(self.key(x), 'b s (n h) -> b s n h', n=self.num_heads)
        pre_softmax = t.einsum('b q n h, b k n h -> b n q k', queries, keys)
        return t.softmax(pre_softmax / t.sqrt(t.tensor(self.head_dim)), dim = -1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        input: shape (batch, seq, embedding_dim)
        '''
        attn = self.attn_pattern(x)
        values = rearrange(self.value(x), 'b s (n h) -> b s n h', n=self.num_heads)
        outs =  rearrange(t.einsum('b n q k, b k n h -> b q n h', attn, values), 'b q n h -> b q (n h)')
        return self.output(outs)


class MLPLayer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embedding_dim = config.embedding_dim
        self.layer1 = nn.Linear(self.embedding_dim, 4 * self.embedding_dim)
        self.nonlin = nn.ReLU()
        self.layer2 = nn.Linear(4 * self.embedding_dim, self.embedding_dim)
        

    def forward(self, x: t.Tensor):
        return self.layer2(self.nonlin(self.layer1(x)))
    

class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attn_layer = AttnLayer(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLPLayer(config)

    def forward(self, x: t.Tensor) -> t.Tensor:
        output = x + self.attn_layer(self.ln1(x))
        return output + self.mlp(self.ln2(x))


class TransformerModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.num_tokens = config.num_tokens
        self.embedding_dim = config.embedding_dim
        self.seq_len = config.seq_len
        self.num_layers = config.num_layers

        self.token_embedding = nn.Embedding(self.num_tokens, self.embedding_dim)
        self.pos_embedding = nn.Embedding(self.seq_len, self.embedding_dim)
        self.layers = nn.Sequential(
            *[TransformerBlock(config) for _ in range(self.num_layers)]
        )
        self.final_ln = nn.LayerNorm(self.embedding_dim)
        self.unembedding = nn.Linear(self.embedding_dim, self.num_tokens)

    def forward(self, x: t.Tensor):
        '''
        input: shape (batch, seq)
        output: logits at the final (padding) token
        '''

        res = self.token_embedding(x) + self.pos_embedding(t.arange(self.seq_len))
        logits = self.unembedding(self.final_ln(self.layers(res)))
        return logits[...,-1,:]

# %%

config = ModelConfig(
    num_layers=3,
    num_heads=2,
    embedding_dim=20,
    num_tokens=10000,
    seq_len=11
)

test_data = rearrange(t.arange(11 * 5), '(b s) -> b s', b = 5)
test_model = TransformerModel(config)

test_model(test_data)

# %%
