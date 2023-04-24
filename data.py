# %%

import torch as t
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DataConfig:
    num_feats: int
    num_topics: int  # Should maybe be called topics per token
    topics_per_word: int
    num_words: int
    

class ToyDataset():

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.num_feats = config.num_feats
        self.num_topics = config.num_topics
        self.topics_per_word = config.topics_per_word
        self.num_words = config.num_words
        assert self.topics_per_word % self.num_topics == 0
        self.tokens_per_word = self.topics_per_word // self.num_topics

    def sample(self, size: t.Size = ()) -> Tuple[t.Tensor, t.Tensor]:
        # Create a set of features for each word
        topics = t.randint(self.num_feats, size=size+(self.num_topics,))
        words = t.cat(
            [repeat(topics, '... n -> ... w n', w=self.num_words),
            t.randint(self.num_feats, size+(self.num_words, self.topics_per_word - self.num_topics))],
            dim=-1
        )
        
        # Shuffle the order of the features in each word
        indices = t.argsort(t.rand(words.shape))
        shuffled_words = t.gather(words, -1, indices)

        # Make sure the features of the output are sorted
        topics, _ = t.sort(topics, dim=-1)

        # Group the features into words
        grouped_words = rearrange(shuffled_words, '... w (t n) -> ... (w t) n', t=self.tokens_per_word)
        tokens = self.num_feats * grouped_words[...,0] + grouped_words[...,1]  # TODO This only works if num_topics = 2
        outputs = self.num_feats * topics[...,0] + topics[...,1]

        return tokens, outputs

# %%

config = DataConfig(
    num_feats=10,
    num_topics=2,
    topics_per_word=4,
    num_words=5
)

dataset = ToyDataset(config)

# %%

dataset.sample((5,))

# %%
