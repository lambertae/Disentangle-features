import torch
import torch.nn as nn

from src.models.base_model import BaseModel

class DirectSolver(BaseModel):

    def __init__(self, num_hidden, num_examples, guess_factor, lamb, negative_penalty, activations):
        super().__init__()
        self.lamb = lamb
        self.negative_penalty = negative_penalty
        self.activations = activations
        num_features = guess_factor * num_hidden

        self.feature_emb = nn.Parameter(torch.empty((num_features, num_hidden), requires_grad=True))
        nn.init.xavier_normal_(self.feature_emb)
        self.features_of_samples = nn.Parameter(torch.zeros((num_examples, num_features), requires_grad=True))
        nn.init.xavier_normal_(self.features_of_samples, gain = lamb / 0.1)

    def forward(self, x):
        # For this method, you should pass in the entire set of activtions @ once to run SGD on it
        return self.lamb * torch.norm(self.features_of_samples * (1 + (self.features_of_samples < 0) * self.negative_penalty) * torch.norm(self.feature_emb, dim = 1, p = 2), p=1) + torch.norm(self.features_of_samples @ self.feature_emb - x, p=2) ** 2
    
    def get_learned_features(self):
        return self.feature_emb.T.cpu().detach().numpy()