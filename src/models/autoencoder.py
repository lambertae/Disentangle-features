import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.utils import norm_

class AutoEncoder(BaseModel):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        self.encoder.apply(init_weights)
        torch.nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        with torch.no_grad():
            num_coeffs = torch.count_nonzero(encoded, dim=1)
            num_coeffs = torch.sum(num_coeffs)
        norm_(self.decoder.weight)
        decoded = self.decoder(encoded)
        return encoded, decoded, num_coeffs
    
    def get_learned_features(self):
        return self.decoder.weight.detach().cpu().numpy()