import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.utils import norm_

class DeepAutoEncoder(BaseModel):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2, bias=True),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*4, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim*4, input_dim, bias=False)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.enc1.apply(init_weights)
        self.enc2.apply(init_weights)
        self.enc3.apply(init_weights)
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        encoded = self.enc3(x)

        with torch.no_grad():
            num_coeffs = torch.count_nonzero(encoded, dim=1)
            num_coeffs = torch.sum(num_coeffs)

        norm_(self.decoder.weight)
        decoded = self.decoder(encoded)
        return encoded, decoded, num_coeffs

    def get_learned_features(self):
        return self.decoder.weight.detach().cpu().numpy()