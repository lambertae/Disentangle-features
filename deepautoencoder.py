import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

def norm_(w):
    with torch.no_grad():
        norm = torch.linalg.norm(w, ord=2, dim=0)
        w /= norm


class AutoEncoder(nn.Module):
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


class DeepAutoEncoder(nn.Module):
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
                torch.nn.init.xavier_uniform_(m.weight)
        #self.enc1.apply(init_weights)
        #self.enc2.apply(init_weights)
        #self.enc3.apply(init_weights)
        torch.nn.init.orthogonal_(self.decoder.weight)

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
    
def l1_reg(x):
    # x is num_examples x num_features
    n_x = torch.linalg.norm(x, ord=1, dim=1) / x.shape[1]
    return torch.mean(n_x)


def run_disentangle(num_times, dataset, num_features=512, feature_dim=256, reg_param=0.1, batch_size=256, epochs=1000, lr=1e-3):
    print(f"Executing with num_features: {num_features} and feature_dim: {feature_dim} and reg_param: {reg_param}")

    recon_loss_fn = nn.MSELoss()
    loader = DataLoader(dataset, batch_size = batch_size)

    # Run multiple times and get best performance
    for _ in range(num_times):

        model = DeepAutoEncoder(feature_dim, num_features)
        model = model.to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model = model.double()
        model.train()

        t = trange(epochs, desc='Epoch loss: ', leave=True)
        #patience = 500
        best_epoch_loss = 100000000
        for _ in t:
            running_loss, num_examples = 0, 0
            best_coeffs = 0
            avg_coeffs = 0.0
            num_count = 0
            for _, features in enumerate(loader):
                optimizer.zero_grad()
                features = features.to('cuda')
                encoded, decoded, num_coeffs = model(features)
                loss =  recon_loss_fn(decoded, features) +  reg_param * l1_reg(encoded)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_examples += len(features)
                avg_coeffs += num_coeffs
                num_count += 1
            
            epoch_loss = running_loss / num_count
            avg_coeffs /= num_examples
            t.set_description(f"Epoch loss: {epoch_loss:.10f}, Avg coeffs: {avg_coeffs:.3f}")
            t.refresh()
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_coeffs = avg_coeffs
        model = model.to('cpu')
    return model, best_coeffs