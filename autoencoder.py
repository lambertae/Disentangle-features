# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import torch.nn.functional as F
def norm_(w):
    with torch.no_grad():
        norm = torch.linalg.norm(w, ord=2, dim=0)
        w /= norm

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lamb = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain = lamb / 0.1)
        self.encoder.apply(init_weights)
        torch.nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        with torch.no_grad():
            num_coeffs = torch.count_nonzero(encoded, dim=1)
            num_coeffs = num_coeffs.sum() / num_coeffs.shape[0]
        norm_(self.decoder.weight)
        decoded = self.decoder(encoded)
        return encoded, decoded, num_coeffs
    
def autoEncoder(dataset, guess_factor = 8, lr = 1e-2, epochs = 8, batch_size = 128, init_lamb = 0.1): 
    import torch as t
    import numpy as np
    from tqdm import trange
    solver_lamb = init_lamb
    while 1:   
        model = AutoEncoder(dataset.shape[1], dataset.shape[1] * guess_factor, solver_lamb).to(dataset.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        for epoch in trange(epochs):
            # Shuffle the dataset
            perm = torch.randperm(dataset.shape[0])
            acti = dataset[perm]
            for i in range(0, len(acti), batch_size):
                batch = acti[i:min(len(acti), i+batch_size)]
                optimizer.zero_grad()
                encoded, decoded, num_coeffs = model(batch)
                loss = t.norm(decoded - batch, 2) ** 2 + solver_lamb * t.norm(encoded, p=1)
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Num Coeffs: {num_coeffs.item():.4f}') 
        with torch.no_grad():
            encoded = model.encoder(dataset).cpu().clone().detach()
        srt_feats = abs(encoded).cpu().detach().numpy()
        sorted = np.flip(np.sort(srt_feats, axis = 1), axis = 1)
        pltarray = np.mean(sorted, axis = 0) #* np.arange(1, 101)
        maxA = pltarray[0]
        
        print(f"lamb: {solver_lamb}, maxA: {maxA}")
        if solver_lamb > 0.05 * maxA and solver_lamb < 0.2 * maxA:
            info = {"lamb": solver_lamb, "maxA": maxA, "model": model}
            return encoded, model.decoder.weight.cpu().clone().detach().T, info
        else:
            solver_lamb = 0.1 * maxA
# %%
import torch as t
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
hid = 5
true = 5
samples = 2000
emb = t.randn(true, hid) 
emb /= t.norm(emb, dim = 1).reshape(-1, 1)
emb = emb.to(device)
# print(t.norm(emb, dim = 1))
ground_truth = (t.rand((samples, true)) < 3 / 20).to(torch.float32).to(device)
acti = ground_truth @ emb
# %%
print(acti.shape)
feat, guess_emb, info = autoEncoder(acti, guess_factor = 8, lr = 3e-3, epochs = 1000, batch_size = 128, init_lamb = 0.1)
# %%
print(guess_emb.shape, emb.shape)
import numpy as np
def mmcs(D, F):
    dreg = D / np.linalg.norm(D, axis = 1)[..., np.newaxis]
    freg = F / np.linalg.norm(F, axis = 1)[..., np.newaxis]
    print(np.linalg.norm(dreg, axis = 1))
    print(dreg.shape, freg.shape)
    cs = np.einsum('ij,kj->ik', dreg, freg)
    maxv = np.amax(cs, axis=1)
    mmcs = np.mean(maxv)
    return mmcs
print(mmcs(emb.cpu().numpy(), guess_emb.cpu().numpy()))


# %%
gd_feat, gd_guess_emb, gd_info = autoEncoder(acti, guess_factor = 8, lr = 3e-3, epochs = 1000, batch_size = 128, init_lamb = 0.1)