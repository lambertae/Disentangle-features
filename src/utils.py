import torch
import numpy as np

def norm_(w):
    with torch.no_grad():
        norm = torch.linalg.norm(w, ord=2, dim=0)
        w /= norm

def l1_reg(x):
    # x is num_examples x num_features
    n_x = torch.linalg.norm(x, ord=1, dim=1) / x.shape[1]
    return torch.mean(n_x)

def mmcs_fn(D, F):
    # F is our original input and is already has unit length columns
    F_norm = F / np.linalg.norm(F, ord=2, axis=0)
    D_norm = D / np.linalg.norm(D, ord=2, axis=0)
    cs = np.einsum('ij,jk->ik', D_norm.T, F_norm)
    max = np.amax(cs, axis=0)
    mmcs = np.average(max)
    return mmcs