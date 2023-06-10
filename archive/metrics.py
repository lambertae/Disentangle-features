import numpy as np
import torch
from sklearn.metrics import r2_score 
from scipy.stats import pearsonr
def r2(x, y):
    return r2_score(x, y)
def p_value(x, y):
    return pearsonr(x, y)[1]

def average_r2(ground_truth, acti, allow_prefix = False):
    baseline_r2 = []
    for i in range(ground_truth.shape[1]):
        id_list = []
        for j in range(acti.shape[1]):
            # print(r2(ground_truth[:, i].cpu().numpy(), feat[:, j].cpu().numpy()))
            rval = r2(ground_truth[:, i].cpu().numpy(), acti[:, j].cpu().numpy())
            id_list.append([rval, j])
        id_list.sort(reverse=True)
        sumacti = 0
        
        maxr2 = -100
        for j in range(acti.shape[1]):
            sumacti = sumacti + acti[:, id_list[j][1]]
            maxr2 = max(maxr2, r2(ground_truth[:, i].cpu().numpy(), sumacti.cpu().numpy()))
            if not allow_prefix:
                break
        baseline_r2.append(maxr2)
    return np.mean(baseline_r2)
def mmcs(D, F):
    # D ground truth embeddings
    D = D.cpu().numpy()
    F = F.cpu().numpy()
    dreg = D / np.linalg.norm(D, axis = 1)[..., np.newaxis]
    freg = F / np.linalg.norm(F, axis = 1)[..., np.newaxis]
    cs = np.einsum('ij,kj->ik', dreg, freg)
    maxv = np.amax(cs, axis=1)
    mmcs = np.mean(maxv)
    return mmcs