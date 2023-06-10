
# %%
import sys
sys.path.append('/scratch/dengm/distent/Disentangle-features/archive')
from features import *
from metrics import *
import os
import torch as t
import torch.nn as nn
# from features import *

# %%
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def noisy_toy(prob, noise_level, hidden, features, size, dtype = t.float32):    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    true_emb = t.randn(features, hidden).to(device)
    true_emb /= t.norm(true_emb, dim = 1).reshape(-1, 1)
    true_emb = true_emb.to(device)
    ground_truth = (t.rand((size, features)) < prob).to(dtype).to(device)
    acti = ground_truth @ true_emb
    noise = t.randn(acti.shape).to(acti.device) * noise_level
    acti = acti + noise * noise_level
    info = {"emb": true_emb, "feat": ground_truth}
    return acti, info
r2_best_list = [[] for i in range(3)]
r2_pref_list = [[] for i in range(3)]
sparsity_list = [[] for i in range(3)]
mmcs_list = [[] for i in range(3)]
answer_list = [[] for i in range(3)]
dimrange = [10, 20, 40, 80, 160, 320]
for dims in dimrange:
    num_feats = 5
    features = 4 * dims
    acti, metainfo = noisy_toy(num_feats / features, 0.3 / (dims ** 0.5), dims, features, 8192)
    true_feat = metainfo["feat"]
    true_emb = metainfo["emb"]
    for tp in range(3):
        if tp == 0: # autoencoder
            feat, emb, info = AutoEncoder_solver(acti.clone(), guess_factor = 8, lr = 1e-3, epochs = 2000, batch_size = 1024, init_lamb = 0.35, lamb_left = 0.4, lamb_right = 0.6, 
                                                 adaptive = False, use_deep = False)
        elif tp == 1: # deep
            feat, emb, info = AutoEncoder_solver(acti.clone(), guess_factor = 8, lr = 1e-3, epochs = 2000, batch_size = 1024, init_lamb = 0.35, lamb_left = 0.4, lamb_right = 0.6, 
                                                 adaptive = False, use_deep = True) 
        else:
            feat, emb, info = GD_solver(acti.clone(), guess_factor = 8, lr = 3e-3, steps = 10000, init_lamb = 0.35, lamb_left = 0.4, lamb_right = 0.6, adaptive = False)
        answer_list[tp].append((feat, emb, info))
        corr_best = average_r2(true_feat[:, :min(len(true_feat), 100)], feat, False)
        corr_pref = average_r2(true_feat[:, :min(len(true_feat), 100)], feat, True)
        sparsity = metric_total_acti(feat)
        mmcs_val = mmcs(true_emb, emb)
        r2_best_list[tp].append(corr_best)
        r2_pref_list[tp].append(corr_pref)
        sparsity_list[tp].append(sparsity)
        mmcs_list[tp].append(mmcs_val)
        print(f"Dimension:{dims}, type:{tp}")
        print(f"Average correlation:{corr_best}, while allowing prefix:{corr_pref}")
        print(f"Sparsity:{sparsity}, actual:{metric_total_acti(true_feat.cpu())}")
        print(f"MMCS:{mmcs_val}")
# %%
# %%
