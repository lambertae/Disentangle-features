
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
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

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
r2list = [[] for i in range(3)]
sparsity_list = [[] for i in range(3)]
mmcs_list = [[] for i in range(3)]
answer_list = [[] for i in range(3)]
dimrange = [20, 40, 80, 160, 320]
for dims in dimrange:
    num_feats = 5
    features = 4 * dims
    acti, metainfo = noisy_toy(num_feats / features, 0.3 / (dims ** 0.5), dims, features, 4096 * 4)
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
        corr = average_r2(true_feat, feat)
        sparsity = metric_total_acti(feat)
        mmcs_val = mmcs(true_emb, emb)
        r2list[tp].append(corr)
        sparsity_list[tp].append(sparsity)
        mmcs_list[tp].append(mmcs_val)
        print(f"Dimension:{dims}, type:{tp}")
        print(f"Average correlation:{corr}")
        print(f"Sparsity:{sparsity}, actual:{metric_total_acti(true_feat.cpu())}")
        print(f"MMCS:{mmcs_val}")
# %%
import matplotlib.pyplot as plt
name = ["Autoencoder", "Deep", "GD"]
def plot_figure(values, title):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.xscale("log")
    plt.xticks(dimrange, dimrange)
    for i in range(3):
        plt.plot(dimrange, values[i], label = name[i])
    plt.legend()
    plt.savefig(f"exp1_large_{title}.png")
    plt.show()
plot_figure(r2list, "Average correlation")
plot_figure(sparsity_list, "Sparsity")
plot_figure(mmcs_list, "MMCS")
# %%
