# %%
import sys
sys.path.append('/scratch/dengm/distent/Disentangle-features/archive')
from features import *
from metrics import *
import os
import torch as t
import torch.nn as nn
import pickle
def norminfty(x):
    return x.abs().max(dim = 1)[0] + 1e-8
def normp(p):
    return lambda x: x.abs().pow(p).sum(dim = 1)
def norm_ord(x, p):
    return normp(p)(x)
def sparse_metric(pw):
    return lambda feat: (norm_ord(feat, pw) / norminfty(feat) ** pw).mean()
def sparse_metric_mean(pw):
    return lambda feat: (norm_ord(feat, pw).mean() / norminfty(feat).mean() ** pw) 
def prt_metric(name, func):
    print(f"{name}: decoding {func(feat)}, true {func(true_feat)}")
# %%
# use lru_cache to cache results
from functools import lru_cache
@lru_cache(maxsize = None)
def load_wrapper(type, name):
    import json
    import pickle
    namestr = f"run_{type}_{name}.pkl"
    with open(namestr, "rb") as f:
        feat, emb, info = pickle.load(f)
    config_file = "config.json"
    config_dict = dict()
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if namestr in config_dict:
        args = config_dict[namestr]["args"]
        kwargs = config_dict[namestr]["kwargs"]
        print("Loaded config: ", args, kwargs)
    return feat, emb, info
# %%
import matplotlib.pyplot as plt
num_feats_list = [3, 5, 10, 20, 0]
dims_list = [20, 40, 80, 160]
power_coeff = [0.7, 1, 1.5]
# set figure size (len(num_feats_list) * 10, 10)
plt.figure(figsize = (10, (len(num_feats_list) // 2 + 1) * 10))
for num_feat in num_feats_list:
    plt.subplot(len(num_feats_list) // 2 + 1, 2, num_feats_list.index(num_feat) + 1)
    # plt.ylim(0, 20)
    plt.xscale("log")
    plt.xticks(dims_list, dims_list)
    for p in power_coeff:
        metric_list = []
        mean_metric_list = []
        for dims in dims_list:
            res = load_wrapper("gd", f"toy_dim{dims}_feats{num_feat if num_feat != 0 else dims}")
            feat, emb, info = res
            n_feats = num_feat if num_feat != 0 else dims
            ratio = sparse_metric_mean(p)(feat) * (1+p)-p#/ ((n_feats - 1) / (1 + p) + 1)
            mean_metric_list.append(ratio)
            # mean_metric_list.append((feat))
        # plt.plot(dims_list, metric_list, label = f"p = {p}")
        plt.plot(dims_list, mean_metric_list, label = f"p = {p} mean")
    # if num_feat != 0:
    #     plt.plot(dims_list, [num_feat] * len(dims_list), label = "Ground Truth", linestyle = "--")
    if num_feat == 0:
        plt.title(f"Sparsity of dims features")
    else:
        plt.title(f"Sparsity of {num_feat} features")
    plt.xlabel("Dimension")
    plt.ylabel("Estimated Sparsity/Actual sparsity")
    plt.legend()
plt.savefig("sparsity.png")
# %%
