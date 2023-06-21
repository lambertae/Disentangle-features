# %%
import sys
sys.path.append('/scratch/dengm/distent/Disentangle-features/archive')
from features import *
from metrics import *
import os
import torch as t
import torch.nn as nn
import pickle
# from features import *

# %%
def wrapper(type, name, *args, **kwargs):
    import json
    import pickle
    
    if type == "ae":
        res = AutoEncoder_solver(*args, **kwargs)
    elif type == "gd":
        res = GD_solver(*args, **kwargs)
    else:
        raise NotImplementedError
    feat, emb, info = res
    # save
    namestr = f"run_{type}_{name}.pkl"
    config_file = "config.json"
    # check if config file exists
    config_dict = dict()
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_dict = json.load(f)
    # turn args, kwargs into json-serializable, removing tensors
    args = [str(arg) for arg in args if not isinstance(arg, t.Tensor)]
    kwargs = {key: str(val) for key, val in kwargs.items() if not isinstance(val, t.Tensor)}
    config_dict[namestr] = {"args": args, "kwargs": kwargs}
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    with open(namestr, "wb") as f:
        pickle.dump((feat, emb, info), f)
    return res
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def noisy_toy(prob, noise_level, hidden, features, size, dtype = t.float32):    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    true_emb = t.randn(features, hidden).to(device)
    true_emb /= t.norm(true_emb, dim = 1).reshape(-1, 1)
    true_emb = true_emb.to(device)
    ground_truth = (t.rand((size, features)) < prob).to(dtype).to(device) * t.rand((size, features)).to(dtype).to(device)
    acti = ground_truth @ true_emb
    noise = t.randn(acti.shape).to(acti.device) * noise_level
    acti = acti + noise * noise_level
    info = {"emb": true_emb, "feat": ground_truth}
    return acti, info
num_feats_list = [0]
dims_list = [20, 40, 80, 160, 320]
for num_feats in num_feats_list:
    for dims in dims_list:
        # if num_feats == 0:
        num_feats = dims
        features = dims * 4
        acti, metainfo = noisy_toy(num_feats / features, 0.3 / (dims ** 0.5), dims, features, 4096)
        true_feat = metainfo["feat"]
        true_emb = metainfo["emb"]
        res = wrapper("gd", f"toy_dim{dims}_feats{num_feats}", acti.clone(), guess_factor = 8, lr = 3e-3, steps = 10000, init_lamb = 0.35, lamb_left = 0.4, lamb_right = 0.6, adaptive = False)
# %%

# num_feats_list = [3, 5, 10, 20, 0]
# dims_list = [20, 40, 80, 160, 320]
# for num_feats in num_feats_list:
#     for dims in dims_list:
#         if num_feats == 0:
#             num_feats = dims
#         features = dims * 4
#         acti, metainfo = noisy_toy(num_feats / features, 0.3 / (dims ** 0.5), dims, features, 4096)
#         true_feat = metainfo["feat"]
#         true_emb = metainfo["emb"]
#         res = wrapper("gd", f"toy_dim{dims}_feats{num_feats}", acti.clone(), guess_factor = 8, lr = 3e-3, steps = 10000, init_lamb = 0.35, lamb_left = 0.4, lamb_right = 0.6, adaptive = False)