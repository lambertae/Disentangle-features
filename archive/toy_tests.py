# %%

from features import *
import os
import torch as t
import torch.nn as nn
from features import *
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# %%
def GD_solver(dataset, guess_factor = 8, lr = 2e-3, steps = 10000, init_lamb = 0.1, adaptive = True, negative_penalty = 1):
    '''
    dataset: (n_samples, n_hiddens): record the activations
    Return:
        cur_feat: (n_samples, n_features): the features of samples
        cur_emb: (n_features, n_hiddens): the embedding of features
        info: a dictionary containing loss, loss_est, lamb, maxA
    '''
    # dataset: (n_samples, n_hiddens): record the activations 
    import torch
    import torch.nn as nn
    import numpy as np
    from tqdm import trange
    from dataclasses import dataclass

    @dataclass
    class Config:
        n_features: int
        n_hidden: int
        n_samples: int
        negative_penalty: float

    class Solver(nn.Module):

        def __init__(self, 
                    config,        
                    activations, 
                    lamb):
        # self.feature_emb: (n_features, n_hidden); self.features_of_samples: (n_samples, n_features)
        # Goal: while ensuring columns of self.feature_emb have norm 1, minimize self.config.lamb * L_1(self.features_of_samples) + L2(self.features_of_samples @ self.feature_emb - activations)
            super().__init__()
            device = activations.device
            self.config = config
            self.lamb = lamb
            self.feature_emb = nn.Parameter(torch.empty((config.n_features, config.n_hidden), requires_grad=True).to(device))
            nn.init.xavier_normal_(self.feature_emb)
            self.features_of_samples = nn.Parameter(torch.zeros((config.n_samples, config.n_features), requires_grad=True).to(device))
            nn.init.xavier_normal_(self.features_of_samples, gain = lamb / 0.1)
            # self.features_of_samples = self.features_of_samples * lamb / 0.1
            self.activations = torch.Tensor(activations).to(device)
            self.negative_penalty = config.negative_penalty


        def loss(self):
        # if enforce non negative: need to penalize negative weights
            return self.lamb * torch.norm(self.features_of_samples * (1 + (self.features_of_samples < 0) * self.negative_penalty) * torch.norm(self.feature_emb, dim = 1, p = 2), p=1) + torch.norm(self.features_of_samples @ self.feature_emb - self.activations, p=2) ** 2

    solver_lamb = init_lamb
    while 1:
        config = Config(
            n_features = dataset.shape[1] * guess_factor,
            n_hidden = dataset.shape[1],
            n_samples = dataset.shape[0], 
            negative_penalty = negative_penalty
        )
        solver_lr = lr
        solver_steps = steps
        # wandb.config.lamb = solver_lamb
        # wandb.config.solver_lr = solver_lr
        # wandb.config.solver_steps = solver_steps
        solver = Solver(config, activations=dataset, lamb=solver_lamb)
        # solver.training()
        solver_optimizer = torch.optim.Adam(solver.parameters(), lr=solver_lr)

        acc = 0
        with trange(solver_steps) as tr:
            for i in tr:
                solver_optimizer.zero_grad()
                loss = solver.loss()
                loss.backward(retain_graph=True)
                solver_optimizer.step()
                # wandb.log({"solver_loss": loss.item()})
                if i % 100 == 0:
                    tr.set_postfix(accuracy = acc, loss = loss.item())
        with torch.inference_mode():
            solver.eval()
            solver.features_of_samples *= torch.norm(solver.feature_emb, dim = 1, p = 2)
            solver.feature_emb /= torch.norm(solver.feature_emb, dim = 1, p = 2)[:, None]
        cur_feat = solver.features_of_samples.cpu().clone().detach()
        cur_emb = solver.feature_emb.clone().cpu().detach()
        
        srt_feats = abs(cur_feat).cpu().detach().numpy()
        sorted = np.flip(np.sort(srt_feats, axis = 1), axis = 1)
        pltarray = np.mean(sorted, axis = 0) #* np.arange(1, 101)
        maxA = pltarray[0]
        if (solver_lamb > 0.05 * maxA and solver_lamb < 0.2 * maxA) or not adaptive:
            c_est = solver.loss().item() / maxA / solver_lamb / config.n_samples
            info = {"loss": solver.loss().item(), "loss_est": c_est, "lamb": solver_lamb, "maxA": maxA}
            return cur_feat, cur_emb, info
        else:
            solver_lamb = 0.1 * maxA

# %%
import torch as t
import numpy as np
from sklearn.metrics import r2_score 
# p value of the features
from scipy.stats import pearsonr

def p_value(x, y):
    return pearsonr(x, y)[1]
noise_level = 0.3
hid = 10
true = 30
samples = 4000
emb = t.randn(true, hid).to(device)
emb /= t.norm(emb, dim = 1).reshape(-1, 1)
emb = emb.to(device)
ground_truth = (t.rand((samples, true)) < 3 / 30).to(t.float32).to(device)
acti = ground_truth @ emb
print(device)
noise = t.randn(acti.shape).to(acti.device)
noise = noise / t.norm(noise, dim = 1, p = 2)[:, None]
acti = acti + noise * noise_level


def r2(x, y):
    return r2_score(x, y)
# %%
print(acti.shape, ground_truth.shape)
baseline_r2 = []
for i in range(ground_truth.shape[1]):
    maxr2 = -100
    for j in range(acti.shape[1]):
        # print(r2(ground_truth[:, i].cpu().numpy(), feat[:, j].cpu().numpy()))
        rval = r2(ground_truth[:, i].cpu().numpy(), acti[:, j].cpu().numpy())
        if rval > maxr2:
            maxr2 = rval
    baseline_r2.append(maxr2)
print(np.mean(baseline_r2))
baseline = np.mean(baseline_r2)
# %%
lamb_list = [1e-4, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
mean_r2 = []
mmcs_list = []
feat_list = []
emb_list = []
info_list = []
for lamb in lamb_list:
    #lamb = 0.1 * (4 ** loglamb)
    feat, guess_emb, info = GD_solver(acti, guess_factor = 8, lr = 2e-3, steps = 5000, init_lamb = lamb, adaptive = False)
    feat, guess_emb = condense_features(feat, guess_emb, 0.9)
    print(feat.shape, guess_emb.shape)
    feat_list.append(feat)
    emb_list.append(guess_emb)
    info_list.append(info)
    print(info)
    r2s = []
    # print(ground_truth[:, 0], feat[:, 0])
    for i in range(ground_truth.shape[1]):
        maxr2 = -100
        best_id = -1
        pv = 0
        for j in range(feat.shape[1]):
            # print(r2(ground_truth[:, i].cpu().numpy(), feat[:, j].cpu().numpy()))
            rval = r2(ground_truth[:, i].cpu().numpy(), feat[:, j].cpu().numpy())
            p = p_value(ground_truth[:, i].cpu().numpy(), feat[:, j].cpu().numpy())
            if rval > maxr2:
                maxr2 = rval
                best_id = j
                pv = p
        r2s.append(maxr2)
    mean_r2.append(np.mean(r2s))
    print(np.mean(r2s))
    def mmcs(D, F):
        dreg = D / np.linalg.norm(D, axis = 1)[..., np.newaxis]
        freg = F / np.linalg.norm(F, axis = 1)[..., np.newaxis]
        # print(np.linalg.norm(dreg, axis = 1))
        # print(dreg.shape, freg.shape)
        cs = np.einsum('ij,kj->ik', dreg, freg)
        maxv = np.amax(cs, axis=1)
        mmcs = np.mean(maxv)
        print(mmcs)
        return mmcs
    mmcs_list.append(mmcs(emb.cpu().numpy(), guess_emb.cpu().numpy()))
    print(mmcs_list[-1])

# %%
import matplotlib.pyplot as plt
# get 2 x 2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# plot the data on first subplot
axs[0, 0].plot(lamb_list, mean_r2)
axs[0, 0].set_title('R2 score')
# plot the data on second subplot
axs[0, 1].plot(lamb_list, mmcs_list, 'tab:orange')
axs[0, 1].set_title('MMCS')
# plot the data on third subplot
axs[1, 0].plot(lamb_list, [min(info["loss_est"], 20) for info in info_list], 'tab:green', label = "loss_est")
axs[1, 0].plot(lamb_list, [ground_truth.sum(dim = 1).mean().item() for info in info_list], 'tab:red', label = "True")
axs[1, 0].set_title('loss_est')


axs[1, 1].plot(lamb_list, [metric_total_acti(x) for x in feat_list], 'tab:blue', label = "Average activated features")
axs[1, 1].set_title('Number of activated samples')
axs[1, 1].plot(lamb_list, [ground_truth.sum(dim = 1).mean().item() for info in info_list], 'tab:red', label = "True")

# legend
axs[1, 0].legend()
axs[1, 1].legend()

# log scale the x axis
for ax in axs.flat:
    # ax.set(xlabel='lambda', ylabel='value')
    ax.set_xscale("log")
# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
    # ax.label_outer()
plt.show()

# xscale log
# %%

plt.xscale("log")
plt.title("R2 score of the features")
plt.xlabel("lambda")
plt.ylabel("R2 score")
plt.plot(lamb_list, mean_r2, label = "R2 score")
# plt.plot(lamb_list, [baseline] * len(lamb_list), label = "baseline")
plt.legend()

# %%
import matplotlib.pyplot as plt
# xscale log
plt.xscale("log")
plt.title("MMCS of feature embedding")
plt.xlabel("lambda")
plt.ylabel("MMCS")
plt.plot(lamb_list, mmcs_list, label = "MMCS")
# plt.plot(lamb_list, [baseline] * len(lamb_list), label = "baseline")
plt.legend()

# %%
import matplotlib.pyplot as plt
# xscale log
plt.xscale("log")
plt.title("Estimated number of features based on lambda")
plt.xlabel("lambda")
plt.ylabel("MMCS")
plt.plot(lamb_list[:-1],[x['loss_est'] for x in info_list[:-1]], label = "lambda est")

plt.plot(lamb_list[:-1],[3 for x in info_list[:-1]], label = "actual")
# plt.plot(lamb_list, [baseline] * len(lamb_list), label = "baseline")
plt.legend()

# %%

import matplotlib.pyplot as plt
# xscale log
plt.xscale("log")
plt.title("Estimated number of features based on activation")
plt.xlabel("lambda")
plt.ylabel("MMCS")
plt.plot(lamb_list,[metric_total_acti(x) for x in feat_list], label = "est")

plt.plot(lamb_list,[3 for x in info_list], label = "actual")
# plt.plot(lamb_list, [baseline] * len(lamb_list), label = "baseline")
plt.legend()
# %%
mmcs_list = []
lambda_list = []
# for loglamb in range(0.05, 1.5, 0.05):
    # lamb = 0.1 * (2 ** loglamb)

inters = []
for lamb in [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
    feat, guess_emb, info = GD_solver(acti, guess_factor = 8, lr = 1e-2, steps = 4000, init_lamb = lamb, adaptive = False)
    feat2, emb2, info2 = GD_solver(acti, guess_factor = 8, lr = 1e-2, steps = 4000, init_lamb = lamb, adaptive = False)
    inter_feats, inter_emb = intersecting_feats(feat, guess_emb, emb2, 0.9)
    print(inter_emb.shape)
    inter_feats, inter_emb = condense_features(inter_feats, inter_emb, 0.9)
    print(inter_emb.shape)
    inters.append(inter_emb)
    mmcs_list.append(mmcs(emb.cpu().numpy(), inter_emb.cpu().numpy()))
    lambda_list.append(lamb)

# %%
import matplotlib.pyplot as plt
plt.plot(lambda_list, mmcs_list)

# %%
extent = [mmcs(inters[i].cpu().numpy(), inters[j].cpu().numpy()) for i in range(len(inters)) for j in range(0, len(inters))]
# plot extent
plt.imshow(np.array(extent).reshape(len(inters), len(inters)))
plt.colorbar()
# %%
baseline = mmcs(acti.cpu().numpy(), t.randn_like(guess_emb).cpu().numpy())
# get PCA of acti
from sklearn.decomposition import PCA
pca = PCA(n_components=acti.shape[1])
pca.fit(acti.cpu().numpy())
pca_emb = t.Tensor(pca.components_).to(acti.device)
pca_mmcs = mmcs(acti.cpu().numpy(), pca_emb.cpu().numpy())

# %%
