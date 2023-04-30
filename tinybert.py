# %%
from transformers import AutoModel

model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
emb = model.embeddings.word_embeddings.weight.detach().numpy()
print(emb.shape)

# %%
# get the word for each id
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# get 5000 elts without replacement
def get_acti(samples, emb, tokenizer):
    idList = np.random.choice(emb.shape[0], samples, replace = False)
    words = tokenizer.convert_ids_to_tokens(idList)
    embs = emb[idList]
    return embs, words

embs, words = get_acti(emb.shape[0], emb, tokenizer)
# %%


def GD_solver(dataset, guess_factor = 8, lr = 2e-3, steps = 10000, init_lamb = 0.1, negative_penalty = 1, adaptive = True):
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
            self.config = config
            self.lamb = lamb
            self.feature_emb = nn.Parameter(torch.empty((config.n_features, config.n_hidden), requires_grad=True).to(device))
            nn.init.xavier_normal_(self.feature_emb)
            self.features_of_samples = nn.Parameter(torch.zeros((config.n_samples, config.n_features), requires_grad=True).to(device))
            nn.init.xavier_normal_(self.features_of_samples, gain = lamb / 0.1)
            # self.features_of_samples = self.features_of_samples * lamb / 0.1
            self.activations = activations.clone().detach().to(device)
            self.negative_penalty = config.negative_penalty
            # now: add feature_emb and features_of_samples to parameters


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
        print(solver.activations.shape)
        # train
        # solver.to("cuda")
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
        if (solver_lamb > 0.1 * maxA and solver_lamb < 0.4 * maxA) or not adaptive:
            c_est = solver.loss().item() / maxA / solver_lamb / config.n_samples
            info = {"loss": solver.loss().item(), "loss_est": c_est, "lamb": solver_lamb, "maxA": maxA}
            return cur_feat, cur_emb, info
        else:
            solver_lamb = 0.2 * maxA
# from features import *
import os
import torch as t
os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)
print(embs.shape)
res_feat, res_emb, info = GD_solver(t.Tensor(embs).to(device), 15)
print(res_feat.shape, res_emb.shape, info)
# res_feat, res_emb = condense_features(res_feat, res_emb)
# print(res_feat.shape, res_emb.shape)

# %%
# save
np.save("bert/bert_tiny_res_feat.npy", res_feat.cpu().detach().numpy())
np.save("bert/bert_tiny_res_emb.npy", res_emb.cpu().detach().numpy())
with open("bert/bert_tiny_info.json", "w") as f:
    f.write(str(info))
# save embs, words
np.save("bert/bert_tiny_embs.npy", embs)
np.save("bert/bert_tiny_words.npy", words)

# %%
res_feat2, res_emb2, info2 = GD_solver(t.Tensor(embs).to(device), 15, init_lamb = info["lamb"]) 
res_feat2, res_emb2 = condense_features(res_feat2, res_emb2)
# %%
print(info, info2)

# save res_feat
np.save("res_feat.npy", res_feat.cpu().detach().numpy())


# %%
def display_feat(id, num):
    sorted = res_feat[:, i].argsort(descending = True)
    for j in range(num):
        # print(sorted[j].item(), res_feat[sorted[j], i].item())
        print(words[sorted[j]])

sum_feat = res_feat.sum(dim = 0)
import random
num = random.randint(0, 300)
print(num)
best_dims = sum_feat.argsort(descending = True)[num: num + 1]
for i in best_dims:
    display_feat(i, 20)

# %%
print(sum_feat.shape)
# %%
