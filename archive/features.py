# %%


def GD_solver(dataset, guess_factor = 8, lr = 2e-3, steps = 10000, init_lamb = 0.1, lamb_left = 0.4, lamb_right = 0.6, negative_penalty = 1, adaptive = True):
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if (solver_lamb > lamb_left * maxA and solver_lamb < lamb_right * maxA) or not adaptive:
            c_est = solver.loss().item() / maxA / solver_lamb / config.n_samples
            info = {"loss": solver.loss().item(), "loss_est": c_est, "lamb": solver_lamb, "maxA": maxA, "guess_factor": guess_factor, "negative_penalty": negative_penalty, "lr": lr, "steps": steps, "size": dataset.shape[0]}
            return cur_feat, cur_emb, info
        else:
            solver_lamb = (lamb_left + lamb_right) / 2 * maxA

def get_feat_top(words, feat, id, num):
    result = []
    result_acti = []
    sorted = feat[:, id].argsort(descending = True)
    for j in range(num):
        result.append(words[sorted[j]])
        result_acti.append(feat[sorted[j], id].item())
    return result, result_acti


def metric_total_acti(feats1):
    import numpy as np
    srt_feats = abs(feats1).detach().numpy()
    sorted = np.flip(np.sort(srt_feats, axis = 1), axis = 1)
    pltarray = np.mean(sorted, axis = 0)
    return pltarray.sum() / pltarray[0]

def intersecting_feats(feat1, emb1, emb2, threshold = 0.95):
    import torch
    print(feat1.shape, emb1.shape, emb2.shape)
    dot_prod = torch.matmul(emb1, emb2.T)
    # only keep the intersection
    in_inter = (dot_prod > threshold).sum(dim = 1) > 0
    return feat1[:, in_inter], emb1[in_inter]

def condense_features(feats, emb, threshold = 0.95):
  '''
  Input:
    feats: a tensor of shape (n_samples, n_features)
    emb: a tensor of shape (n_features, n_emb)
    threshold: a float number
  Returns:
    new_feats (n_samples, n_new_features), new_emb (n_new_features, n_emb)
    where features are condensed when their embeddings are similar i.e. dot product > threshold

  '''
    
  def first_nonzero(x, axis=0):
      nonz = (x > 0)
      return ((nonz.cumsum(axis) == 1) & nonz).max(axis)
  
  import torch
  dot_prod = torch.matmul(emb, emb.T)
  print(feats.shape, emb.shape, dot_prod.shape)
  
  n_feats = feats.shape[1]
  min_idx = [0] * n_feats
  for i in range(n_feats):
    for j in range(n_feats):
      if dot_prod[i][j] > threshold:
        min_idx[i] = j
        break
  new_feats = torch.zeros_like(feats)
  index_id = dict()
  id_cnt = 0
  for i in range(n_feats):
    if min_idx[i] == i:
      index_id[i] = id_cnt
      id_cnt += 1
    else:
      index_id[i] = index_id[min_idx[i]]
  result_feats = torch.zeros(feats.shape[0], id_cnt).to(feats.device)
  for i in range(n_feats):
    result_feats[:, index_id[i]] += feats[:, i]
  result_emb = torch.zeros(id_cnt, emb.shape[1]).to(feats.device)
  for i in range(n_feats):
    result_emb[index_id[i]] += emb[i]
  #normalize result_emb to have L2 norm 1
  result_emb = result_emb / torch.norm(result_emb, dim = 1, keepdim = True)
  return result_feats, result_emb
# %%