# %%

def AutoEncoder_solver(dataset, guess_factor = 8, lr = 2e-3, epochs = 1000, batch_size = 256, init_lamb = 0.1, lamb_left = 0.4, lamb_right = 0.6, adaptive = True, use_deep = False):
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
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import trange

    def norm_(w):
        with torch.no_grad():
            norm = torch.linalg.norm(w, ord=2, dim=0)
            w /= norm

    class AutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.ReLU()
            )
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
            self.encoder.apply(init_weights)
            torch.nn.init.orthogonal_(self.decoder.weight)

        def forward(self, x):
            encoded = self.encoder(x)
            with torch.no_grad():
                num_coeffs = torch.count_nonzero(encoded, dim=1)
                num_coeffs = torch.sum(num_coeffs)
            norm_(self.decoder.weight)
            decoded = self.decoder(encoded)
            return encoded, decoded, num_coeffs

    class DeepAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()

            self.enc1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 4, bias=True),
                nn.ReLU()
            )
            self.enc2 = nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 2, bias=True),
                nn.ReLU()
            )
            self.enc3 = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim, bias=True),
                nn.ReLU()
            )
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

            
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
            #self.enc1.apply(init_weights)
            #self.enc2.apply(init_weights)
            #self.enc3.apply(init_weights)
            torch.nn.init.orthogonal_(self.decoder.weight)

        def forward(self, x):
            x = self.enc1(x)
            x = self.enc2(x)
            encoded = self.enc3(x)

            with torch.no_grad():
                num_coeffs = torch.count_nonzero(encoded, dim=1)
                num_coeffs = torch.sum(num_coeffs)

            norm_(self.decoder.weight)

            decoded = self.decoder(encoded)

            return encoded, decoded, num_coeffs
    def l1_reg(x):
        # x is num_examples x num_features
        n_x = torch.linalg.norm(x, ord=1, dim=1) / x.shape[1]
        return torch.mean(n_x)


    def run_disentangle(dataset, num_features=512, feature_dim=256, reg_param=0.1, batch_size=256, epochs=1000, lr=1e-3):
        print(f"Executing with num_features: {num_features} and feature_dim: {feature_dim} and reg_param: {reg_param}")

        recon_loss_fn = nn.MSELoss()
        loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        model = AutoEncoder(feature_dim, num_features)
        if use_deep:
            model = DeepAutoEncoder(feature_dim, num_features)
        model = model.to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model = model.double()
        model.train()
        # print(model.decoder.weight.shape)

        t = trange(epochs, desc='Epoch loss: ', leave=True)
        #patience = 500
        best_epoch_loss = 100000000
        for _ in t:
            running_loss, num_examples = 0, 0
            best_coeffs = 0
            avg_coeffs = 0.0
            num_count = 0
            for _, features in enumerate(loader):
                optimizer.zero_grad()
                features = features.to('cuda').double()
                encoded, decoded, num_coeffs = model(features)
                loss =  recon_loss_fn(decoded, features) +  reg_param * l1_reg(encoded)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_examples += len(features)
                avg_coeffs += num_coeffs
                num_count += 1
            
            epoch_loss = running_loss / num_count
            avg_coeffs /= num_examples
            t.set_description(f"Epoch loss: {epoch_loss:.10f}, Avg coeffs: {avg_coeffs:.3f}")
            t.refresh()
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_coeffs = avg_coeffs
        print(f"Best coeffs:{best_coeffs}")
        return model, best_coeffs
    solver_lamb = init_lamb
    while 1:
      model, coeffs = run_disentangle(dataset, num_features=dataset.shape[1] * guess_factor, feature_dim=dataset.shape[1], reg_param=solver_lamb, batch_size=batch_size, epochs=epochs, lr=lr)
      import numpy as np 
      encoded, decoded, num_coeffs = model(dataset.to('cuda').double())
      with torch.inference_mode():
          model.eval()
          cur_feat = encoded.cpu().clone().detach()
          cur_emb = model.decoder.weight.T.clone().cpu().detach()
          print(cur_emb.shape)
      
      srt_feats = abs(cur_feat).cpu().detach().numpy()
      sorted = np.flip(np.sort(srt_feats, axis = 1), axis = 1)
      pltarray = np.mean(sorted, axis = 0) #* np.arange(1, 101)
      maxA = pltarray[0]
      if (solver_lamb > lamb_left * maxA and solver_lamb < lamb_right * maxA) or not adaptive:
          info = {"lamb": solver_lamb, "maxA": maxA, "guess_factor": guess_factor, "lr": lr}
          return cur_feat, cur_emb, info
      else:
          solver_lamb = (lamb_left + lamb_right) / 2 * maxA

def SVD_solver(dataset, guess_factor = 8, lr = 2e-3, steps = 10000, greedy_step = 100, init_lamb = 0.1, lamb_left = 0.4, lamb_right = 0.6, negative_penalty = 1, adaptive = True, init_feat = None, init_emb = None):
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

    # activations: (S, E)
    # embs: (F, E)
    # lamb: float
    # feats: (S, F)

    def greedy_feats(embs, lamb):
        embs = embs.clone().detach().to(dataset.device)
        # no grad
        embs.requires_grad_(False)
        # norms = torch.norm(embs, dim = 1, p = 2)
        embs = embs / torch.norm(embs, dim = 1, p = 2)[:, None]
        ids = torch.arange(dataset.shape[0]).to(dataset.device)
        feats = torch.zeros((dataset.shape[0], embs.shape[0])).to(dataset.device)
        remainder = dataset.clone().detach()
        remainder.requires_grad_(False)
        eps = 1e-2
        while ids.shape[0] > 0:
            # print(ids.shape)
            dot_prod = torch.einsum("se,fe->sf", remainder, embs) # (Sx, F)
            max_elts, max_ids = torch.max(dot_prod, dim = 1)
            mask = max_elts > lamb / 2 + eps
            if mask.sum() == 0:
                break
            remainder = remainder[mask]
            sel_ids = ids[mask]
            sel_mxid = max_ids[mask]
            sel_dot = max_elts[mask] - lamb / 2
            remainder -= sel_dot[:, None] * embs[sel_mxid, :]
            feats[sel_ids, sel_mxid] += sel_dot
            ids = sel_ids
        # feats = feats / norms[None, :]
        return feats, embs
    
    def construction_loss(feat, embs):
        return (feat @ (embs / embs.norm(dim = 1, p = 2)[:, None]) - dataset).norm(p = 2) ** 2
    def total_loss(feat, embs, lamb):
        return lamb * feat.norm(p = 1).sum() + construction_loss(feat, embs)

    solver_lamb = init_lamb
    while 1:
        if init_emb is None:
            embs = torch.randn((dataset.shape[1] * guess_factor, dataset.shape[1])).to(dataset.device)
        else:
            embs = init_emb.clone().detach().to(dataset.device)
            print("init")
        if init_feat is None:
            feats = torch.randn((dataset.shape[0], dataset.shape[1] * guess_factor)).to(dataset.device) * (solver_lamb / 0.1)
        else:
            feats = init_feat.clone().detach().to(dataset.device)
            print("init")
        feats = feats * embs.norm(dim = 1, p = 2)
        embs = embs / embs.norm(dim = 1, p = 2)[:, None]
        # print("init construction loss", construction_loss(feats, embs))
        # print("init total loss", total_loss(feats, embs, solver_lamb))

        class Solver(nn.Module):
            def __init__(self, embs):
                super().__init__()
                self.embs = nn.Parameter(embs.clone().detach().to(embs.device))
            def loss(self):
                return construction_loss(feats, self.embs)
        solver = Solver(embs)
        optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
        
        with trange(steps) as tr:
            for i in tr:
                if i % greedy_step == 0:
                    feats = greedy_feats(solver.embs.clone().detach(), solver_lamb)[0]
                optimizer.zero_grad()
                loss = solver.loss()
                loss.backward(retain_graph=True)
                optimizer.step()
                # wandb.log({"solver_loss": loss.item()})
                if i % greedy_step == 0:
                    rec_loss = construction_loss(feats, solver.embs)
                    tr.set_postfix(loss = total_loss(feats, solver.embs, solver_lamb).item(), rec_loss = rec_loss.item())
        
        with torch.inference_mode():
            solver.eval()
            # feats *= torch.norm(solver.embs, dim = 1, p = 2)
            solver.embs /= torch.norm(solver.embs, dim = 1, p = 2)[:, None]
        cur_feat = feats.cpu().clone().detach()
        cur_emb = solver.embs.clone().cpu().detach()
        
        srt_feats = abs(cur_feat).cpu().detach().numpy()
        sorted = np.flip(np.sort(srt_feats, axis = 1), axis = 1)
        pltarray = np.mean(sorted, axis = 0) #* np.arange(1, 101)
        maxA = pltarray[0]
        # print(maxA, solver_lamb)
        if (solver_lamb > lamb_left * maxA and solver_lamb < lamb_right * maxA) or not adaptive:
            loss = total_loss(feats, solver.embs, solver_lamb)
            c_est = loss.item() / maxA / solver_lamb / dataset.shape[0]
            info = {"loss": loss.item(), "loss_est": c_est, "lamb": solver_lamb, "maxA": maxA, "guess_factor": guess_factor, "negative_penalty": negative_penalty, "lr": lr, "steps": steps, "size": dataset.shape[0]}
            return cur_feat, cur_emb, info
        else:
            solver_lamb = (lamb_left + lamb_right) / 2 * maxA


def GD_solver(dataset, guess_factor = 8, lr = 2e-3, steps = 10000, init_lamb = 0.1, lamb_left = 0.4, lamb_right = 0.6, negative_penalty = 1, adaptive = True, init_feat = None, init_emb = None):
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
            if init_emb is None:
                nn.init.xavier_normal_(self.feature_emb)
            else:
                self.feature_emb = nn.Parameter(init_emb.clone().detach().to(device))
            self.features_of_samples = nn.Parameter(torch.zeros((config.n_samples, config.n_features), requires_grad=True).to(device))
            if init_feat is None:
                nn.init.xavier_normal_(self.features_of_samples, gain = lamb / 0.1)
            else:
                self.features_of_samples = nn.Parameter(init_feat.clone().detach().to(device))
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
                rec_loss = torch.norm(solver.features_of_samples @ solver.feature_emb - solver.activations, p=2) ** 2
                solver_optimizer.step()
                # wandb.log({"solver_loss": loss.item()})
                if i % 100 == 0:
                    tr.set_postfix(accuracy = acc, loss = loss.item(), rec_loss = rec_loss.item())
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
    '''
    Get the top num words in feat[:, id]
    Return: a list of words, a list of activations
    '''
    result = []
    result_acti = []
    sorted = feat[:, id].argsort(descending = True)
    for j in range(num):
        result.append(words[sorted[j]])
        result_acti.append(feat[sorted[j], id].item())
    return result, result_acti


def metric_total_acti(feats1):
    import numpy as np
    srt_feats = abs(feats1).clone().cpu().detach().numpy()
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

def wrapper(type, name, *args, **kwargs):
    import json
    import pickle
    import os
    import torch as t
    
    if type == "ae":
        res = AutoEncoder_solver(*args, **kwargs)
    elif type == "gd":
        res = GD_solver(*args, **kwargs)
    elif type == "svd":
        res = SVD_solver(*args, **kwargs)
    else:
        raise NotImplementedError
    feat, emb, info = res
    # save
    savedir = "./saved"
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    namestr = f"{savedir}/run_{type}_{name}.pkl"
    config_file = f"{savedir}/config.json"
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
# lru_cache import
from functools import lru_cache
@lru_cache(maxsize = None)
def load_wrapper(type, name):
    import json
    import pickle
    import os
    savedir = "./saved"
    namestr = f"{savedir}/run_{type}_{name}.pkl"
    with open(namestr, "rb") as f:
        feat, emb, info = pickle.load(f)
    config_file = f"{savedir}/config.json"
    config_dict = dict()
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if namestr in config_dict:
        args = config_dict[namestr]["args"]
        kwargs = config_dict[namestr]["kwargs"]
        print("Loaded config: ", args, kwargs)
    return feat, emb, info