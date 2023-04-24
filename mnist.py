# %%

import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from model import TransformerModel, ModelConfig
from data import ToyDataset, DataConfig
from tqdm import tqdm
from dataclasses import dataclass
from features import *
import os
#cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print(device)

# %%
def Train_MNIST_net():
  '''
  Train a simple neural network on MNIST dataset
	Return: a trained MNIST network. 
  '''
  # MNIST train 
  import torch
  import torchvision
  import torch.nn as nn

  n_epochs = 3
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10

  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_test, shuffle=True)
  examples = enumerate(test_loader)
  batch_idx, (example_data, example_targets) = next(examples)
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
          self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
          self.conv2_drop = nn.Dropout2d()
          self.fc1 = nn.Linear(320, 6)
          self.fc2 = nn.Linear(6, 10)

      def forward(self, x):
          x = F.relu(F.max_pool2d(self.conv1(x), 2))
          x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
          x = x.view(-1, 320)
          x = F.relu(self.fc1(x))
          x = F.dropout(x, training=self.training)
          x = self.fc2(x)
          return F.log_softmax(x)
  network = Net()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
  def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = network(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  test()
  for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
  return network


def Generate_MNIST(net):
  import torch
  import torchvision
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
    
  '''
  net: a trained network
  return: a tuple of (activations, data)
  '''
  
  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=64, 
    shuffle=True)
  
  def activation(net, x):
    net.eval()
    with torch.inference_mode():
      x = F.relu(F.max_pool2d(net.conv1(x), 2))
      x = F.relu(F.max_pool2d(net.conv2_drop(net.conv2(x)), 2))
      x = x.view(-1, 320)
      x = net.fc1(x)
      return x
  datas = []
  actis = []
  targets = []
  for data, target in train_loader:
    actis.append(activation(net, data))
    datas.append(data)
    targets.append(target)
  return torch.cat(actis, 0), torch.cat(datas, 0), torch.cat(targets, 0)
# %%
net = Train_MNIST_net()
# %%
acti, data, target = Generate_MNIST(net)
print(acti.shape, data.shape, target.shape)
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
            device = activations.device
            self.config = config
            self.lamb = lamb
            self.feature_emb = nn.Parameter(torch.empty((config.n_features, config.n_hidden), requires_grad=True)).to(device)
            nn.init.xavier_normal_(self.feature_emb)
            self.features_of_samples = nn.Parameter(torch.zeros((config.n_samples, config.n_features), requires_grad=True)).to(device)
            nn.init.xavier_normal_(self.features_of_samples, gain = lamb / 0.1)
            # self.features_of_samples = self.features_of_samples * lamb / 0.1
            self.activations = torch.Tensor(activations)
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
        if (solver_lamb > 0.1 * maxA and solver_lamb < 0.4 * maxA) or not adaptive:
            c_est = solver.loss().item() / maxA / solver_lamb / config.n_samples
            info = {"loss": solver.loss().item(), "loss_est": c_est, "lamb": solver_lamb, "maxA": maxA}
            return cur_feat, cur_emb, info
        else:
            solver_lamb = 0.2 * maxA
samples = 5000
feat, emb, info1 = GD_solver(acti[:samples], lr = 0.01, steps = 4000, init_lamb = 0.5)
# %%
print(info1)
feat, emb = condense_features(feat, emb, 0.9)
print(feat.shape, emb.shape)
# %%
feat2, emb2, info2 = GD_solver(acti[:samples], lr = 0.01, steps = 4000, init_lamb = 1.04)
feat3, emb3 = intersecting_feats(feat, emb, emb2, 0.9)
# %%
# feat, emb = intersecting_feats(feat, emb, emb2, 0.95)
# feat, emb = condense_features(feat, emb, 0.95)
# %%# %%
# save acti, data, target, feat, emb, label
# import pickle
# with open('MNIST_data.pkl', 'wb') as f:
#     pickle.dump((acti, data, target, feat, emb, label), f)

# # load acti, data, target, feat, emb, label
# import pickle
# with open('MNIST_data.pkl', 'rb') as f:
#   acti, data, target, feat, emb, label = pickle.load(f)
# samples = 5000
# print(target.shape)
import torch
def result(net, acti):
  net.eval()
  import torch.nn.functional as F
  with torch.inference_mode():
    output = net.fc2(F.relu(acti))
    output = F.log_softmax(output)
    loss = F.nll_loss(output, target[:acti.shape[0]], size_average=True).item()
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(target[:acti.shape[0]].data.view_as(pred)).sum()
    print(f'Avg. loss: {loss:.4f}, Accuracy: {correct}/{len(pred)} ({correct/len(pred)*100:.0f}%)')
result(net, acti[:samples])
result(net, feat2[:samples] @ emb2)
result(net, acti[:samples] * (torch.rand_like(acti[:samples])))

result(net, (feat2[:samples] * (torch.rand_like(feat2[:samples])) )@ emb2)

# %%
torch.abs((feat2[:samples] @ emb2) - acti[:samples]).mean()
# torch.abs(acti[:samples]).mean()
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 3, figsize=(20, 20))
for i in range(3):
   for j in range(3):
      X = acti[:samples, i]
      Y = acti[:samples, j]
      axs[i, j].scatter(X, Y, c = target[:samples], s = 1)
#        axs[i, j].imshow(emb2[i * 6 + j].reshape(28, 28))
#        axs[i, j].axis('off')
# plt.scatter(X, Y, c = target[:samples], s = 1)
# %%
fig, axs = plt.subplots(3, 3, figsize=(20, 20))
for i in range(3):
   for j in range(3):
      X = feat[:samples, i]
      Y = feat[:samples, j]
      axs[i, j].scatter(X, Y, c = target[:samples], s = 1)
# %%

X = feat[:samples, 9]
Y = feat[:samples, 12]
plt.scatter(X, Y, c = target[:samples], s = 0.1)
# %%

X = acti[:samples, 0]
Y = acti[:samples, 2]
plt.scatter(X, Y, c = target[:samples], s = 0.1)

# %%
# run tsne on the features
import sklearn
from sklearn.manifold import TSNE
def tsne_show(feat):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feat)
    X = tsne_results[:, 0]
    Y = tsne_results[:, 1]
    import matplotlib.pyplot as plt
    plt.scatter(X, Y, c = target[:feat.shape[0]], s = 1)
    plt.show()
# %%
tsne_show(feat[:samples])
tsne_show(acti[:samples])


# %%
import torch
tsne_show(feat[:samples] * (torch.rand_like(feat[:samples]) + 0.5))

# %%

import torch
tsne_show(acti[:samples])
tsne_show(acti[:samples] * torch.rand_like(acti[:samples]) * torch.rand_like(acti[:samples]))

# %%
import torch
tsne_show(feat[:samples])
tsne_show(feat[:samples] * torch.rand_like(feat[:samples]))
tsne_show(feat[:samples] * torch.rand_like(feat[:samples]) * torch.rand_like(feat[:samples]))

# %%
# import r2_score from sklearn
from sklearn.metrics import r2_score
import numpy as np
# pearsonr
from scipy.stats import pearsonr
r2s = []
for j in range(10):
  maxr2 = -1e9
  true_eq = target[:samples] == j
  true_eq = true_eq.float()
  for k in range(feat.shape[1]):
    r2 = pearsonr(true_eq, feat[:samples, k])[0]
    # print(pearsonr(true_eq, acti[:samples, k])[1]) 
    if r2 > maxr2:
        maxr2 = r2
  print(maxr2)
  r2s.append(maxr2)
print(np.mean(r2s))
   
# %%
# get pca of acti
from sklearn.decomposition import PCA
pca = PCA(n_components=acti.shape[1])
pca.fit(acti[:samples])
print(pca.explained_variance_ratio_)
# get pca coordinates
pca_acti = pca.transform(acti[:samples])
print(pca_acti.shape)

# %%
r2s = []
for j in range(10):
  maxr2 = -1e9
  true_eq = target[:samples] == j
  true_eq = true_eq.float()
  for k in range(pca_acti.shape[1]):
    r2 = pearsonr(true_eq, pca_acti[:samples, k])[0]
    # print(pearsonr(true_eq, acti[:samples, k])[1]) 
    if r2 > maxr2:
        maxr2 = r2
  print(maxr2)
  r2s.append(maxr2)
print(np.mean(r2s))
# %%
pca_acti = t.Tensor(pca_acti)
tsne_show(pca_acti[:samples])
tsne_show(pca_acti[:samples] * torch.rand_like(pca_acti[:samples]))
tsne_show(pca_acti[:samples] * torch.rand_like(pca_acti[:samples]) * torch.rand_like(pca_acti[:samples]))

# change pca back to acti
# %%
acti_pca = t.Tensor(pca.inverse_transform(pca_acti))
result(net, acti_pca[:samples])
result(net, t.Tensor(pca.inverse_transform(pca_acti[:samples] * torch.rand_like(pca_acti[:samples]))))
# %%

fig, axs = plt.subplots(3, 3, figsize=(20, 20))
for i in range(3):
   for j in range(3):
      X = pca_acti[:samples, i]
      Y = pca_acti[:samples, j]
      axs[i, j].scatter(X, Y, c = target[:samples], s = 1)
# %%

X = pca_acti[:samples, 0]
Y = pca_acti[:samples, 1]
plt.scatter(X, Y, c = target[:samples], s = 1)
# %%
