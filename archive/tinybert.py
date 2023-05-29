# %%
from transformers import AutoModel

model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
emb = model.embeddings.word_embeddings.weight.detach().numpy()
print(emb.shape)
from transformers import AutoTokenizer
import numpy as np
import torch as t

os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# get 5000 elts without replacement
# def get_acti(samples, emb, tokenizer):
#     idList = np.random.choice(emb.shape[0], samples, replace = False)
#     words = tokenizer.convert_ids_to_tokens(idList)
#     embs = emb[idList]
#     return embs, words

# embs, words = get_acti(emb.shape[0], emb, tokenizer)
# try to load from 
embs = np.load(f"bert/bert_tiny_embs.npy")
words = np.load(f"bert/bert_tiny_words.npy")
np.save(f"bert/bert_tiny_embs.npy", embs)
np.save(f"bert/bert_tiny_words.npy", words)
# %%
import torch as t
import numpy as np
import torch.nn.functional as F
import json
def get_feat_top(feat, id, num):
    result = []
    result_acti = []
    sorted = feat[:, id].argsort(descending = True)
    for j in range(num):
        result.append(words[sorted[j]])
        result_acti.append(feat[sorted[j], id].item())
    return result, result_acti
def get_feat_emb_info(name):
    feat = np.load(f"bert/{name}_feat.npy")
    emb = np.load(f"bert/{name}_emb.npy")
    with open("bert/info.json", "r") as f:
        info_dict = json.load(f)
    info = info_dict[name]
    return t.Tensor(feat).to(device), t.Tensor(emb).to(device), info
# %%
f1, e1, inf1 = get_feat_emb_info("f1")
f2, e2, inf2 = get_feat_emb_info("f2")
ftot, etot, info_tot = get_feat_emb_info("ftot")

# %%
for i in range(0, 4000, 4):
    print(get_feat_top(ftot, i, 20)[0])

# %%

sum_feat = ftot.sum(dim = 0)
best_dims = sum_feat.argsort(descending = True)
# get_feat_top(ftot, best_dims[random.randint(0, 400)], 20)
for i in range(0, 50, 1):
#i = 12
    print(i, sum_feat[best_dims[i]].item())
    print(get_feat_top(ftot, best_dims[i], (ftot[:, best_dims[i]] > 0.05).sum())[0])
    print()

# %%
import random
def illustrate_example(feat, word_id):
    print(words[word_id])
    acts = feat[word_id].argsort(descending = True)
    for j in range(20):
        total_act = feat[word_id].sum()
        if (feat[word_id, acts[j]] * 20 < total_act):
            break
        print(acts[j].item(), feat[word_id, acts[j]].item(), feat[:, acts[j]].sum().item())
        print(get_feat_top(feat, acts[j], 20)[0])

illustrate_example(ftot, random.randint(0, ftot.shape[0]))
# find PCA of embs
# %% Compared to PCA and rand
from sklearn.decomposition import PCA
pca = PCA(n_components=embs.shape[1])
pca.fit(embs)
# print(pca.explained_variance_ratio_)
new_coords = pca.transform(embs)
for i in range(0, 100, 1):
    print(get_feat_top(t.Tensor(new_coords).to(device), i, 20)[0])

# %%

for i in range(0, 128, 1):
    print(get_feat_top(t.Tensor(embs).to(device), i, 20)[0])
# %%
rand_dir = t.randn(embs.shape[1], embs.shape[1]).to(device)
rand_dir = rand_dir / rand_dir.norm(dim = 0)
rand_coords = t.matmul(t.Tensor(embs).to(device), rand_dir)

for i in range(0, 128, 1):
    print(get_feat_top(t.Tensor(rand_coords).to(device), i, 20)[0])



# %% Run & store f1, f2, ftot
from features import *
import os
import torch as t
from features import metric_total_acti
f1, e1, inf1 = GD_solver(t.Tensor(embs).to(device)[:10000], init_lamb = 0.15, lamb_left = 0.4, lamb_right = 0.6, guess_factor=32, lr = 5e-3, steps = 8000)
print(inf1, metric_total_acti(f1))
print(get_feat_top(f1, 0, 10))
f2, e2, inf2 = GD_solver(t.Tensor(embs).to(device)[:10000], init_lamb = inf1["lamb"], lamb_left = 0.4, lamb_right = 0.6, guess_factor=32, lr = 5e-3, steps = 8000)
print(inf2, metric_total_acti(f2))
print(get_feat_top(f2, 0, 10))
import random
from features import intersecting_feats, condense_features
# f1dense = condense_features(f1, e1)
# print(f1dense.shape)
fint, eint = intersecting_feats(f1, e1, e2, 0.95)
print(fint.shape, eint.shape)
fint_dense, eint_dense = condense_features(fint, eint)
print(fint_dense.shape)
sum_feat = f1.sum(dim = 0)
best_dims = sum_feat.argsort(descending = True)
get_feat_top(f1, best_dims[random.randint(0, 400)], 15)
# save f1, e1, inf1, f2, e2, inf2
import json 
def save(feat, emb, info, name):
    
    info_dict = dict()
    # try to load
    if os.path.exists("bert/info.json"):
        with open("bert/info.json", "r") as f:
            info_dict = json.load(f)
    np.save(f"bert/{name}_feat.npy", feat.cpu().detach().numpy())
    np.save(f"bert/{name}_emb.npy", emb.cpu().detach().numpy())
    info_dict[name] = str(info)
    print("!!!", info)
    with open("bert/info.json", "w") as f:
        json.dump(info_dict, f)
inf1["steps"] = 8000
inf2["steps"] = 8000
inf1["lr"] = 5e-3
inf2["lr"] = 5e-3
inf1["guess_factor"] = 32
inf2["guess_factor"] = 32
inf1["size"] = 10000
inf2["size"] = 10000
save(f1, e1, inf1, "f1")
save(f2, e2, inf2, "f2")

np.save(f"bert/bert_tiny_embs.npy", embs)
np.save(f"bert/bert_tiny_words.npy", words)
ftot, etot, inftot = GD_solver(t.Tensor(embs).to(device), init_lamb = 0.15, lamb_left = 0.4, lamb_right = 0.6, guess_factor=32, lr = 5e-3, steps = 8000)
save(ftot, etot, inftot, "ftot")

# %%

# %%
# %%
