from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# export XDG_CACHE_HOME=/scratch/dengm/stable-diffusion-webui/pip_cache
import os
os.environ["XDG_CACHE_HOME"] = "/scratch/dengm/distent/pip_cache"
os.environ['PIP_CACHE_DIR'] = "/scratch/dengm/distent/pip_cache"
import sys
sys.path.append('/scratch/dengm/distent/Disentangle-features/archive')
from features import *
from metrics import *
import os
import torch as t
import torch.nn as nn
import pickle
import numpy as np
import torch as t

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model_list = ['roneneldan/TinyStories-1M', 'roneneldan/TinyStories-3M', 'roneneldan/TinyStories-33M', 'EleutherAI/gpt-neo-125M']#, 'EleutherAI/gpt-neo-1.3B']


bert_list = ['prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-small', 'prajjwal1/bert-medium']

for name in ['gpt', 'EleutherAI/gpt-neo-1.3B']:
    suffix = name.split('/')[-1]
    mbs = np.load(f"{suffix}_embs.npy")
    words = np.load(f"{suffix}_words.npy")
    for lam_range in [0.1, 0.2]:
        f1, e1, inf1 = wrapper('svd', f'lamb{lam_range}_{suffix}_task', t.Tensor(mbs).to(device), init_lamb = lam_range, lamb_left = lam_range - 0.02, lamb_right = lam_range + 0.02, guess_factor=8, lr = 3e-3, steps = 2000, greedy_step = 20)
        print(inf1)
exit(0)

last_lamb = 0.6
load = True
for name in model_list:
    suffix = name.split('/')[-1]
    if not load:
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(0)
        model = AutoModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        def get_acti(samples, emb, tokenizer):
            idList = np.random.choice(emb.shape[0], samples, replace = False)
            words = tokenizer.convert_ids_to_tokens(idList)
            embs = emb[idList]
            return embs, words
        emb = model.embeddings.word_embeddings.weight.detach().numpy()
        embs, words = get_acti(emb.shape[0], emb, tokenizer)
        np.save(f"{suffix}_embs.npy", embs)
        np.save(f"{suffix}_words.npy", words)
    else:
        embs = np.load(f"{suffix}_embs.npy")
        words = np.load(f"{suffix}_words.npy")
    
    # print(embs.shape)
    f1, e1, inf1 = wrapper('svd', f'{suffix}_task', t.Tensor(embs).to(device), init_lamb = last_lamb, lamb_left = 0.08, lamb_right = 0.12, guess_factor=8, lr = 3e-3, steps = 2000, greedy_step = 20)
    last_lamb = inf1['lamb']
    print(inf1)
# export TRANSFORMERS_CACHE=/scratch/dengm/distent/pip_cache

for name in model_list + bert_list:
    suffix = name.split('/')[-1]
    if not load:
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(0)
        model = AutoModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        def get_acti(samples, emb, tokenizer):
            idList = np.random.choice(emb.shape[0], samples, replace = False)
            words = tokenizer.convert_ids_to_tokens(idList)
            embs = emb[idList]
            return embs, words
        emb = model.embeddings.word_embeddings.weight.detach().numpy()
        embs, words = get_acti(emb.shape[0], emb, tokenizer)
        np.save(f"{suffix}_embs.npy", embs)
        np.save(f"{suffix}_words.npy", words)
    else:
        embs = np.load(f"{suffix}_embs.npy")
        words = np.load(f"{suffix}_words.npy")
    
    # np.save(f"{suffix}_words.npy", words)
    # print(embs.shape)
    f1, e1, inf1 = wrapper('svd', f'lamb0.2_{suffix}_task', t.Tensor(embs).to(device), init_lamb = last_lamb, lamb_left = 0.18, lamb_right = 0.22, guess_factor=8, lr = 3e-3, steps = 2000, greedy_step = 20)
    last_lamb = inf1['lamb']
    print(inf1)

# 3929780