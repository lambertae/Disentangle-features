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

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,6,7"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model_list = ['roneneldan/TinyStories-1M', 'roneneldan/TinyStories-3M', 'roneneldan/TinyStories-33M', 'EleutherAI/gpt-neo-125M']#, 'EleutherAI/gpt-neo-1.3B']


bert_list = ['prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-small', 'prajjwal1/bert-medium']
bert_list = bert_list + model_list
last_lamb = 0.6
for name in bert_list:
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    np.random.seed(0)
    suffix = name.split('/')[-1]
    embs = np.load(f"{suffix}_embs.npy")
    words = np.load(f"{suffix}_words.npy")
    fold, eold, infold = load_wrapper('gd', f'{suffix}_task')
    f1, e1, inf1 = wrapper('gd', f'cont_{suffix}_task', t.Tensor(embs).to(device), init_lamb = infold['lamb'], lamb_left = 0.4, lamb_right = 0.6, guess_factor=8, lr = 1e-3, steps = 16000, init_feat = fold, init_emb = eold)