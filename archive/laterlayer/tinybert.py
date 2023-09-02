# %%
import sys
sys.path.append('/scratch/dengm/distent/Disentangle-features/archive')
from features import *
from metrics import *

import os
os.environ["XDG_CACHE_HOME"] = "/scratch/dengm/distent/pip_cache"
os.environ['PIP_CACHE_DIR'] = "/scratch/dengm/distent/pip_cache"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import xml.etree.ElementTree as ET
import random
import nltk
import gzip

first_time = False
if first_time:
    def get_sentences():
        # Download the Wikipedia abstracts file
        url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract1.xml.gz'
        filename = 'enwiki-latest-abstract1.xml.gz'
        urllib.request.urlretrieve(url, filename)
        sentences = []
        nltk.download('punkt')
        with gzip.open(filename, 'rb') as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for elem in root:
                text = elem.find('abstract').text
                if text is not None:
                    for sentence in nltk.sent_tokenize(text):
                        sentences.append(sentence)

        sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences if sentence]

        with open('sentences.txt', 'w') as f:
            for sentence in sentences:
                f.write(' '.join(sentence) + '\n')
        print(len(sentences))
    get_sentences()

with open('sentences.txt', 'r') as f:
    sentences = f.read().splitlines()
from transformers import AutoModel, AutoTokenizer
# for model_name in ['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium']:
for model_name in ['bert-mini', 'bert-small', 'bert-medium']:
    model = AutoModel.from_pretrained(f"prajjwal1/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"prajjwal1/{model_name}")

    def bert_encode(texts, tokenizer, max_len=512):
        all_tokens = []
        all_masks = []
        all_segments = []
        seqs = []
        for text in texts:
            text = tokenizer.tokenize(text)
            
            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]   
            seqs.append(input_sequence)
            pad_len = max_len-len(input_sequence)
            tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            all_tokens.append(tokens)
        return np.array(all_tokens), seqs
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    dataloader = DataLoader(sentences, batch_size=512, shuffle=True)
        
    def get_acti(model, inputs, layer_id):
        # print(inputs.shape)
        
        with torch.inference_mode():
            # model.encoder.layer[layer_id].attention.self._forward_hooks.clear()
            # remove all hooks
            for lid in range(len(model.encoder.layer)):
                model.encoder.layer[lid].attention.self._forward_hooks.clear()

            activations = []
            # define a hook function to get the output of the layer
            def hook(module, input, output):
                activations.append(output[0])

            # register the hook to the layer of interest
            layer = model.encoder.layer[layer_id]
            layer.attention.self.register_forward_hook(hook)
            model(inputs)
            activation = activations[0][:, :, :]

            return activation

        
    import torch
    import torch.nn as nn
    import numpy as np
    from tqdm import trange
    from dataclasses import dataclass
    # activations: (S, E)
    # embs: (F, E)
    # lamb: float
    # feats: (S, F)

    lamb = 0.1

    def greedy_feats(dataset, embs, lamb):
        with torch.inference_mode():
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
            # free ids, remainder
            del ids, remainder
            # feats = feats / norms[None, :]
            return feats

    def construction_loss(dataset, feat, embs):
        return (feat @ (embs / embs.norm(dim = 1, p = 2)[:, None]) - dataset).norm(p = 2) ** 2
    def total_loss(dataset, feat, embs, lamb):
        return lamb * feat.norm(p = 1).sum() + construction_loss(dataset, feat, embs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sizes = {'bert-tiny': 128, 'bert-mini': 256, 'bert-small': 512, 'bert-medium': 512}
    layers = {'bert-tiny': 2, 'bert-mini': 4, 'bert-small': 4, 'bert-medium': 8}
    for layer_id in range(layers[model_name]):
        if layer_id == 0 and model_name == "bert-tiny":
            continue
        print(f"working {layer_id}")
        
            
        solver_lamb = 0.1
        hidden_size = sizes[model_name]
        dict_size = hidden_size * 16

        embs = torch.randn((dict_size, hidden_size)).to(device)

        class Solver(nn.Module):
            def __init__(self, embs):
                super().__init__()
                self.embs = nn.Parameter(embs.clone().detach().to(embs.device))
            def loss(self, dataset, feats):
                return construction_loss(dataset, feats, self.embs)
        solver = Solver(embs)

        optimizer = torch.optim.Adam(solver.parameters(), lr=1e-3)
        epochs = 2
        gd_steps = 20
        import tqdm

        for i in range(epochs):  # Fixed outer tqdm usage
            rec_loss = 0
            loss_total = 0
            sparsity = 0
            L_inf = 0
            L_inf_list = []
            with tqdm.tqdm(dataloader, desc="Processing batches") as tr:  # Fixed inner tqdm usage and added description
                for round, bat in enumerate(tr):
                    # bat = bat[:1024]
                    # print(round, bat)
                    
                    batch, seq = bert_encode(bat, tokenizer, max_len=160)
                    nonzero = (batch != 0).sum(axis=1)
                    batch = batch[:, :np.max(nonzero)]
                    if round % 2 == 0:
                        torch.cuda.empty_cache()
                    
                    acti = get_acti(model, torch.tensor(batch).to(device), layer_id)
                    acti = torch.concat([acti[i, :nonzero[i], :] for i in range(acti.shape[0])], dim=0)
                    
                    feats = greedy_feats(acti, solver.embs, solver_lamb).clone().requires_grad_(True)
                    for i in range(gd_steps):
                        loss = solver.loss(acti, feats)
                        # loss.backward(retain_graph=True)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    with torch.inference_mode():
                        loss_total = total_loss(acti, feats, solver.embs, solver_lamb).item() / acti.shape[0]
                        rec_loss = construction_loss(acti, feats, solver.embs).item() / acti.shape[0]
                        sparsity = (feats).abs().sum() / feats.abs().max(dim=1)[0].sum()
                        L_inf = (feats).abs().max(dim=1)[0].mean()
                        L_inf_list.append(L_inf.item())
                    
                    del acti, feats

                    # Moved the condition inside the for-loop
                    if round % 10 == 0:
                        cur_rounds = 1 #round + 1
                        tr.set_postfix(loss=loss_total / cur_rounds,
                                    rec_loss=rec_loss / cur_rounds,
                                    sparsity=sparsity / cur_rounds,
                                    L_inf=L_inf / cur_rounds,
                                    loss_est=loss_total / (L_inf * solver_lamb))
            L_inf_list = np.array(L_inf_list)
            # solver_lamb = np.mean(L_inf_list) * 0.1
            ckpt_emb = solver.embs.clone().detach()
            ckpt_emb /= torch.norm(ckpt_emb, dim=1, p=2)[:, None]
            torch.save(ckpt_emb, f"ckpt/{model_name}_layer{layer_id}.pt")

        with torch.inference_mode():
            solver.eval()
            solver.embs /= torch.norm(solver.embs, dim=1, p=2)[:, None]

        torch.save(solver.embs,  f"ckpt/{model_name}_layer{layer_id}.pt")

# 3398807
# %%
