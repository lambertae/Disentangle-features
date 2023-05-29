# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import xml.etree.ElementTree as ET
import random
import nltk
import gzip

# %%




# Extract a random subset of sentences

# %%
# num_sentences = 10000
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
                # print(len(text))
                # break

                for sentence in nltk.sent_tokenize(text):
                    sentences.append(sentence)


    # Preprocess the sentences (e.g., remove punctuation, lowercase everything)
    sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences if sentence]

    # Save the sentences to a file
    with open('sentences.txt', 'w') as f:
        for sentence in sentences:
            f.write(' '.join(sentence) + '\n')
    print(len(sentences))

# %%
with open('sentences.txt', 'r') as f:
    sentences = f.read().splitlines()
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# randomly select 10000 sentences
random.seed(0)
num_sentences = 20000
sentences = random.sample(sentences, num_sentences)

# %%

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

import torch 
import os 

# dir = "/data/scratch/dengm/distent/Disentangle-features/bert/bert-classification"
# train_df = pd.read_csv(os.path.join(dir, "train.csv"), encoding='ISO-8859-1', low_memory=False)

# print(train_df.head())
tokens, seqs = bert_encode(sentences, tokenizer, max_len=160)


# %%

layer_id = 0
pos_id = 10
def get_acti(model, inputs, layer_id, pos_id):
    # clear hooks
    print(inputs.shape)
    model.encoder.layer[layer_id].attention.self._forward_hooks.clear()

    activations = []

    # define a hook function to get the output of the layer
    def hook(module, input, output):
        activations.append(output[0])

    # register the hook to the layer of interest
    layer = model.encoder.layer[layer_id]
    layer.attention.self.register_forward_hook(hook)

    # pass the inputs through the model
    with torch.no_grad():
        model(inputs)
    print(activations[0].shape)

    # extract the activation at the specified position
    activation = activations[0][:, pos_id, :]

    return activation

# %%
dir = "/data/scratch/dengm/distent/Disentangle-features/bert/bert-classification"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 4, 6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from features import *
acti = None
def save_acti():
    global acti
    acti = get_acti(model, torch.tensor(tokens), layer_id, pos_id)
    np.save(os.path.join(dir, f"bert_layer{layer_id}_pos{pos_id}"), acti.cpu().numpy())
def load_acti():
    global acti
    acti = torch.tensor(np.load(os.path.join(dir, f"bert_layer{layer_id}_pos{pos_id}.npy")))
    print(acti.shape)
load_acti()
# %%
feat, emb, info = None, None, None
def save_feat():
    global feat, emb, info
    feat, emb, info = GD_solver(acti[:10000].to(device), 16, init_lamb = 0.45)
    from features import metric_total_acti
    print(info, metric_total_acti(feat))
    info["layer"] = layer_id
    info["pos"] = pos_id
    def save(feat, emb, info, name):
        import json
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
    save(feat, emb, info, f"Layer{layer_id}_{pos_id}")
def load_feat():
    global feat, emb, info
    import json
    feat = torch.tensor(np.load(f"bert/Layer{layer_id}_{pos_id}_feat.npy"))
    emb = torch.tensor(np.load(f"bert/Layer{layer_id}_{pos_id}_emb.npy"))
    with open("bert/info.json", "r") as f:
        info = json.load(f)[f"Layer{layer_id}_{pos_id}"]
    print(info)
load_feat()

# %%
# save sentences to dir/"sampled_sentences.txt"
def save_sentences():
    with open(dir+"/sampled_sentences.txt", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
def load_sentences():
    global sentences
    with open(dir+"/sampled_sentences.txt", "r") as f:
        sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]
load_sentences()

# %%
def get_feat_top(feat, id, num):
    result = []
    result_acti = []
    sorted = feat[:, id].argsort(descending = True)
    for j in range(num):
        result.append(seqs[sorted[j]])
        result_acti.append(feat[sorted[j], id].item())
    return result, result_acti

sum_feat = feat.sum(dim = 0)
best_dims = sum_feat.argsort(descending = True)
for i in range(0, len(best_dims)):
    print(sum_feat[best_dims[i]])

# %%
def print_seq(seq):
    display_seq = seq
    if (pos_id < len(seq)):
        # add * to the current token
        display_seq = seq[:pos_id] + ["*", "*", seq[pos_id], "*", "*"] + seq[pos_id+1:]
    else:
        display_seq = seq + ["*"]
    result = tokenizer.decode(tokenizer.convert_tokens_to_ids(display_seq))
    # change "* *" to "***"
    result = result.replace("* *", "***")
    print(result)
for i in range(0, 2000, 5):
    print(f"Maximum activated sentences for dim {best_dims[i].item()}")
    sequences = get_feat_top(feat, best_dims[i], 10)[0]
    for seq in sequences:
        current_token = seq[pos_id] if pos_id < len(seq) else "None"
        print_seq(seq)
    print()
# %%
for i in range(0, 100):
    # print(f"Maximum activated sentences for dim {best_dims[i].item()}")
    sequences = get_feat_top(acti[:10000], i, 10)[0]
    for seq in sequences:
        current_token = seq[pos_id] if pos_id < len(seq) else "None"
        print_seq(seq)
    print()
# %%

import random
def illustrate_example(feat, word_id):
    print_seq(seqs[word_id])
    acts = feat[word_id].argsort(descending = True)
    for j in range(20):
        total_act = feat[word_id].sum()
        if (feat[word_id, acts[j]] * 20 < total_act):
            break
        print(acts[j].item(), feat[word_id, acts[j]].item(), feat[:, acts[j]].sum().item())
        for seq in get_feat_top(feat, acts[j], 10)[0]:
            print_seq(seq)

illustrate_example(feat, random.randint(0, feat.shape[0]))
# %%
