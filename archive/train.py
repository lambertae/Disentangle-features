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
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print(device)

# %%

@dataclass
class TrainingConfig:
    batches: int
    batch_size: int


def validate(model: TransformerModel, dataset: ToyDataset):
    inputs, outputs = dataset.sample((1000,))
    padding = t.full(inputs.shape[:-1], dataset.num_feats ** 2).unsqueeze(-1)
    inputs = t.cat([inputs, padding], dim=-1)

    logits = model(inputs)
    guesses = logits.argmax(dim=-1)

    correct = (guesses == outputs)
    # for i in range(correct.shape[0]):
    #     if correct[i] == 0:
    #         print(f"Inputs: {inputs[i]}")
    #         print(f"Sorted outputs: {t.argsort(logits[i], descending=True)[:5]}")
    #         print(f"Correct output: {outputs[i]}")
    return t.count_nonzero(correct) / correct.shape[-1]

def get_activations(model: TransformerModel, dataset: ToyDataset, layer: int, batch_size: int = 5000):
    inputs, outputs = dataset.sample((batch_size,))
    padding = t.full(inputs.shape[:-1], dataset.num_feats ** 2).unsqueeze(-1)
    inputs = t.cat([inputs, padding], dim=-1)

    activations = {}

    def hook_fn(module, input, output) -> None:
        activations[layer] = output

    handle = model.layers[layer].register_forward_hook(hook_fn)
    logits = model(inputs)
    handle.remove()

    return inputs, outputs, activations[layer]

def train(model: TransformerModel, dataset: ToyDataset, config: TrainingConfig):

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(range(config.batches)):
        inputs, outputs = dataset.sample((config.batch_size,))
        padding = t.full(inputs.shape[:-1], dataset.num_feats ** 2).unsqueeze(-1)
        inputs = t.cat([inputs, padding], dim=-1)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, outputs)
        loss.backward()
        optimizer.step()
        
        wandb.log({"Loss": loss.item()})
        # if batch % 1 == 0:
        #     print(f"Loss: {loss.item()}")
        if batch % 100 == 0:
            valid_accuracy = validate(model, dataset)
            wandb.log({"Validation accuracy": valid_accuracy})
            # print(f"Validation accuracy: {valid_accuracy}")

# %%

if __name__ == "__main__":
    dataset_config = DataConfig(
        num_feats=10,
        num_topics=2,
        topics_per_word=4,
        num_words=5
    )
    model_config = ModelConfig(
        num_layers=4,
        num_heads=3,
        embedding_dim=6,
        num_tokens=dataset_config.num_feats ** 2 + 1,
        seq_len=dataset_config.num_words*dataset_config.num_topics+1
    )
    train_config = TrainingConfig(
        batches=5000,
        batch_size=100
    )

    dataset = ToyDataset(dataset_config)
    model = TransformerModel(model_config,)

    wandb.init(name="toy-transformer", project="superposition")
    train(model, dataset, train_config)

# %%

inputs, outputs, acts = get_activations(model, dataset, layer=2)

np.save("inputs.npy", inputs.detach().numpy())
np.save("outputs.npy", outputs.detach().numpy())
np.save("acts.npy", acts.detach().numpy())
acts = acts[:,-1,:]
feats1, embeds1, info = GD_solver(acts, steps=2000, lr=1e-2, init_lamb=0.25, guess_factor=16)
feats1, embeds1 = condense_features(feats1, embeds1)

# %%
def metric_total_acti(feats1):
    import numpy as np
    srt_feats = abs(feats1).detach().numpy()
    sorted = np.flip(np.sort(srt_feats, axis = 1), axis = 1)
    pltarray = np.mean(sorted, axis = 0)
    return pltarray.sum() / pltarray[0]
print(feats1.shape, embeds1.shape, metric_total_acti(feats1))

# %%
for j in range(1):
    feats2, embeds2, info = GD_solver(acts, steps=2000, lr=1e-2, init_lamb=0.25, guess_factor=16)
    feats2, embeds2 = condense_features(feats2, embeds2)

# %%
# save the features

np.save("feats1.npy", feats1.detach().numpy())
np.save("feats2.npy", feats2.detach().numpy())

np.save("embeds1.npy", embeds1.detach().numpy())
np.save("embeds2.npy", embeds2.detach().numpy())

feats3, embeds3 = intersecting_feats(feats1, embeds1, embeds2, 0.9)
print(feats3.shape, embeds3.shape, metric_total_acti(feats3))

# save output

# %%
# # load
feats1 = t.from_numpy(np.load("feats1.npy"))
# feats2 = t.from_numpy(np.load("feats2.npy"))

embeds1 = t.from_numpy(np.load("embeds1.npy"))
# embeds2 = t.from_numpy(np.load("embeds2.npy"))

# inputs = t.from_numpy(np.load("inputs.npy"))
# outputs = t.from_numpy(np.load("outputs.npy"))


# %%
print(metric_total_acti(feats1), metric_total_acti(feats2), metric_total_acti(feats3))
# %%
int_feats = feats3.clone()
import matplotlib.pyplot as plt
def illustrate(feature_id, threshold):
    print(f"Feature {feature_id}:")
    indices = t.argsort(int_feats[:, feature_id], descending=True)
    for j in range(5):
        print(f"id: {indices[j]} {inputs[indices[j]].detach().numpy()} with activation {int_feats[indices[j], feature_id]:.2f}, output {outputs[indices[j]].detach().numpy()}")
    feats_count = [0] * 10
    for j in range(len(indices)):
       id = indices[j]
       if int_feats[id, feature_id] < threshold:
         break
       res = outputs[id].detach().numpy()
       feats_count[res % 10] += 1
       feats_count[res // 10] += 1
    # show histogram in a separate window
    plt.figure()
    plt.bar(range(10), feats_count)
    plt.show()

# for i in range(int_feats.shape[-1]):
    # illustrate(i, 1.0)
    # print top 10
    # top10 = indices[:, i][:10]
    # print(indices[:10])
    # max_activating = t.gather(inputs, 0, indices[:10])
    # print(max_activating)\
# %%
import random
def show_sample(sample):
    print(f"Sample: {inputs[sample]} Output: {outputs[sample]}")
    ids = t.argsort(int_feats[sample], descending=True)
    for j in range(4):
        print(f"Feature {ids[j]}: {int_feats[sample, ids[j]]}")
        illustrate(ids[j], int_feats[sample, ids[j]])
# for i in range(1):
#     sample = random.randint(0, int_feats.shape[0] - 1)
   

# %%
show_sample(random.randint(0, int_feats.shape[0] - 1))
# %%
