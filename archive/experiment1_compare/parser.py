# parse outputs with formate
# Dimension:80, type:0
# Average correlation:0.36130141966516544
# Sparsity:22.03834647184605, actual:5.024569988250732
# MMCS:0.8006702579971481
# Executing with num_features: 640 and feature_dim: 80 and reg_param: 0.35
# %%
r2list = [[] for i in range(3)]
sparsity_list = [[] for i in range(3)]
mmcs_list = [[] for i in range(3)]
dimrange = [20, 40, 80, 160, 320]
name = ["Autoencoder", "Deep", "GD"]
with open("nohup.out", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("Dimension:"):
            dim = int(line.split(",")[0].split(":")[1])
            tp = int(line.split(",")[1].split(":")[1])
            corr = float(lines[lines.index(line) + 1].split(":")[1])
            sparsity = float(lines[lines.index(line) + 2].split(":")[1].split(",")[0])
            mmcs_val = float(lines[lines.index(line) + 3].split(":")[1])
            print(corr, sparsity, mmcs_val)
            r2list[tp].append(corr)
            sparsity_list[tp].append(sparsity)
            mmcs_list[tp].append(mmcs_val)
import matplotlib.pyplot as plt
def plot_figure(values, title):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.xscale("log")
    plt.xticks(dimrange, dimrange)
    for i in range(3):
        plt.plot(dimrange, values[i], label = name[i])
    plt.legend()
    plt.savefig(f"exp1_{title}.png")
    plt.show()
plot_figure(r2list, "Average correlation")
plot_figure(sparsity_list, "Sparsity")
plot_figure(mmcs_list, "MMCS")
# %%
