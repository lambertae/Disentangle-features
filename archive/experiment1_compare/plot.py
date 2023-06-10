# %%
import matplotlib.pyplot as plt
import pickle
name = ["Autoencoder", "Deep", "GD"]
# read from "exp1_compare.pkl"
with open("exp1_compare.pkl", "rb") as f:
    value_dict = pickle.load(f)
r2_best_list = value_dict["corr_best"]
r2_pref_list = value_dict["corr_pref"]
sparsity_list = value_dict["sparsity"]
mmcs_list = value_dict["mmcs"]
dimrange = [10, 20, 40, 80, 160, 320]
def plot_figure(values, title):
    # save 
    # plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.xscale("log")
    plt.xticks(dimrange, dimrange)
    for i in range(3):
        plt.plot(dimrange, values[i], label = name[i])
    plt.legend()
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plot_figure(r2_best_list, "Average best single correlation")
plt.subplot(2, 2, 2)
plot_figure(r2_pref_list, "Average best prefix correlation")
plt.subplot(2, 2, 3)
plot_figure(sparsity_list, "Sparsity")
plt.subplot(2, 2, 4)
plot_figure(mmcs_list, "MMCS")
plt.savefig("exp1_compare.png")
plt.show()
# %%
