import json
import matplotlib.pyplot as plt
import sys
import glob
import numpy as np
from scipy import stats

pytorch_files = "results/kuhn_pytorch*"
lgbm_files = "results/kuhn_lgb*"

def load_data(file_start):
    data = {"x": [], "y": []}
    for file_name in glob.glob(file_start):
        with open(file_name) as f:
            d = json.load(f)
            x = sorted([int(k) for k in d.keys()])
            y = [d[str(i)] for i in x]
            data["x"].append(x)
            data["y"].append(y)
    return data

def check_data(data):
    prev_x = None
    for x, y in zip(data["x"], data["y"]):
        if len(x) != len(y):
            raise Exception("Values loaded incorectly, len(X) != len(y)")
        if prev_x is not None:
            if prev_x != x:
                raise Exception("Different files have different iterations")
        

def load_and_plot(use_log_scale):
    pytorch_data = load_data(pytorch_files)
    lgbm_data = load_data(lgbm_files)

    check_data(pytorch_data)
    check_data(lgbm_data)

    pytorch_mean = np.mean(pytorch_data["y"], axis=0)
    n = len(pytorch_data["y"])
    sem = stats.sem(pytorch_data["y"], axis=0)  # shape: same as mean
    confidence = 0.95
    pytorch_ci = sem * stats.t.ppf((1 + confidence) / 2., df=n-1)

    lgbm_mean = np.mean(lgbm_data["y"], axis=0)
    n = len(lgbm_data["y"])
    sem = stats.sem(lgbm_data["y"], axis=0)  # shape: same as mean
    confidence = 0.95
    lgbm_ci = sem * stats.t.ppf((1 + confidence) / 2., df=n-1)

    plt.figure(figsize=(6,4.5))

    if use_log_scale:
        plt.yscale("log")
    
    plt.plot(pytorch_data["x"][0], pytorch_mean, label="DeepCFRPytorch", color='b')
    plt.fill_between(pytorch_data["x"][0], (pytorch_mean - pytorch_ci), (pytorch_mean + pytorch_ci), color='b', alpha=.1)
    
    plt.plot(lgbm_data["x"][0], lgbm_mean, label="DeepCFRLGBM", color='r')
    plt.fill_between(lgbm_data["x"][0], (lgbm_mean - lgbm_ci), (lgbm_mean + lgbm_ci), color='r', alpha=.1)

    plt.plot([lgbm_data["x"][0][0], lgbm_data["x"][0][-1]], [0.9166666666666666]*2, label="Random", color="black")

    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if use_log_scale:
        plt.savefig("results/fig_kuhn_log.pdf", dpi=300)
    else:
        plt.savefig("results/fig_kuhn.pdf", dpi=300)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    load_and_plot(use_log)