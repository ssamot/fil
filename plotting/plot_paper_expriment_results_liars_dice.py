import json
import matplotlib.pyplot as plt
import sys
import glob
import numpy as np
from scipy import stats

pytorch_files = "results/liars_dice_pytorch*"
lgbm_files = "results/liars_dice_lgb*"

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

    data = [
        {"name": "DeepCFR", "data": pytorch_data, "color" : "blue"},
        {"name": "LGBM", "data": lgbm_data, "color" : "orange"},
    ]

    for d in data:
        check_data(d["data"])

        data_mean = np.mean(d["data"]["y"], axis=0)
        n = len(d["data"]["y"])
        sem = stats.sem(d["data"]["y"], axis=0)  # shape: same as mean
        confidence = 0.95
        ci = sem * stats.t.ppf((1 + confidence) / 2., df=n-1)
        d["mean"] = data_mean
        d["ci"] = ci

    plt.figure(figsize=(6,4.5))

    if use_log_scale:
        plt.yscale("log")

    for d in data:
        plt.plot(d["data"]["x"][0], d["mean"], label=d["name"], color=d["color"])
        plt.fill_between(d["data"]["x"][0], (d["mean"] - d["ci"]), (d["mean"] + d["ci"]), color=d["color"], alpha=.1)

    plt.plot([lgbm_data["x"][0][0], lgbm_data["x"][0][-1]], [1.3101190476190476]*2, label="Random", color="black")

    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Exploitability over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if use_log_scale:
        plt.savefig("results/fig_goofspiel_4_log.pdf", dpi=300)
    else:
        plt.savefig("results/fig_goofspiel_4.pdf", dpi=300)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    load_and_plot(use_log)