import sys
from plot_paper_expriment_general import load_data, load_and_plot

pytorch_files = "results/goofspiel_4_pytorch*"
lgbm_files = "results/goofspiel_4_lgb*"

def load(use_log_scale):
    pytorch_data = load_data(pytorch_files)
    lgbm_data = load_data(lgbm_files)

    data = [
        {"name": "DeepCFR", "data": pytorch_data, "color" : "blue"},
        {"name": "LGBM", "data": lgbm_data, "color" : "orange"},
    ]

    load_and_plot(data, 1.4166666666666665, "kuhn", use_log_scale)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    load(use_log)