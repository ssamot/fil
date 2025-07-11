import sys
from plot_paper_expriment_general import load_data, load_and_plot

pytorch_files = "results/leduc_pytorch*"
lgbm_files = "results/leduc_lgb*"
pt_splines_files = "results/leduc_pt_splines*"
pt_dt_files = "results/leduc_pt_dt*"

def load(use_log_scale):
    pytorch_data = load_data(pytorch_files)
    lgbm_data = load_data(lgbm_files)
    pt_splines_data = load_data(pt_splines_files)
    pt_dt_data = load_data(pt_dt_files)

    data = [
        {"name": "DeepCFR", "data": pytorch_data, "color" : "blue"},
        {"name": "LGBM", "data": lgbm_data, "color" : "orange"},
        {"name": "Public table splines", "data": pt_splines_data, "color" : "green"},
        {"name": "Public table decision tree", "data": pt_dt_data, "color" : "red"}
    ]

    load_and_plot(data, 4.747222222222222, "leduc", use_log_scale)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    load(use_log)