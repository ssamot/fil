import sys
from plot_paper_expriment_general import load_data, load_and_plot
from plot_paper_experiment_config import COLOR_PYTORCH, COLOR_LGBM
from plot_paper_experiment_config import TEXT_PYTORCH, TEXT_LGBM

pytorch_files = "results/liars_dice_pytorch*"
lgbm_files = "results/liars_dice_lgb*"

def load(use_log_scale):
    pytorch_data = load_data(pytorch_files)
    lgbm_data = load_data(lgbm_files)

    data = [
        {"name": TEXT_PYTORCH, "data": pytorch_data, "color" : COLOR_PYTORCH},
        {"name": TEXT_LGBM, "data": lgbm_data, "color" : COLOR_LGBM},
    ]

    load_and_plot(data, 1.3101190476190476, "liars_dice", use_log_scale)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    load(use_log)