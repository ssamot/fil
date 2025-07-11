import sys
from plot_paper_expriment_general import load_data, load_and_plot
from plot_paper_experiment_config import COLOR_PYTORCH, COLOR_SPLINES, COLOR_DT
from plot_paper_experiment_config import TEXT_PYTORCH, TEXT_SPLINES, TEXT_DT

pytorch_files = "results/leduc_ignore_jack_queen_pytorch*"
pt_splines_files = "results/leduc_ignore_jack_queen_pt_splines*"
pt_dt_files = "results/leduc_ignore_jack_queen_pt_dt*"

def load(use_log_scale):
    pytorch_data = load_data(pytorch_files)
    pt_splines_data = load_data(pt_splines_files)
    pt_dt_data = load_data(pt_dt_files)

    data = [
        {"name": TEXT_PYTORCH, "data": pytorch_data, "color" : COLOR_PYTORCH},
        {"name": TEXT_SPLINES, "data": pt_splines_data, "color" : COLOR_SPLINES},
        {"name": TEXT_DT, "data": pt_dt_data, "color" : COLOR_DT}
    ]

    load_and_plot(data, 4.747222222222222, "leduc_ignore_queens", use_log_scale)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    load(use_log)