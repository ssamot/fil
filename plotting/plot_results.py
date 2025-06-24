import json
import matplotlib.pyplot as plt
import sys

def load_and_plot(files, use_log_scale=False):
    for file in files:
        with open(file) as f:
            data = json.load(f)
            x = sorted([int(k) for k in data.keys()])
            y = [data[str(i)] for i in x]
            plt.plot(x, y, label=file)

    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Exploitability over Iterations")
    if use_log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if "--log" in args:
        use_log = True
        args.remove("--log")

    if not args:
        print("Usage: python plot_jsons.py [--log] file1.json file2.json ...")
        sys.exit(1)

    json_files = args
    load_and_plot(json_files, use_log)