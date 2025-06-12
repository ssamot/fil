import json
import matplotlib.pyplot as plt
import sys

def load_and_plot(files):
    for file in files:
        with open(file) as f:
            data = json.load(f)
            # Convert string keys to integers, sort them
            x = sorted([int(k) for k in data.keys()])
            y = [data[str(i)] for i in x]
            plt.plot(x, y, label=file)

    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Exploitability over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_jsons.py file1.json file2.json ...")
        sys.exit(1)

    json_files = sys.argv[1:]
    load_and_plot(json_files)