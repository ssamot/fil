from absl import app
from absl import flags
from absl import logging

import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability
from our_cfr import batched_external_sampling_mccfr

import tqdm
import json

def main(unused_argv):
    np.random.seed(0)
    game = pyspiel.load_game("leduc_poker")
    es_solver = batched_external_sampling_mccfr.ExternalSamplingSolver(game)
    batched_iterations = 19100
    batch_size = 2
    tree_traversals = 0
    results = {}
    with tqdm.tqdm(range(batched_iterations)) as pbar:
        for i in range(batched_iterations):
            if i == 5000:
                batch_size = 4
            if i == 9900:
                batch_size = 8
            if i == 13500:
                batch_size = 16
            es_solver.batched_iteration(batch_size)
            if(tree_traversals%100 == 0):
                conv = exploitability.nash_conv(game, es_solver.average_policy())
                results[tree_traversals] = conv
            pbar.update(1)

            pbar.set_description(f"Iteration {batch_size}")
            pbar.set_postfix({"exploitability": f"{conv:.6f}"})
            tree_traversals += batch_size

    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))
    results[tree_traversals] = conv

    results_file = "./results/mccfr_leduc_batched.json"
    with open(results_file, 'w') as results_file:
        json.dump(results, results_file)

if __name__ == "__main__":
  app.run(main)