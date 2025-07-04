from absl import app
from absl import flags
from absl import logging

import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr

import tqdm
import json

def main(unused_argv):
    np.random.seed(0)
    game = pyspiel.load_game("leduc_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.SIMPLE)
    cfr_iterations = 180000
    results = {}
    with tqdm.tqdm(range(cfr_iterations)) as pbar:
        for i in range(cfr_iterations):
            es_solver.iteration()
            if((i%100) == 0):
                conv = exploitability.nash_conv(game, es_solver.average_policy())
                results[i] = conv
            pbar.update(1)            
            pbar.set_description(f"Iteration {i}")
            pbar.set_postfix({"exploitability": f"{conv:.6f}"})
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))
    results[i] = conv

    results_file = "./results/mccfr_leduc.json"
    with open(results_file, 'w') as results_file:
        json.dump(results, results_file)

if __name__ == "__main__":
  app.run(main)