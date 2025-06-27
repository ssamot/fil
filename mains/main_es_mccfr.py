from absl import app
from absl import flags
from absl import logging

import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability

import tqdm

from our_cfr import external_sampling_mccfr_with_lgb

def main(unused_argv):
    np.random.seed(0)
    game = pyspiel.load_game("leduc_poker")
    es_solver = external_sampling_mccfr_with_lgb.ExternalSamplingSolver(game, 1000)
    cfr_iterations = 10000
    with tqdm.tqdm(range(cfr_iterations)) as pbar:
        for i in range(cfr_iterations):
            es_solver.iteration()
            if ((i % 100) == 0):
                conv = exploitability.nash_conv(game, es_solver.average_policy())
            pbar.update(1)

            pbar.set_description(f"Iteration {i}")
            pbar.set_postfix({"exploitability": f"{conv:.6f}"})
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))

if __name__ == "__main__":
  app.run(main)