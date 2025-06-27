from absl import app
from absl import flags
from absl import logging

import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr

import tqdm

def main(unused_argv):
    np.random.seed(0)
    game = pyspiel.load_game("leduc_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in tqdm.tqdm(range(10000)):
        es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))

if __name__ == "__main__":
  app.run(main)