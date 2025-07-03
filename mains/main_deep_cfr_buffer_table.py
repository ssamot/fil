# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Make sure the script works no matter from where it is called
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

"""Python Deep CFR example."""

from absl import app
from absl import flags
from absl import logging

import collections
import json
import importlib.util

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel
from deepcfr import deep_cfr_buffer_table as deep_cfr

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, 'Path to the config file')
flags.mark_flag_as_required('config')


def main(unused_argv):
  import time
  start = time.time()
  config_path = os.path.abspath(FLAGS.config)
  spec = importlib.util.spec_from_file_location("config", config_path)
  config = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(config)
  logging.info("Loading %s", config.game_name)
  game = pyspiel.load_game(config.game_name)
  deep_cfr_solver = deep_cfr.DeepCFRSolver(
    game,
    num_iterations=config.num_iterations,
    num_traversals=config.num_traversals,
    batch_size_advantage=config.batch_size_advantage * config.advantage_network_train_steps,
    batch_size_strategy=config.batch_size_strategy * config.policy_network_train_steps,
    memory_capacity=config.memory_capacity)

  results = {}
  for i in range(deep_cfr_solver._num_iterations):
    for p in range(deep_cfr_solver._num_players):
      for _ in range(deep_cfr_solver._num_traversals):
        deep_cfr_solver._traverse_game_tree(deep_cfr_solver._root_node, p)
      deep_cfr_solver._learn_advantage_network(p)

    # Train policy network.
    deep_cfr_solver._learn_strategy_network()
    deep_cfr_solver._iteration += 1

    if i % config.log_frequency == 0:
      average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)
      conv = exploitability.nash_conv(game, average_policy)

      logging.info("Iteration: {} NashConv: {}".format(i, conv))
      results[i] = conv
  
  results_file = config.results_file_base + "buffer_table.json"
  with open(results_file, 'w') as results_file:
    json.dump(results, results_file)
  end = time.time()
  print("Run took", end - start, "seconds")

if __name__ == "__main__":
  app.run(main)
