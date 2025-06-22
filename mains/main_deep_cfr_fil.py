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
from deepcfr import deep_cfr_fil as deep_cfr
import pyspiel

FLAGS = flags.FLAGS
from tqdm import tqdm

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
        policy_network_layers=config.policy_network_layers,
        advantage_network_layers=config.advantage_network_layers,
        num_iterations=config.num_iterations,
        num_traversals=config.num_traversals,
        learning_rate=config.learning_rate,
        batch_size_advantage=config.batch_size_advantage,
        batch_size_strategy=config.batch_size_strategy,
        memory_capacity=config.memory_capacity,
        policy_network_train_steps=config.policy_network_train_steps,
        advantage_network_train_steps=config.advantage_network_train_steps,
        reinitialize_advantage_networks=config.reinitialize_advantage_networks,
        cat_split_dims=config.cat_split_dims,
        cat_real_dims=config.cat_real_dims,
        fil_groups=config.fil_groups)

    results = {}
    advantage_losses = collections.defaultdict(list)
    with tqdm(range(deep_cfr_solver._num_iterations)) as pbar:
        for i in range(deep_cfr_solver._num_iterations):
            for p in range(deep_cfr_solver._num_players):
                for _ in range(deep_cfr_solver._num_traversals):
                    deep_cfr_solver._traverse_game_tree(deep_cfr_solver._root_node, p)
                if deep_cfr_solver._reinitialize_advantage_networks:
                    # Re-initialize advantage network for p and train from scratch.
                    deep_cfr_solver.reinitialize_advantage_network(p)
                advantage_losses[p].append(deep_cfr_solver._learn_advantage_network(p))

            # Train policy network.
            deep_cfr_solver.reinitialize_policy_network()
            policy_loss = deep_cfr_solver._learn_strategy_network()
            deep_cfr_solver._iteration += 1

            if i % config.log_frequency == 0:
                average_policy = policy.tabular_policy_from_callable(
                    game, deep_cfr_solver.action_probabilities)
                conv = exploitability.nash_conv(game, average_policy)

                # logging.info("Iteration: {} NashConv: {}".format(i, conv))
                results[i] = conv
            pbar.update(1)
            pbar.set_description(f"Iteration {i}")
            pbar.set_postfix({"exploitability": f"{conv:.6f}"})

    results_file = config.results_file_base + "fil.json"
    with open(results_file, 'w') as results_file:
        json.dump(results, results_file)
    end = time.time()
    print("Run took", end - start, "seconds")


if __name__ == "__main__":
    app.run(main)
