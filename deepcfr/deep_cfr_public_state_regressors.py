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

"""Implements Deep CFR Algorithm.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game.  To train these networks a fixed ring buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.
"""

import collections
import math
import random
import numpy as np
from scipy import stats
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import sklearn

from open_spiel.python import policy
import pyspiel

def create_regressor():
  return sklearn.pipeline.Pipeline([('polynomial_features', sklearn.preprocessing.PolynomialFeatures(degree=4)), ('linear_regressor', sklearn.linear_model.LinearRegression())])

AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", "public_state_key hand_strenght iteration advantage action")

StrategyMemory = collections.namedtuple(
    "StrategyMemory", "public_state_key hand_strenght iteration strategy_action_probs")

class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.
  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)


class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm with PyTorch.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.

  Note: batch sizes default to `None` implying that training over the full
        dataset in memory is done by default.  To sample from the memories you
        may set these values to something less than the full capacity of the
        memory.
  """

  def __init__(self,
               game,
               num_iterations: int = 100,
               num_traversals: int = 20,
               batch_size_advantage=None,
               batch_size_strategy=None,
               memory_capacity: int = int(1e6),
               policy_network_train_steps: int = 1,
               advantage_network_train_steps: int = 1,
               reinitialize_advantage_networks: bool = True,
               public_state_indexes = []):
    """Initialize the Deep CFR algorithm.

    Args:
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: (int) Number of training iterations.
      num_traversals: (int) Number of traversals per iteration.
      learning_rate: (float) Learning rate.
      batch_size_advantage: (int or None) Batch size to sample from advantage
        memories.
      batch_size_strategy: (int or None) Batch size to sample from strategy
        memories.
      memory_capacity: Number af samples that can be stored in memory.
      policy_network_train_steps: Number of policy network training steps (per
        iteration).
      advantage_network_train_steps: Number of advantage network training steps
        (per iteration).
      reinitialize_advantage_networks: Whether to re-initialize the advantage
        network before training on each iteration.
    """
    all_players = list(range(game.num_players()))
    super(DeepCFRSolver, self).__init__(game, all_players)
    self._game = game
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
      # `_traverse_game_tree` does not take into account this option.
      raise ValueError("Simulatenous games are not supported.")
    self._batch_size_advantage = batch_size_advantage
    self._batch_size_strategy = batch_size_strategy
    self._policy_network_train_steps = policy_network_train_steps
    self._advantage_network_train_steps = advantage_network_train_steps
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    self._embedding_size = len(self._root_node.information_state_tensor(0))
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._reinitialize_advantage_networks = reinitialize_advantage_networks
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1
    self._public_state_indexes = public_state_indexes
    self._public_state_map = {}
    self._hand_strenght_map = {}

    # Define strategy network, loss & memory.
    self._strategy_memories = ReservoirBuffer(memory_capacity)
    self._policy_regressors = {}

    # Define advantage network, loss & memory. (One per player)
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._advantage_regressors = [
        {} for _ in range(self._num_players)
    ]

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def get_table_key(self, state):
    infostate_string = state.information_state_string
    if infostate_string in self._public_state_map:
      return self._public_state_map[infostate_string]
    tensor = state.information_state_tensor()
    tensor = np.concatenate((tensor[:self._public_state_indexes[0]], tensor[self._public_state_indexes[1]:]))
    self._public_state_map[infostate_string] = str(tensor)
    return self._public_state_map[infostate_string]

  def get_state_hand_strenght(self, state):
    infostate_string = state.information_state_string
    if infostate_string in self._hand_strenght_map:
      return self._hand_strenght_map[infostate_string]
    tensor = state.information_state_tensor()
    private_card = np.argmax(tensor[2:8])
    public_card = np.argmax(tensor[8:14]) if np.max(tensor[8:14]) > 0 else None
    self._hand_strenght_map[infostate_string] = self.get_hand_strenght(private_card, public_card)
    return self._hand_strenght_map[infostate_string]

  def get_hand_strenght(self, private_card, public_card):
    if public_card is None:
      return (private_card // 2) / 5
    else:
      if (private_card == public_card + 1 and private_card % 2 == 1) or (private_card == public_card - 1 and private_card % 2 == 0):
        return (3 + private_card // 2) / 5
      else:
        return (private_card // 2) / 5

  def solve(self):
    """Solution logic for Deep CFR.

    Traverses the game tree, while storing the transitions for training
    advantage and policy networks.

    Returns:
      1. (nn.Module) Instance of the trained policy network for inference.
      2. (list of floats) Advantage network losses for
        each player during each iteration.
      3. (float) Policy loss.
    """
    advantage_losses = collections.defaultdict(list)
    for _ in tqdm(range(self._num_iterations)):
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        self._update_advantage(p)
      self._iteration += 1
      # Train policy network.
    policy_loss = self._compute_policy()
    return self._policy_network, advantage_losses, policy_loss

  def _traverse_game_tree(self, state, player):
    """Performs a traversal of the game tree.

    Over a traversal the advantage and strategy memories are populated with
    computed advantage values and matched regrets respectively.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index for this traversal.

    Returns:
      (float) Recursively returns expected payoffs for each action.
    """
    expected_payoff = collections.defaultdict(float)
    if state.is_terminal():
      # Terminal state get returns.
      return state.returns()[player]
    elif state.is_chance_node():
      # If this is a chance node, sample an action
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = np.random.choice(chance_outcome, p=chance_proba)
      return self._traverse_game_tree(state.child(action), player)
    elif state.current_player() == player:
      sampled_regret = collections.defaultdict(float)
      # Update the policy over the info set & actions via regret matching.
      _, strategy = self._sample_action_from_advantage(state, player)
      for action in state.legal_actions():
        expected_payoff[action] = self._traverse_game_tree(
            state.child(action), player)
      cfv = 0
      for a_ in state.legal_actions():
        cfv += strategy[a_] * expected_payoff[a_]
      for action in state.legal_actions():
        sampled_regret[action] = expected_payoff[action]
        sampled_regret[action] -= cfv
      sampled_regret_arr = [0] * self._num_actions
      for action in sampled_regret:
        sampled_regret_arr[action] = sampled_regret[action]
      self._advantage_memories[player].add(
          AdvantageMemory(self.get_table_key(state), self.get_state_hand_strenght(state), self._iteration,
                          sampled_regret_arr, action))
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = np.array(strategy)
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories.add(
          StrategyMemory(self.get_table_key(state), self.get_state_hand_strenght(state), self._iteration,
              strategy))
      return self._traverse_game_tree(state.child(sampled_action), player)

  def _sample_action_from_advantage(self, state, player):
    """Returns an info state policy by applying regret-matching.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index over which to compute regrets.

    Returns:
      1. (list) Advantage values for info state actions indexed by action.
      2. (list) Matched regrets, prob for actions indexed by action.
    """
    legal_actions = state.legal_actions(player)
    public_state_key = self.get_table_key(state)
    hand_strenght = self.get_state_hand_strenght(state)
    if public_state_key in self._advantage_regressors[player]:
      raw_advantages = self._advantage_regressors[player][public_state_key].predict([[hand_strenght]])[0]
    else:
      raw_advantages = np.ones(self._num_actions)
    advantages = [max(0., advantage) for advantage in raw_advantages]
    cumulative_regret = np.sum([advantages[action] for action in legal_actions])
    matched_regrets = np.array([0.] * self._num_actions)
    if cumulative_regret > 0.:
      for action in legal_actions:
        matched_regrets[action] = advantages[action] / cumulative_regret
    else:
      matched_regrets[max(legal_actions, key=lambda a: raw_advantages[a])] = 1
    return advantages, matched_regrets

  def action_probabilities(self, state, player_id=None):
    """Computes action probabilities for the current player in state.

    Args:
      state: (pyspiel.State) The state to compute probabilities for.
      player_id: unused, but needed to implement the Policy API.

    Returns:
      (dict) action probabilities for a single batch.
    """
    del player_id
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    public_state_key = self.get_table_key(state)
    hand_strenght = self.get_state_hand_strenght(state)
    if public_state_key in self._policy_regressors:
      probs = self._policy_regressors[public_state_key].predict([[hand_strenght]])[0]
    else:
      probs = np.zeros(self._num_actions)
      probs[legal_actions] = 1./len(legal_actions)
    return {action: probs[action] for action in legal_actions}

  def _learn_advantage_regressors(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      (float) The average loss over the advantage network.
    """
    if self._batch_size_advantage:
      if self._batch_size_advantage > len(self._advantage_memories[player]):
        samples = self._advantage_memories[player]
      else:
        samples = self._advantage_memories[player].sample(self._batch_size_advantage)
    else:
      samples = self._advantage_memories[player]
    
    training_data = {}
    for s in samples:
      if s.public_state_key in training_data:
        training_data[s.public_state_key][0].append([s.hand_strenght])
        training_data[s.public_state_key][1].append(s.advantage)
      else:
        training_data[s.public_state_key] = [[[s.hand_strenght]], [s.advantage]]

    for public_state_key in training_data:
      self._advantage_regressors[player][public_state_key] = create_regressor()
      self._advantage_regressors[player][public_state_key].fit(training_data[public_state_key][0], training_data[public_state_key][1])

  def _learn_strategy_regressors(self):
    """Compute the loss over the strategy network.

    Returns:
      (float) The average loss obtained on this batch of transitions or `None`.
    """
    if self._batch_size_strategy:
      if self._batch_size_strategy > len(self._strategy_memories):
        samples = self._strategy_memories
      else:
        samples = self._strategy_memories.sample(self._batch_size_strategy)
    else:
      samples = self._strategy_memories

    training_data = {}
    for s in samples:
      if s.public_state_key in training_data:
        training_data[s.public_state_key][0].append([s.hand_strenght])
        training_data[s.public_state_key][1].append(s.strategy_action_probs)
      else:
        training_data[s.public_state_key] = [[[s.hand_strenght]], [s.strategy_action_probs]]

    for public_state_key in training_data:
      self._policy_regressors[public_state_key] = create_regressor()
      self._policy_regressors[public_state_key].fit(training_data[public_state_key][0], training_data[public_state_key][1])