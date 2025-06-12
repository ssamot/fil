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
the strategy profiles of the game. To train these networks a reservoir buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.

This implementation uses skip connections as described in the paper if two
consecutive layers of the advantage or policy network have the same number
of units, except for the last connection. Before the last hidden layer
a layer normalization is applied.
"""

import collections
import os
import random
import warnings

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

from tqdm import tqdm

import jax
import jax.numpy as jnp

from open_spiel.python import policy
import pyspiel

# The size of the shuffle buffer used to reshuffle part of the data each
# epoch within one training iteration
ADVANTAGE_TRAIN_SHUFFLE_SIZE = 100000
STRATEGY_TRAIN_SHUFFLE_SIZE = 1000000

def replace_row(tensor, index, new_row):
    before = keras.ops.slice(tensor, [0], [index])
    after = keras.ops.slice(tensor, [index + 1], [tensor.shape[0] - index - 1])
    new_row = keras.ops.expand_dims(new_row, axis=0)
    return keras.ops.concatenate([before, new_row, after], axis=0)

class ReservoirBuffer:
    def __init__(self, capacity, shapes, dtypes=None):
        self.capacity = capacity
        self.count = 0
        self.shapes = shapes
        self.n_fields = len(shapes)
        self.dtypes = dtypes or [jnp.float32] * self.n_fields

        self.buffers = [
            jnp.zeros((capacity, *shape), dtype=dtype)
            for shape, dtype in zip(shapes, self.dtypes)
        ]

    def add(self, data_tuple):
        assert len(data_tuple) == self.n_fields

        if self.count < self.capacity:
            idx = self.count
        else:
            idx = random.randint(0, self.count)
            if idx >= self.capacity:
                self.count += 1
                return  # discard sample

        for i, data in enumerate(data_tuple):
            tensor = jnp.expand_dims(jnp.asarray(data, dtype=self.dtypes[i]), axis=0)
            self.buffers[i] = jax.lax.dynamic_update_index_in_dim(
                self.buffers[i],
                tensor,
                idx,
                axis=0
            )

        self.count += 1

    def sample_batch(self, batch_size):
        valid_size = min(self.count, self.capacity)
        indices = random.sample(range(valid_size), batch_size)
        indices = jnp.array(indices)
        return tuple(jnp.take(buf, indices, axis=0) for buf in self.buffers)
    
    def __len__(self):
      return min(self.count, self.capacity)


class SkipDense(keras.layers.Layer):
  """Dense Layer with skip connection."""

  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.hidden = keras.layers.Dense(units, kernel_initializer='he_normal')

  def call(self, x):
    return self.hidden(x) + x


class PolicyNetwork(keras.Model):
  """Implements the policy network as an MLP.

  Implements the policy network as a MLP with skip connections in adjacent
  layers with the same number of units, except for the last hidden connection
  where a layer normalization is applied.
  """

  def __init__(self,
               input_size,
               policy_network_layers,
               num_actions,
               activation='leakyrelu',
               **kwargs):
    super().__init__(**kwargs)
    self._input_size = input_size
    self._num_actions = num_actions
    if activation == 'leakyrelu':
      self.activation = keras.layers.LeakyReLU(alpha=0.2)
    elif activation == 'relu':
      self.activation = keras.layers.ReLU()
    else:
      self.activation = activation

    self.softmax = keras.layers.Softmax()

    self.hidden = []
    prevunits = 0
    for units in policy_network_layers[:-1]:
      if prevunits == units:
        self.hidden.append(SkipDense(units))
      else:
        self.hidden.append(
            keras.layers.Dense(units, kernel_initializer='he_normal'))
      prevunits = units
    self.normalization = keras.layers.LayerNormalization()
    self.lastlayer = keras.layers.Dense(
        policy_network_layers[-1], kernel_initializer='he_normal')

    self.out_layer = keras.layers.Dense(num_actions)

  def call(self, inputs):
    """Applies Policy Network.

    Args:
        inputs: Tuple representing (info_state, legal_action_mask)

    Returns:
        Action probabilities
    """
    x, mask = inputs
    for layer in self.hidden:
      x = layer(x)
      x = self.activation(x)

    x = self.normalization(x)
    x = self.lastlayer(x)
    x = self.activation(x)
    x = self.out_layer(x)
    x = keras.ops.where(mask == 1, x, -10e20)
    x = self.softmax(x)
    return x


class AdvantageNetwork(keras.Model):
  """Implements the advantage network as an MLP.

  Implements the advantage network as an MLP with skip connections in
  adjacent layers with the same number of units, except for the last hidden
  connection where a layer normalization is applied.
  """

  def __init__(self,
               input_size,
               adv_network_layers,
               num_actions,
               activation='leakyrelu',
               **kwargs):
    super().__init__(**kwargs)
    self._input_size = input_size
    self._num_actions = num_actions
    if activation == 'leakyrelu':
      self.activation = keras.layers.LeakyReLU(alpha=0.2)
    elif activation == 'relu':
      self.activation = keras.layers.ReLU()
    else:
      self.activation = activation

    self.hidden = []
    prevunits = 0
    for units in adv_network_layers[:-1]:
      if prevunits == units:
        self.hidden.append(SkipDense(units))
      else:
        self.hidden.append(
            keras.layers.Dense(units, kernel_initializer='he_normal'))
      prevunits = units
    self.normalization = keras.layers.LayerNormalization()
    self.lastlayer = keras.layers.Dense(
        adv_network_layers[-1], kernel_initializer='he_normal')

    self.out_layer = keras.layers.Dense(num_actions)

  def call(self, inputs):
    """Applies Policy Network.

    Args:
        inputs: Tuple representing (info_state, legal_action_mask)

    Returns:
        Cumulative regret for each info_state action
    """
    x, mask = inputs
    for layer in self.hidden:
      x = layer(x)
      x = self.activation(x)

    x = self.normalization(x)
    x = self.lastlayer(x)
    x = self.activation(x)
    x = self.out_layer(x)
    x = mask * x

    return x


class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.
  """

  def __init__(self,
               game,
               policy_network_layers=(256, 256),
               advantage_network_layers=(128, 128),
               num_iterations: int = 100,
               num_traversals: int = 100,
               learning_rate: float = 1e-3,
               batch_size_advantage: int = 2048,
               batch_size_strategy: int = 2048,
               memory_capacity: int = int(1e6),
               policy_network_train_steps: int = 5000,
               advantage_network_train_steps: int = 750,
               reinitialize_advantage_networks: bool = True):
    """Initialize the Deep CFR algorithm.

    Args:
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: Number of iterations.
      num_traversals: Number of traversals per iteration.
      learning_rate: Learning rate.
      batch_size_advantage: (int) Batch size to sample from advantage memories.
      batch_size_strategy: (int) Batch size to sample from strategy memories.
      memory_capacity: Number of samples that can be stored in memory.
      policy_network_train_steps: Number of policy network training steps (one
        policy training iteration at the end).
      advantage_network_train_steps: Number of advantage network training steps
        (per iteration).
      reinitialize_advantage_networks: Whether to re-initialize the advantage
        network before training on each iteration.
      save_advantage_networks: If provided, all advantage network itearations
        are saved in the given folder. This can be useful to implement SD-CFR
        https://arxiv.org/abs/1901.07621
      save_strategy_memories: saves the collected strategy memories as a
        tfrecords file in the given location. This is not affected by
        memory_capacity. All memories are saved to disk and not kept in memory
    """
    all_players = list(range(game.num_players()))
    super(DeepCFRSolver, self).__init__(game, all_players)
    self._game = game
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
      # `_traverse_game_tree` does not take into account this option.
      raise ValueError('Simulatenous games are not supported.')
    self._batch_size_advantage = batch_size_advantage
    self._batch_size_strategy = batch_size_strategy
    self._policy_network_train_steps = policy_network_train_steps
    self._advantage_network_train_steps = advantage_network_train_steps
    self._policy_network_layers = policy_network_layers
    self._advantage_network_layers = advantage_network_layers
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    self._embedding_size = len(self._root_node.information_state_tensor(0))
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._reinitialize_advantage_networks = reinitialize_advantage_networks
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1
    self._learning_rate = learning_rate

    # Initialize policy network, loss, optmizer
    self._reinitialize_policy_network()

    # Initialize advantage networks, losses, optmizers
    self._adv_networks = []
    self._adv_networks_train = []
    self._loss_advantages = []
    self._optimizer_advantages = []
    self._advantage_train_step = []
    for player in range(self._num_players):
      self._adv_networks.append(
          AdvantageNetwork(self._embedding_size, self._advantage_network_layers,
                           self._num_actions))
      self._adv_networks_train.append(
          AdvantageNetwork(self._embedding_size,
                            self._advantage_network_layers, self._num_actions))
      self._loss_advantages.append(keras.losses.MeanSquaredError())
      self._optimizer_advantages.append(
          keras.optimizers.Adam(learning_rate=learning_rate))
      self._adv_networks[player].compile(optimizer=self._optimizer_advantages[player], loss='mse')
      self._adv_networks_train[player].compile(optimizer=self._optimizer_advantages[player], loss='mse')
      self._advantage_train_step.append(
          self._get_advantage_train_graph(player))

    self._create_memories(int(memory_capacity))

  def _reinitialize_policy_network(self):
    """Reinitalize policy network and optimizer for training."""
    self._policy_network = PolicyNetwork(self._embedding_size,
                                          self._policy_network_layers,
                                          self._num_actions)
    self._optimizer_policy = keras.optimizers.Adam(
        learning_rate=self._learning_rate)
    self._policy_network.compile(optimizer=self._optimizer_policy, loss='mse')
    self._loss_policy = keras.losses.MeanSquaredError()

  def _reinitialize_advantage_network(self, player):
    """Reinitalize player's advantage network and optimizer for training."""
    self._adv_networks_train[player] = AdvantageNetwork(
        self._embedding_size, self._advantage_network_layers,
        self._num_actions)
    self._optimizer_advantages[player] = keras.optimizers.Adam(
        learning_rate=self._learning_rate)
    self._adv_networks_train[player].compile(optimizer=self._optimizer_advantages[player], loss='mse')
    self._advantage_train_step[player] = (
        self._get_advantage_train_graph(player))

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def _create_memories(self, memory_capacity):
    """Create memory buffers and associated feature descriptions."""
    self._strategy_memories = ReservoirBuffer(memory_capacity, ((self._embedding_size,), (self._num_actions,), (1,), (self._num_actions,)))
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity, ((self._embedding_size,), (self._num_actions,), (1,), (self._num_actions,))) for _ in range(self._num_players)
    ]

  def solve(self):
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for _ in tqdm(range(self._num_iterations)):
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for p and train from scratch.
          self._reinitialize_advantage_network(p)
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

  def _traverse_game_tree(self, state, player):
    """Performs a traversal of the game tree using external sampling.

    Over a traversal the advantage and strategy memories are populated with
    computed advantage values and matched regrets respectively.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index for this traversal.

    Returns:
      Recursively returns expected payoffs for each action.
    """
    if state.is_terminal():
      # Terminal state get returns.
      return state.returns()[player]
    elif state.is_chance_node():
      # If this is a chance node, sample an action
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = np.random.choice(chance_outcome, p=chance_proba)
      return self._traverse_game_tree(state.child(action), player)
    elif state.current_player() == player:
      # Update the policy over the info set & actions via regret matching.
      _, strategy = self._sample_action_from_advantage(state, player)
      exp_payoff = 0 * strategy
      for action in state.legal_actions():
        exp_payoff[action] = self._traverse_game_tree(
            state.child(action), player)
      ev = np.sum(exp_payoff * strategy)
      samp_regret = (exp_payoff - ev) * state.legal_actions_mask(player)
      self._advantage_memories[player].add((state.information_state_tensor(),
                                            samp_regret, self._iteration,
                                            state.legal_actions_mask(player)))
      return ev
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = strategy
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories.add((state.information_state_tensor(other_player), strategy, self._iteration, state.legal_actions_mask(other_player)))
      return self._traverse_game_tree(state.child(sampled_action), player)

  def _get_matched_regrets(self, info_state, legal_actions_mask, player):
    """TF-Graph to calculate regret matching."""
    advs = self._adv_networks[player](
        (keras.ops.expand_dims(info_state, axis=0), keras.ops.expand_dims(legal_actions_mask, axis=0)),
        training=False)[0]
    advantages = keras.ops.maximum(advs, 0)
    summed_regret = keras.ops.sum(advantages)
    if summed_regret > 0:
      matched_regrets = advantages / summed_regret
    else:
      matched_regrets = keras.ops.one_hot(
          keras.ops.argmax(keras.ops.where(legal_actions_mask == 1, advs, -10e20)),
          self._num_actions)
    return advantages, matched_regrets

  def _sample_action_from_advantage(self, state, player):
    """Returns an info state policy by applying regret-matching.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index over which to compute regrets.

    Returns:
      1. (np-array) Advantage values for info state actions indexed by action.
      2. (np-array) Matched regrets, prob for actions indexed by action.
    """    
    info_state = keras.ops.convert_to_tensor(state.information_state_tensor(player))
    legal_actions_mask = keras.ops.convert_to_tensor(state.legal_actions_mask(player))
    advantages, matched_regrets = self._get_matched_regrets(
        info_state, legal_actions_mask, player)
    return keras.ops.convert_to_numpy(advantages), keras.ops.convert_to_numpy(matched_regrets)

  def action_probabilities(self, state, player_id=None):
    """Returns action probabilities dict for a single batch."""
    del player_id  # unused
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    legal_actions_mask = keras.ops.convert_to_tensor(state.legal_actions_mask(cur_player))
    info_state_vector = keras.ops.convert_to_tensor(state.information_state_tensor())
    if len(info_state_vector.shape) == 1:
      info_state_vector = keras.ops.expand_dims(info_state_vector, axis=0)
    probs = self._policy_network((info_state_vector, legal_actions_mask),
                                 training=False)
    probs = keras.ops.convert_to_numpy(probs)
    return {action: probs[0][action] for action in legal_actions}

  def _get_advantage_train_graph(self, player):
    """Return TF-Graph to perform advantage network train step."""
    def train_step(info_states, advantages, iterations, masks, iteration):
        model = self._adv_networks_train[player]
        
        # Prepare data
        x = (info_states, masks)
        y = advantages,
        sample_weight = iterations * 2 / iteration
        
        # Use Keras' train_on_batch which handles gradients internally
        loss = model.train_on_batch(
            x, y, 
            sample_weight=sample_weight,
            return_dict=False
        )

        return loss

    return train_step

  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      The average loss over the advantage network of the last batch.
    """

    if self._batch_size_advantage:
        if self._batch_size_advantage > len(self._advantage_memories[player]):
          ## Skip if there aren't enough samples
          return None

    for _ in range(self._advantage_network_train_steps):
      data = self._advantage_memories[player].sample_batch(self._batch_size_advantage)
      main_loss = self._advantage_train_step[player](*data, self._iteration)

      self._adv_networks[player].set_weights(
          self._adv_networks_train[player].get_weights())
    return main_loss

  def _learn_strategy_network(self):
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on the last training batch of transitions
      or `None`.
    """
    def train_step(info_states, action_probs, iterations, masks):
        model = self._policy_network
        
        # Prepare data
        x = (info_states, masks)
        y = action_probs
        sample_weight = iterations * 2 / self._iteration
        
        # Use Keras' train_on_batch which handles gradients internally
        loss = model.train_on_batch(
            x, y,
            sample_weight=sample_weight,
            return_dict=False
        )
        
        return loss

    for _ in range(self._policy_network_train_steps):
      data = self._strategy_memories.sample_batch(self._batch_size_strategy)
      main_loss = train_step(*data)

    return main_loss
