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

"""Python base module for the implementations of Monte Carlo Counterfactual Regret Minimization."""

import enum
import numpy as np
from open_spiel.python import policy

import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from policies import policy_manipulation_and_conversion as conversion

import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

class ResetType(enum.Enum):
  NONE = 0
  REGRET = 1
  ALL = 2

REGRET_INDEX = 0
AVG_POLICY_INDEX = 1
INFOSTATE_TENSOR_INDEX = 2

infoset_key_map = {}

def create_regressor():
  return MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=1000,         # Number of boosting rounds
    #learning_rate=0.05,        # Small to avoid overshooting
    #num_leaves=128,            # Controls tree complexity
    #max_depth=10,              # Maximum tree depth
    #min_child_samples=30,      # Minimum samples per leaf
    #subsample=0.9,             # Row sampling
    #subsample_freq=1,
    #colsample_bytree=0.9,      # Feature sampling
    #reg_alpha=0.1,             # L1 regularization
    #reg_lambda=0.1,            # L2 regularization
    verbose=-1))

def get_infostate_key(state, game):
     infoset_string = state.information_state_string(state.current_player())
     if infoset_string in infoset_key_map:
        return infoset_key_map[infoset_string]
     game_name = str(game)[:-2]
     new_key = tuple(conversion.convert_categorical_to_sane(conversion.convert_one_hot_to_cat(state.information_state_tensor(state.current_player()), game_name), game_name))
     infoset_key_map[infoset_string] = new_key
     return new_key

class AveragePolicy(policy.Policy):
  """A policy object representing the average policy for MCCFR algorithms."""

  def __init__(self, game, player_ids, infostates):
    # Do not create a copy of the dictionary
    # but work on the same object
    super().__init__(game, player_ids)
    self._infostates = infostates

  def action_probabilities(self, state, player_id=None):
    """Returns the MCCFR average policy for a player in a state.

    If the policy is not defined for the provided state, a uniform
    random policy is returned.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for which we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state. If the policy is defined for the state, this
      will contain the average MCCFR strategy defined for that state.
      Otherwise, it will contain all legal actions, each with the same
      probability, equal to 1 / num_legal_actions.
    """
    if player_id is None:
      player_id = state.current_player()
    legal_actions = state.legal_actions()
    info_state_key = get_infostate_key(state, self.game)
    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is None:
      return {a: 1 / len(legal_actions) for a in legal_actions}
    cumstrat = retrieved_infostate[AVG_POLICY_INDEX][legal_actions]
    avstrat = (
        cumstrat /
        cumstrat.sum())
    return {legal_actions[i]: avstrat[i] for i in range(len(legal_actions))}


class MCCFRSolverBase(object):
  """A base class for both outcome MCCFR and external MCCFR."""

  def __init__(self, game, max_infostates, reset_type):
    self._game = game
    self._infostates = {}  # infostate keys -> [regrets, avg strat]
    self._num_players = game.num_players()
    self._regret_regressor = None
    self._policy_regressor = None
    self._max_infostates = max_infostates
    self._iteration = 0
    self._num_actions = game.num_distinct_actions()
    self._reset_type = reset_type

  def _get_infostate_key(self, state):
     return get_infostate_key(state, self._game)

  def _lookup_infostate_info(self, state):
    """Looks up an information set table for the given key.

    Args:
      info_state_key: information state key (string identifier).
      num_legal_actions: number of legal actions at this information state.

    Returns:
      A list of:
        - the average regrets as a numpy array of shape [num_legal_actions]
        - the average strategy as a numpy array of shape
        [num_legal_actions].
          The average is weighted using `my_reach`
    """
    info_state_key = self._get_infostate_key(state)

    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is not None:
      return retrieved_infostate

    # Start with a small amount of regret and total accumulation, to give a
    # uniform policy: this will get erased fast.
    if len(self._infostates) == self._max_infostates:
        return None
    else:
        if self._iteration == 0:
            self._infostates[info_state_key] = [
                np.ones(self._num_actions, dtype=np.float64) / 1e6,
                np.ones(self._num_actions, dtype=np.float64) / 1e6,
                state.information_state_tensor(state.current_player())
            ]
        else:
            infostate_tensor = state.information_state_tensor(state.current_player())
            self._infostates[info_state_key] = [
                self._regret_regressor.predict([infostate_tensor])[0],
                self._policy_regressor.predict([infostate_tensor])[0],
                infostate_tensor
            ]
    return self._infostates[info_state_key]

  def _add_regret(self, info_state_key, action_idx, amount):
    if info_state_key in self._infostates:
        self._infostates[info_state_key][REGRET_INDEX][action_idx] += amount

  def _add_avstrat(self, info_state_key, action_idx, amount):
    if info_state_key in self._infostates:
        self._infostates[info_state_key][AVG_POLICY_INDEX][action_idx] += amount

  def average_policy(self):
    """Computes the average policy, containing the policy for all players.

    Returns:
      An average policy instance that should only be used during
      the lifetime of solver object.
    """
    return AveragePolicy(self._game, list(range(self._num_players)),
                         self._infostates)

  def _regret_matching(self, state, legal_actions):
    """Applies regret matching to get a policy.

    Args:
      regrets: numpy array of regrets for each action.
      num_legal_actions: number of legal actions at this state.

    Returns:
      numpy array of the policy indexed by the index of legal action in the
      list.
    """
    info_state_key = self._get_infostate_key(state)
    num_legal_actions = len(legal_actions)
    if info_state_key not in self._infostates:
        if self._iteration > 0:
            info_state_tensor = state.information_state_tensor(state.current_player())
            regrets = self._regret_regressor.predict([info_state_tensor])[0]
        else:
            return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions # return uniform policy if the regressor is not yet trained

    else:
       regrets = self._infostates[info_state_key][REGRET_INDEX]
    regrets = regrets[legal_actions]    
    positive_regrets = np.maximum(regrets,
                                  np.zeros(num_legal_actions, dtype=np.float64))
    sum_pos_regret = positive_regrets.sum()
    if sum_pos_regret <= 0:
      return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
    else:
      return positive_regrets / sum_pos_regret

  def _update_regressor(self):
    self._iteration += 1
    self._regret_regressor = create_regressor()
    self._policy_regressor = create_regressor()
    y_regret, y_policy, X = zip(*[(infostate_info[REGRET_INDEX], infostate_info[AVG_POLICY_INDEX], infostate_info[INFOSTATE_TENSOR_INDEX]) for infostate_info in self._infostates.values()])
    y_regret = np.asarray(y_regret)
    y_policy = np.asarray(y_policy)
    X = np.asarray(X)
    self._regret_regressor.fit(X, y_regret)
    self._policy_regressor.fit(X, y_policy)
    # Reset infostate strutures based on the reset type
    if self._reset_type == ResetType.REGRET:
        for infostate_info in self._infostates.values():
            infostate_info[REGRET_INDEX] = self._regret_regressor.predict([infostate_info[INFOSTATE_TENSOR_INDEX]])[0]
    elif self._reset_type == ResetType.ALL:
       self._infostates = {}