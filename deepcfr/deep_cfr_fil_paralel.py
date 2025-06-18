from deepcfr import deep_cfr_fil as deep_cfr

import numpy as np
import collections

from threading import Lock

class DeepCFRSolverParalel(deep_cfr.DeepCFRSolver):
    
    def __init__(self, *args, **kwargs):
      super(DeepCFRSolverParalel, self).__init__(*args, **kwargs)
      self._strategy_lock = Lock()
      self._advantage_lock = Lock()

    def _traverse_game_tree(self, state, player):
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
        with self._advantage_lock:
          self._advantage_memories[player].add(
              deep_cfr.AdvantageMemory(state.information_state_tensor(), self._iteration,
                              sampled_regret_arr, action))
        return cfv
      else:
        other_player = state.current_player()
        _, strategy = self._sample_action_from_advantage(state, other_player)
        # Recompute distribution for numerical errors.
        probs = np.array(strategy)
        probs /= probs.sum()
        sampled_action = np.random.choice(range(self._num_actions), p=probs)
        with self._strategy_lock:
          self._strategy_memories[other_player].add(
              deep_cfr.StrategyMemory(
                  state.information_state_tensor(other_player), self._iteration,
                  strategy))
        return self._traverse_game_tree(state.child(sampled_action), player)