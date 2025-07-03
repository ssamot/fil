import random
from absl.testing import absltest
import numpy as np
import torch

from open_spiel.python import rl_environment
import pyspiel
from open_spiel.python.pytorch import dqn

def test_run_tic_tac_toe(self):
    env = rl_environment.Environment("leduc_poker")
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        dqn.DQN(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[16],
            replay_buffer_capacity=10,
            batch_size=5) for player_id in [0, 1]
    ]
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      current_agent = agents[current_player]
      agent_output = current_agent.step(time_step)
      time_step = env.step([agent_output.action])

    for agent in agents:
      agent.step(time_step)