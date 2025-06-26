from open_spiel.python import policy
from utils.file_manipulation import load_from_file_json, save_to_file_json
import pyspiel
import numpy as np

cat_dims_by_game = {
    "leduc_poker": [2, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2],
    "kuhn_poker": [2, 3, 2, 2, 2]
}

def convert_one_hot_to_cat(one_hot, game_name):
    cat_dims = cat_dims_by_game[game_name]
    splits = np.cumsum(cat_dims)[:-1]
    chunks = np.split(one_hot, splits)

    categorical = np.array([
        np.argmax(chunk) if chunk.any() else len(chunk)
        for chunk in chunks
    ], dtype=int)

    return categorical

def create_categorical(policy: policy.TabularPolicy, game_name):
    cat_policy = {}
    dict_policy = policy.to_dict()
    for state in policy.states:
        cat_policy[tuple(convert_one_hot_to_cat(state.information_state_tensor(), game_name))] = dict_policy[policy._state_key(state, state.current_player())]
    return cat_policy

def one_hot_encode_policy(policy: policy.TabularPolicy):
    oh_policy = {}
    dict_policy = policy.to_dict()
    for state in policy.states:
        oh_policy[tuple(state.information_state_tensor())] = dict_policy[policy._state_key(state, state.current_player())]
    return oh_policy


def load_string_policy_from_file_for_game(file_name, game_name):
    game = pyspiel.load_game(game_name)
    policy_dict = load_from_file_json(file_name)
    ret_policy = policy.TabularPolicy(game)
    for key in policy_dict:
        ret_policy.action_probability_array[ret_policy.state_lookup[key]] = [ap[1] for ap in policy_dict[key]]
    return ret_policy