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

def create_sane(policy: policy.TabularPolicy, game_name):
    cat_policy = create_categorical(policy, game_name)
    sane_policy = {}
    conversion = {}
    for key in cat_policy:
        new_key = tuple(convert_categorical_to_sane(key, game_name))
        if new_key in conversion:
            conversion[new_key].append(key)
        else:
            conversion[new_key] = [key]
        sane_policy[tuple(convert_categorical_to_sane(key, game_name))] = cat_policy[key]
    return sane_policy

def convert_categorical_to_sane(raw_state, game_name):
    if game_name == "leduc_poker":
        CALL = 0
        BET = 1
        NO_ACTION = 2
        NO_CARD = 6

        if len(raw_state) != 11:
            raise ValueError(f"Expected state of length 11, got {len(raw_state)}")
        player = raw_state[0]
        private_card = raw_state[1]
        public_card = raw_state[2]
        actions_first_round = raw_state[3:7]
        actions_second_round = raw_state[7:]

        features = []
        
        # Player position (0 or 1)
        features.append(player)

        # Identify round
        public_card_dealt = public_card != NO_CARD
        features.append(public_card_dealt)

        features.append(public_card)
        
        has_pair = 1 if private_card == public_card else 0
        
        features.append(has_pair) # can be probably removed as that can be deduced from the hand strenght but not sure

        # Hand strength estimation
        if has_pair:
            hand_strength = 0.75 + (private_card / 5.0) * 0.25  # 0.75-1.0 for pairs
        else:
            hand_strength = (private_card / 5.0) * 0.75  # 0.0-0.75 for high cards
        features.append(hand_strength)
        
        # Action sequence analysis
        valid_actions_first_round = [a for a in actions_first_round if a != NO_ACTION]
        valid_actions_second_round = [a for a in actions_second_round if a != NO_ACTION]
        valid_actions = valid_actions_first_round + valid_actions_second_round

        facing_bet = len(valid_actions) > 0 and valid_actions[-1] == BET
        features.append(facing_bet)
        
        # Bet counting
        num_bets_first_round = valid_actions_first_round.count(BET)
        num_bets_second_round = valid_actions_second_round.count(BET)

        features.extend([num_bets_first_round, num_bets_second_round])

        # Identify checks
        checked_first_round = len(valid_actions_first_round) > 0 and valid_actions_first_round[0] == CALL
        checked_second_round = len(valid_actions_second_round) > 0 and valid_actions_second_round[0] == CALL

        features.extend([checked_first_round, checked_second_round])
        
        return np.array(features, dtype=np.float32)
    else:
        raise NotImplementedError