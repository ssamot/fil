from open_spiel.python.algorithms import exploitability, cfr
from open_spiel.python import policy
from utils.file_manipulation import load_from_file_json, save_to_file_json
from policies.policy_manipulation_and_conversion import load_string_policy_from_file_for_game, one_hot_encode_policy, create_categorical, create_sane

import tqdm
import sys
import pyspiel

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def get_sane_cfr_policy(game_name):
    file_name = "../data/cfr_sane_policy_" + game_name + ".json"
    # try:
    #     return load_from_file_json(file_name)
    # except FileNotFoundError:
    #     print("File not found - recalcuating from string policy")
    sane_policy = create_sane(get_cfr_policy(game_name), game_name)
    sane_string_policy = {str(key): value for key, value in sane_policy.items()}
    save_to_file_json(sane_string_policy, file_name)
    return sane_policy

def get_categorical_cfr_policy(game_name):
    file_name = "../data/cfr_categorical_policy_" + game_name + ".json"
    try:
        return load_from_file_json(file_name)
    except FileNotFoundError:
        print("File not found - recalcuating from string policy")
    cat_policy = create_categorical(get_cfr_policy(game_name), game_name)
    cat_string_policy = {str(key): value for key, value in cat_policy.items()}
    save_to_file_json(cat_string_policy, file_name)
    return cat_policy

def get_one_hot_encoded_cfr_policy(game_name):
    file_name = "../data/cfr_onehot_policy_" + game_name + ".json"
    try:
        return load_from_file_json(file_name)
    except FileNotFoundError:
        print("File not found - recalcuating from string policy")
    oh_policy = one_hot_encode_policy(get_cfr_policy(game_name))
    oh_string_policy = {str(key): value for key, value in oh_policy.items()}
    save_to_file_json(oh_string_policy, file_name)
    return oh_policy

def get_string_cfr_policy(game_name):
    file_name = "../data/cfr_string_policy_" + game_name + ".json"
    try:
        load_from_file_json(file_name)
    except FileNotFoundError:
        get_cfr_policy(game_name)        
    return load_from_file_json(file_name)

def get_cfr_policy(game_name):
    file_name = "../data/cfr_string_policy_" + game_name + ".json"
    game = pyspiel.load_game(game_name)
    try:
        return load_string_policy_from_file_for_game(file_name, game_name)
    except FileNotFoundError:
        print("File not found - recalculating the strategy")
    cfr_solver = cfr.CFRPlusSolver(game)
    for _ in tqdm.tqdm(range(10)):
        cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    save_to_file_json(average_policy.to_dict(), file_name)
    return average_policy

def solve_and_compute_exploitability(game_name):
    game = pyspiel.load_game(game_name)
    average_policy = get_cfr_policy(game_name)
    print("Exploitability:", exploitability.nash_conv(game, average_policy))

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if not args or len(args) > 1:
        print("Usage: python sequence_form.py game_name_in_openspiel")
        sys.exit(1)

    solve_and_compute_exploitability(args[0])
    print(len(get_string_cfr_policy(args[0])))
    print(len(get_sane_cfr_policy(args[0])))