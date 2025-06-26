from open_spiel.python.algorithms import exploitability, cfr
from open_spiel.python import policy
from utils.file_manipulation import load_from_file_json, save_to_file_json

import tqdm
import sys
import pyspiel

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def get_cfr_policy(game_name):
    file_name = "../data/cfr_policy_" + game_name + ".json"
    game = pyspiel.load_game(game_name)
    try:
        average_policy_dict = load_from_file_json(file_name)
        average_policy = policy.TabularPolicy(game)
        for key in average_policy_dict:
            average_policy.action_probability_array[average_policy.state_lookup[key]] = [ap[1] for ap in average_policy_dict[key]]
        return average_policy
    except FileNotFoundError:
        print("File not found - recalculating the strategy")
    cfr_solver = cfr.CFRPlusSolver(game)
    for _ in tqdm.tqdm(range(10000)):
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