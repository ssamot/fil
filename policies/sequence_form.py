from open_spiel.python.algorithms import sequence_form_lp, exploitability
import sys
import pyspiel

def solve_game(game_name):
    game = pyspiel.load_game(game_name)
    _, _, policy1, policy2 = sequence_form_lp.solve_zero_sum_game(game)
    br1 = exploitability.best_response(game, policy1, 1)
    br0 = exploitability.best_response(game, policy2, 0)
    print("Exploitability:", (br1["best_response_value"] + br0["best_response_value"])/2)

if __name__ == "__main__":
    args = sys.argv[1:]
    use_log = False

    if not args or len(args) > 1:
        print("Usage: python sequence_form.py game_name_in_openspiel")
        sys.exit(1)

    solve_game(args[0])