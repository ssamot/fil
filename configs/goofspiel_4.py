game_name = "turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=4,points_order=descending))"
num_iterations = 1000
num_traversals = 1500
log_frequency = 1
results_file_base = "../results/goofspiel_4_"
policy_network_layers=(64, 64, 64)
advantage_network_layers=(64, 64, 64)
learning_rate=1e-3
batch_size_advantage=2048
batch_size_strategy=2048
memory_capacity=1e6
policy_network_train_steps=5000
advantage_network_train_steps=750
reinitialize_advantage_networks=True
ignore_cards = None