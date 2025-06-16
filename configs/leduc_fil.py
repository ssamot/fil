game_name = "leduc_poker"
num_iterations = 1000
num_traversals = 15
log_frequency = 1
results_file_base = "../results/leduc_"
policy_network_layers=(64,)
advantage_network_layers=(64,)
learning_rate=1e-3
batch_size_advantage=128
batch_size_strategy=128
memory_capacity=1e6
policy_network_train_steps=10
advantage_network_train_steps=100
reinitialize_advantage_networks=False
#fil_groups=[[((0, 2), (2, 6), (8, 6)), 3], [((14, 2), (16, 2), (18, 2), (20, 2)), 3], [((22, 2), (24, 2), (26, 2), (28, 2)), 3]] # splits to (player, card, card), (first betting round), (second betting round)
#fil_groups=[[((0, 2), (2, 6), (8, 6),(14, 2), (16, 2), (18, 2), (20, 2), (22, 2), (24, 2), (26, 2), (28, 2)), 3]] # splits to (player, card, card), (first betting round), (second betting round)
fil_groups=[[((0, 2), (2, 6), (8, 6)), 2],
            [((14, 2), (16, 2), (18, 2), (20, 2)), 2],
            [((22, 2), (24, 2), (26, 2), (28, 2)), 2]] # splits to (player, card, card), (first betting round), (second betting round)

# fil_groups=[[(0, 2, 8), 3], [(14, 16, 18, 20, 22, 24, 26, 28), 3]] # betting rounds together