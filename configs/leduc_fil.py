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
cat_split_dims=[2,6,6,2,2,2,2,2,2,2,2]
cat_real_dims=[2,6,7,3,3,3,3,3,3,3,3]
fil_groups=[
    [(1, 2, 3), 3],
    [(1, 2, 4), 3],
    [(1, 2, 5), 3],
    [(1, 2, 6), 3],
    [(1, 2, 7), 3],
    [(1, 2, 8), 3],
    [(1, 2, 9), 3],
    [(1, 2, 10), 3],
    [(1, 2), 2],
] # this always uses hand and board with all the actions
# fil_groups=[
#     [(1, 3), 3],
#     [(1, 4), 3],
#     [(1, 5), 3],
#     [(1, 6), 3],
#     [(1, 2, 7), 3],
#     [(1, 2, 8), 3],
#     [(1, 2, 9), 3],
#     [(1, 2, 10), 3],
#     [(1, 2), 2],
# ] # this uses only hand in combination with the actions from the first round where the board card was not yet dealt