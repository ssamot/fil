game_name = "kuhn_poker"
num_iterations = 10000
num_traversals = 1
log_frequency = 1
results_file_base = "../results/kuhn_fil_"
policy_network_layers=(3,)
advantage_network_layers=(3,)
learning_rate=1e-3
batch_size_advantage=256
batch_size_strategy=256
memory_capacity=1e6
policy_network_train_steps=1
advantage_network_train_steps=10
reinitialize_advantage_networks=False
cat_split_dims=[2,3,2,2,2]
cat_real_dims=[2,3,3,3,3]
fil_groups=[
    [(1, 2), 2],
    [(1, 3), 2],
    [(1, 4), 2],
    [(0, 1,), 2],
    [(1,), 1],
]