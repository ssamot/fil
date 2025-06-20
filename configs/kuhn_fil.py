game_name = "kuhn_poker"
num_iterations = 1000
num_traversals = 25
log_frequency = 1
results_file_base = "../results/kuhn_fil_"
policy_network_layers=(3,)
advantage_network_layers=(3,)
learning_rate=1e-3
batch_size_advantage=64
batch_size_strategy=64
memory_capacity=1e6
policy_network_train_steps=500
advantage_network_train_steps=50
reinitialize_advantage_networks=False
cat_dims=[2,3,2,2,2]
fil_groups=[
    [(1, 2), 2],
    [(1, 3), 2],
    [(1, 4), 2],
    [(1,), 1],
]