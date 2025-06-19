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
#fil_groups=[[((0, 2), (2, 3), (5, 2), (7, 2), (9, 2)), 2]]

fil_groups=[
    #[((0, 2), (5, 2)), 2],
    #[((0, 2), (7, 2)), 2],
    #[((0, 2), (9, 2)), 2],
    [((2, 3), (5, 2)), 2],
    [((2, 3), (7, 2)), 2],
    #[((2, 3), (9, 2)), 2],
    [((2, 3),), 1],
    #[((0, 2),), 1],
    #[((2, 3), (0, 2)), 2],
]


# fil_groups=[
#
#     [((2, 3), (5, 2), (7, 2), (9, 2)), 4],
#     [((2, 3),), 1],
#
# ]