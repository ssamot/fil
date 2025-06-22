game_name = "leduc_poker"
num_iterations = 100000
num_traversals = 10
log_frequency = 1
results_file_base = "../results/leduc_"
policy_network_layers=(64,)
advantage_network_layers=(64,)
learning_rate=1e-3
batch_size_advantage=2048
batch_size_strategy=2048
memory_capacity=1e6
policy_network_train_steps=1
advantage_network_train_steps=10
reinitialize_advantage_networks=False
cat_split_dims=[2,6,6,2,2,2,2,2,2,2,2]
cat_real_dims=[2,6,7,3,3,3,3,3,3,3,3]
# fil_groups=[
#     [(2, 3), 2],
#     [(2, 4), 2],
#     [(2, 5), 2],
#     [(2, 6), 2],
#     [(2, 7), 2],
#     [(2, 8), 2],
#     [(2, 9), 2],
#     [(2, 10), 2],
#
#     [(1, 3), 2],
#     [(1, 4), 2],
#     [(1, 5), 2],
#     [(1, 6), 2],
#     [(1, 7), 2],
#     [(1, 8), 2],
#     [(1, 9), 2],
#     [(1, 10), 2],
#
#     [(1, 2), 2],
# ]


# fil_groups=[
# [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 2],
# [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1],
#
# ]
fil_groups=[
    [(0, 1, 2, 3, 4, 5, 6), 7],
    [(0, 1, 2, 7, 8, 9, 10), 7],
    [(0, 1, 2), 3],
    [(0, 1, 2), 2],
    [(0, 1, 2,), 1],

    # [(0, 1, 3), 3],
    # [(0, 1, 4), 3],
    # [(0, 1, 5), 3],
    # [(0, 1, 6), 3],
    # [(0, 1, 2, 7), 4],
    # [(0, 1, 2, 8), 4],
    # [(0, 1, 2, 9), 4],
    # [(0, 1, 2, 10), 4],
 ] # t

# this always uses hand and board with all the actions
# fil_groups=[
#     [(1, 3), 2],
#     [(1, 4), 2],
#     [(1, 5), 2],
#     [(1, 6), 2],
#     [(1, 2, 7), 3],
#     [(1, 2, 8), 3],
#     [(1, 2, 9), 3],
#     [(1, 2, 10), 3],
#     [(1, 2), 2],
#     [(0,), 1]
#   ] # this uses only hand in combination with the actions from the first round where the board card was not yet dealt