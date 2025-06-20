import torch
import numpy as np
from fil.fil_main import generate_data
from fil.fil_torch_ohi import FeatureInteractionLayer as OneHotFil
from fil.fil_torch import FeatureInteractionLayer as NormalFil
from fil.one_hot_to_cat import OneHotToCategoricalLayer

if __name__ == "__main__":
    n_binary_inputs = 12
    groups_generate = [[(0, 1, 2), 3], [(3, 4, 5), 3], [(6, 7, 8), 3], [(9, 10, 11), 3]]
    group_ops = ['and', 'and', 'or', 'xor']
    groups_oh = [[((0, 2), (2, 2), (4, 2)), 3], [((6, 2), (8, 2), (10, 2)), 3], [((12, 2), (14, 2), (16, 2)), 3], [((18, 2), (20, 2), (22, 2)), 3]]
    test_size = 0.2
    rank = 12

    # --- Generate data ---
    X_train, y_train = generate_data(1000, n_binary_inputs, groups_generate, group_ops)
    X_train_oh = np.array([a.flatten() for a in (np.eye(2)[np.asarray(X_train, dtype=np.int32)])])

    X_train_oh = torch.Tensor(X_train_oh)

    one_hot_fil = OneHotFil(groups_oh)

    one_hot_to_cat = OneHotToCategoricalLayer([2]*12)
    fil = NormalFil([2]*12, groups_generate)

    one_hot_fil_result = one_hot_fil(X_train_oh)
    fil_result = fil(one_hot_to_cat(X_train_oh))

    assert torch.equal(one_hot_fil_result, fil_result)