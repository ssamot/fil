import torch
import numpy as np
from fil.fil_main import generate_data
from fil.fil_torch_ohi import FeatureInteractionLayer as OneHotFil
from fil.fil_torch import FeatureInteractionLayer as NormalFil
from fil.one_hot_to_cat import OneHotToCategoricalLayer

import unittest

class TestOneHotFilComparedToOneHotConverterPlusNormalFil(unittest.TestCase):

    def test_output_comparison(self):
        n_binary_inputs = 12
        groups_generate = [[(0, 1, 2), 3], [(3, 4, 5), 3], [(6, 7, 8), 3], [(9, 10, 11), 3]]
        group_ops = ['and', 'and', 'or', 'xor']
        groups_oh = [[((0, 2), (2, 2), (4, 2)), 3], [((6, 2), (8, 2), (10, 2)), 3], [((12, 2), (14, 2), (16, 2)), 3], [((18, 2), (20, 2), (22, 2)), 3]]

        # --- Generate data ---
        X_train, _ = generate_data(1000, n_binary_inputs, groups_generate, group_ops)
        X_train_oh = np.array([a.flatten() for a in (np.eye(2)[np.asarray(X_train, dtype=np.int32)])])

        X_train_oh = torch.Tensor(X_train_oh)

        one_hot_fil = OneHotFil(groups_oh)

        one_hot_to_cat = OneHotToCategoricalLayer([2]*12)
        fil = NormalFil([2]*12, groups_generate)

        one_hot_fil_result = one_hot_fil(X_train_oh)
        fil_result = fil(one_hot_to_cat(X_train_oh))

        self.assertTrue(torch.equal(one_hot_fil_result, fil_result))

    def test_output_comparison_kuhn(self):
        cat_dims=[2,3,2,2,2]
        fil_groups=[
            [(1, 2), 2],
            [(1, 3), 2],
            [(1, 4), 2],
            [(1,), 1],
        ]

        fil_groups_oh=[
            [((2, 3), (5, 2)), 2],
            [((2, 3), (7, 2)), 2],
            [((2, 3), (9, 2)), 2],
            [((2, 3),), 1],
        ]

        X_train_oh = [
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        ]

        X_train_oh = torch.Tensor(np.array(X_train_oh))

        one_hot_fil = OneHotFil(fil_groups_oh)

        one_hot_to_cat = OneHotToCategoricalLayer(cat_dims)
        fil = NormalFil(cat_dims, fil_groups)

        one_hot_fil_result = one_hot_fil(X_train_oh)
        fil_result = fil(one_hot_to_cat(X_train_oh))

        self.assertTrue(torch.equal(one_hot_fil_result, fil_result))


if __name__ == '__main__':
    unittest.main()