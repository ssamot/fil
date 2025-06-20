import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from functools import reduce
from sklearn.metrics import accuracy_score
import numpy as np
import random
from fil.fil_main import generate_data
from tqdm import tqdm

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class FeatureInteractionLayer(nn.Module):
    def __init__(self, cat_dims, groups):
        super().__init__()
        self.cat_dims = cat_dims
        self.groups = groups
        self.total_output_dim = 0
        self.lookup_params = []  # stores (input_idxs, radix_multipliers, dim, offset)

        for group, max_order in groups:
            for combo in itertools.combinations(range(len(group)), max_order):
                idxs = [group[i] for i in combo]
                sizes = [cat_dims[i] for i in idxs]
                dim = int(np.prod(sizes))
                radix = [int(np.prod(sizes[i + 1:])) for i in range(len(sizes))]
                offset = self.total_output_dim
                self.lookup_params.append((idxs, radix, dim, offset))
                self.total_output_dim += dim

    def forward(self, x_cat):
        B = x_cat.size(0)
        out = torch.zeros(B, self.total_output_dim, device=x_cat.device)

        for input_idxs, radix, dim, offset in self.lookup_params:
            x = x_cat[:, input_idxs]  # (B, r)
            radix_tensor = torch.tensor(radix, device=x_cat.device).unsqueeze(0)
            idx = (x * radix_tensor).sum(dim=1)  # (B,)
            out.scatter_(1, (idx + offset).unsqueeze(1), 1.0)

        return out

class CategoricalInteractionModel(nn.Module):
    def __init__(self, cat_dims, groups):
        super().__init__()
        self.encoder = FeatureInteractionLayer(cat_dims, groups)

        # Compute feature dim from a dummy input
        dummy = torch.zeros((1, len(cat_dims)), dtype=torch.long)
        with torch.no_grad():
            out_dim = self.encoder(dummy).shape[1]

        self.linear = nn.Linear(out_dim, 1)

    def forward(self, x_cat):
        x_feat = self.encoder(x_cat)
        return self.linear(x_feat)


if __name__ == '__main__':

    # Fix random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)



    n_binary_inputs = 12
    groups = [[(0, 1, 2), 3], [(3, 4, 5), 2], [(6, 7, 8), 2], [(9, 10, 11), 3]]
    group_ops = ['and', 'and', 'or', 'xor']
    test_size = 0.2
    rank = 12

    # --- Generate data ---
    X_train, y_train = generate_data(200, n_binary_inputs, groups, group_ops)

    X_test, y_test = generate_data(10000, n_binary_inputs, groups, group_ops)

    X_train = torch.tensor(X_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # print(y_test.shape, y_train.shape)
    # exit()


    model = CategoricalInteractionModel(cat_dims=[2]*n_binary_inputs, groups=groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    print(f"Total parameters: {count_parameters(model)[0]}")
    loss_fn = nn.MSELoss()

    total = 1000
    with tqdm(total=total) as pbar:
        for epoch in (range(total )):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_test), y_test)

            spline_preds = model(X_test).detach(). numpy() > 0.5

            spline_acc = accuracy_score(y_test, spline_preds)
            pbar.update(1)

            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix({"acc" :f"{spline_acc:.4f}"})
