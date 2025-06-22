import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from functools import reduce

from sklearn.metrics import accuracy_score
import numpy as np
import random
from fil_main import generate_data
from tqdm import tqdm

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class FeatureInteractionLayer(nn.Module):
    def __init__(self, cat_dims, groups, emb_dim=8):
        """
        cat_dims: list of int — cardinality of each categorical feature
        groups: list of (list of feature indices) — groups of features to interact
        emb_dim: int — embedding dimension for each feature
        """
        super().__init__()
        self.cat_dims = cat_dims
        self.groups = groups
        self.emb_dim = emb_dim

        # One embedding per categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=emb_dim)
            for card in cat_dims
        ])

        # One scalar projection per feature
        self.scalar_projections = nn.ModuleList([
            nn.Linear(emb_dim, 1) for _ in cat_dims
        ])

        # For tracking output dim
        self.total_output_dim = len(groups) * 2  # max + product per group

    def forward(self, x_cat):
        """
        x_cat: LongTensor of shape (B, num_cat_features)
        Returns: Tensor of shape (B, total_output_dim)
        """
        B = x_cat.size(0)
        group_outputs = []

        for group, _ in self.groups:
            # Get embeddings and project each to scalar
            projected_scalars = []
            for idx in group:
                emb = self.embeddings[idx](x_cat[:, idx])         # (B, emb_dim)
                score = self.scalar_projections[idx](emb)         # (B, 1)
                projected_scalars.append(score)

            group_tensor = torch.cat(projected_scalars, dim=1)    # (B, group_size)

            # Compute max and product over the group
            max_val = group_tensor.max(dim=1, keepdim=True).values     # (B, 1)
            prod_val = group_tensor.prod(dim=1, keepdim=True)          # (B, 1)

            group_outputs.append(max_val)
            group_outputs.append(prod_val)

        out = torch.cat(group_outputs, dim=1)  # (B, total_output_dim)
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
