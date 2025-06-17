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
    def __init__(self, cat_dims, groups, layers):
        super().__init__()
        self.cat_dims = cat_dims
        self.groups = groups
        self.total_output_dim = 0
        self.lookup_params = []  # stores (input_idxs, radix_multipliers, dim, offset)
        self.group_layers = nn.ModuleList()
        self.group_splits = []

        for group, max_order in groups:
            group_output_dim = 0
            for r in range(1, max_order + 1):
                for combo in itertools.combinations(range(len(group)), r):
                    idxs = [group[i] for i in combo]
                    sizes = [cat_dims[i] for i in idxs]
                    dim = int(np.prod(sizes))
                    radix = [int(np.prod(sizes[i + 1:])) for i in range(len(sizes))]
                    offset = self.total_output_dim
                    self.lookup_params.append((idxs, radix, dim, offset))
                    self.total_output_dim += dim
                    group_output_dim += dim
            input_size = group_output_dim
            group_layers = []
            for layer_dim in layers:
                output_size = group_output_dim if layer_dim == -1 else layer_dim                
                group_layers.append(nn.Sequential(nn.Linear(input_size, output_size), nn.LeakyReLU()))
                input_size = output_size
            self.group_layers.append(nn.ModuleList(group_layers))
            self.group_splits.append(group_output_dim)

    def forward(self, x_oh):
        x_cat = self.onehot_to_categorical_vectorized(x_oh)
        B = x_cat.size(0)
        out = torch.zeros(B, self.total_output_dim, device=x_cat.device)

        for input_idxs, radix, dim, offset in self.lookup_params:
            x = x_cat[:, input_idxs]  # (B, r)
            radix_tensor = torch.tensor(radix, device=x_cat.device).unsqueeze(0)
            idx = (x * radix_tensor).sum(dim=1)  # (B,)
            out.scatter_(1, (idx + offset).unsqueeze(1), 1.0)
        splits = torch.split(out, self.group_splits, dim=1)
        out_splits = []
        for split, group_layers in zip(splits, self.group_layers):
            for layer in group_layers:
                split = layer(split)
            out_splits.append(split)
        output = torch.cat(out_splits, dim=1)
        return output
    
    def onehot_to_categorical_vectorized(self, x_onehot):
        feature_chunks = torch.split(x_onehot, self.cat_dims, dim=1)
        categorical_indices = [torch.argmax(chunk, dim=1) for chunk in feature_chunks]
        return torch.stack(categorical_indices, dim=1)
    
    def reset(self):
        for group_layers in self.group_layers:
            for layer in group_layers:
                for sublayer in layer:
                    if hasattr(sublayer, 'reset_parameters'):
                        sublayer.reset()

class CategoricalInteractionModel(nn.Module):
    def __init__(self, cat_dims, groups, layers):
        super().__init__()
        self.encoder = FeatureInteractionLayer(cat_dims, groups, layers)

        # Compute feature dim from a dummy input
        dummy = torch.zeros((1, len(cat_dims*2)), dtype=torch.long)
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
    groups_generate = [[(0, 1, 2), 3], [(3, 4, 5), 3], [(6, 7, 8), 3], [(9, 10, 11), 3]]
    group_ops = ['and', 'and', 'or', 'xor']
    test_size = 0.2
    rank = 12

    # --- Generate data ---
    X_train, y_train = generate_data(1000, n_binary_inputs, groups_generate, group_ops)
    X_train_oh = [a.flatten() for a in (np.eye(2)[np.asarray(X_train, dtype=np.int32)])]

    X_test, y_test = generate_data(10000, n_binary_inputs, groups_generate, group_ops)
    X_test_oh = [a.flatten() for a in (np.eye(2)[np.asarray(X_test, dtype=np.int32)])]

    X_train = torch.tensor(X_train_oh, dtype=torch.long)
    X_test = torch.tensor(X_test_oh, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    print(X_test.shape)
    # print(y_test.shape, y_train.shape)
    # exit()


    model = CategoricalInteractionModel(cat_dims=[2]*n_binary_inputs, groups=groups_generate, layers=[-1])
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
