import torch
import torch.nn as nn

class OneHotToCategoricalLayer(nn.Module):
    def __init__(self, cat_dims):
        """
        Args:
            input_indices (list of int): Indices of input features to use.
            functions (list of callables): Function objects with `.arity` and `__call__`.
            depth (int): Maximum depth of function composition.
        """
        super().__init__()
        self._cat_dims = cat_dims

    def forward(self, x):
        feature_chunks = torch.split(x, self._cat_dims, dim=1)
        categorical_indices = [torch.argmax(chunk, dim=1) for chunk in feature_chunks]
        return torch.stack(categorical_indices, dim=1)
    
# === Example Usage ===

if __name__ == "__main__":
    # Input tensor (batch_size, num_features)
    x = torch.tensor([
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    ])

    # Dimensions of each one-hot encoded category
    cat_dims = [2, 3, 2]

    # Construct the layer
    layer = OneHotToCategoricalLayer(cat_dims=cat_dims)

    # Forward pass
    out = layer(x)

    print("Output shape:", out.shape)
    print("Generated features:\n", out)
