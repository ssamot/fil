import torch
import torch.nn as nn
import itertools


# === Function Wrappers with Arity and Names ===

class Add:
    arity = 2
    def __call__(self, x, y):
        return x + y
    def __repr__(self):
        return "add"

class Mul:
    arity = 2
    def __call__(self, x, y):
        return x * y
    def __repr__(self):
        return "mul"

class Min:
    arity = 2
    def __call__(self, x, y):
        return torch.min(x, y)
    def __repr__(self):
        return "min"

class Max:
    arity = 2
    def __call__(self, x, y):
        return torch.max(x, y)
    def __repr__(self):
        return "max"

class SoftIfThenElse:
    arity = 3
    def __call__(self, cond, x, y):
        w = torch.sigmoid(cond)
        return w * x + (1 - w) * y
    def __repr__(self):
        return "soft_if_then_else"


class Avg:
    arity = 2
    def __call__(self, x, y):
        return (x + y) / 2
    def __repr__(self):
        return "avg"
    
def string_to_function(name):
    if name == repr(Add()):
        return Add()
    elif name == repr(Mul()):
        return Mul()
    elif name == repr(Min()):
        return Min()
    elif name == repr(Max()):
        return Max()
    elif name == repr(SoftIfThenElse()):
        return SoftIfThenElse()
    elif name == repr(Avg()):
        return Avg()
    else:
        raise ValueError("Function name " + name + " is not supported.")

# class SoftMajorityVote:
#     arity = 3
#     def __call__(self, x, y, z):
#         probs = torch.sigmoid(torch.stack([x, y, z], dim=-1))
#         vote = probs.sum(dim=-1, keepdim=True) / 3.0  # shape (B, 1)
#         return vote  # Values between 0 and 1
#     def __repr__(self):
#         return "soft_majority_vote"

class SoftArgMax3:
    arity = 3
    def __call__(self, x, y, z):
        stacked = torch.stack([x, y, z], dim=-1)  # shape (B, 1, 3)
        weights = torch.softmax(stacked, dim=-1)
        result = (stacked * weights).sum(dim=-1, keepdim=True)  # shape (B, 1)
        return result
    def __repr__(self):
        return "soft_argmax3"


# class Square:
#     arity = 1
#     def __call__(self, x):
#         return x ** 2
#     def __repr__(self):
#         return "square"


# === Compositional Feature Layer ===

class CompositionalFeatureLayer(nn.Module):
    def __init__(self, input_indices, functions, depth=2):
        """
        Args:
            input_indices (list of int): Indices of input features to use.
            functions (list of callables): Function objects with `.arity` and `__call__`.
            depth (int): Maximum depth of function composition.
        """
        super().__init__()

        if isinstance(functions[0], str):
            functions = [string_to_function(s) for s in functions]

        self.input_indices = input_indices
        self.functions = functions
        self.depth = depth
        self._compositions = []
        self.feature_descriptions = []

        self._build_compositions()

    def _build_compositions(self):
        base_features = [(lambda x, i=i: x[:, [i]], f"x{i}") for i in self.input_indices]
        all_by_depth = {0: base_features}

        for d in range(1, self.depth + 1):
            current = []
            prev = sum([all_by_depth[i] for i in range(d)], [])

            for fn in self.functions:
                arity = fn.arity
                for inputs in itertools.combinations(prev, arity):
                    try:
                        funcs, descs = zip(*inputs)
                        def composed_fn(x, fn=fn, funcs=funcs):  # capture defaults
                            args = [f(x) for f in funcs]
                            return fn(*args)
                        composed_name = f"{fn}(" + ",".join(descs) + ")"
                        current.append((composed_fn, composed_name))
                    except Exception as e:
                        print(f"Skipping {fn} on {descs} due to error: {e}")

            all_by_depth[d] = current

        self._compositions = sum(all_by_depth.values(), [])
        self.feature_descriptions = [desc for _, desc in self._compositions]

    def forward(self, x):
        features = [fn(x) for fn, _ in self._compositions]
        return torch.cat(features, dim=1) if features else torch.zeros((x.size(0), 0), device=x.device)


# === Example Usage ===

if __name__ == "__main__":
    # Input tensor (batch_size, num_features)
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    # Define function objects
    functions = [Add(), Mul(), Min(), Max(), SoftIfThenElse()]

    # Construct the layer
    layer = CompositionalFeatureLayer(input_indices=[0, 1], functions=functions, depth=2)

    # Forward pass
    out = layer(x)

    print("Output shape:", out.shape)
    print("Generated features:\n", out)

    # Print feature names
    print("\nFeature descriptions:")
    for desc in layer.feature_descriptions:
        print(desc)
