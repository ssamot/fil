import torch
import torch.nn as nn

import random
from fil.compositional import Min, Max, Mul, string_to_function
import torch
import torch.nn as nn
import random

class SampledCompositionalFeatureLayer(nn.Module):
    def __init__(self, input_indices, functions, depth, n_features, seed=None):
        super().__init__()
        if isinstance(functions[0], str):
            functions = [string_to_function(s) for s in functions]

        self.input_indices = input_indices
        self.functions = functions
        self.depth = depth
        self.n_features = n_features
        self.seed = seed
        self._rng = random.Random(seed)

        self.feature_expressions = []  # stores tuples like (func, [args]) or input index
        self.feature_names = []        # human-readable names
        self._build_features()

    def _sample_expr(self, depth):
        if depth == 0 or self._rng.random() < 0.5:
            # Base input feature
            idx = self._rng.choice(self.input_indices)
            return idx, f"x{idx}"
        else:
            func = self._rng.choice(self.functions)
            args = [self._sample_expr(depth - 1) for _ in range(func.arity)]
            expr = (func, [arg[0] for arg in args])
            name = f"{func}(" + ",".join(arg[1] for arg in args) + ")"
            return expr, name

    def _build_features(self):
        seen = set()
        attempts = 0
        max_attempts = self.n_features * 10

        while len(self.feature_expressions) < self.n_features and attempts < max_attempts:
            expr, name = self._sample_expr(self.depth)
            if name not in seen:
                self.feature_expressions.append(expr)
                self.feature_names.append(name)
                seen.add(name)
            attempts += 1

    def forward(self, x):
        outputs = []

        for expr in self.feature_expressions:
            feat = self._eval_expr(expr, x)
            outputs.append(feat.unsqueeze(-1))

        return torch.cat(outputs, dim=1)

    def _eval_expr(self, expr, x):
        if isinstance(expr, int):
            return x[:, expr]
        else:
            func, args = expr
            evaluated_args = [self._eval_expr(arg, x) for arg in args]
            return func(*evaluated_args)

    def extra_repr(self):
        return "\n".join(f"{i}: {name}" for i, name in enumerate(self.feature_names))


if __name__ == "__main__":
    # Input tensor (batch_size, num_features)
    x = torch.tensor([
        [1.0, 2.0, 3.0, 5.0, 6.0, ],
        [4.0, 5.0, 6.0, 5.0, 6.0]
    ])

    # Define function objects
    functions = [Mul(), Max()]

    # Construct the layer
    layer = SampledCompositionalFeatureLayer(input_indices=[0, 1, 2,3,4], functions=functions, n_features=32, depth=5)

    # Forward pass
    out = layer(x)

    print("Output shape:", out.shape)
    print(layer)
    #print("Generated features:\n", out)
