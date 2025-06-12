import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.keras import TqdmCallback

import keras
from keras import layers, ops, models
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_layer: keras.layers.Layer):
        self.feature_layer = feature_layer

    def fit(self, X, y=None):
        # No fitting required for this transformer
        return self

    def transform(self, X):
        # Convert input to keras tensor
        x_tensor = ops.convert_to_tensor(X, dtype="float32")
        out = self.feature_layer(x_tensor)
        return ops.convert_to_numpy(out)

class FeatureInteractionLayer(layers.Layer):
    def __init__(self, input_types, groups=None,  **kwargs):
        """
        Constructor for FeatureInteractionLayer. Takes the following parameters

        Parameters
        ----------
        input_types : list of tuples
            Specification of input variable types and their properties. Each tuple follows
            one of two patterns:

            - For categorical variables: ('cat', n_categories)
                * 'cat': str, literal indicating categorical variable type
                * n_categories: int, number of categories in the categorical variable

            - For continuous variables: ('cont', min_value, max_value, k_knots)
                * 'cont': str, literal indicating continuous variable type
                * min_value: float, minimum value in the continuous range
                * max_value: float, maximum value in the continuous range
                * k_knots: int, number of knots for spline interpolation

            Example:
                [('cat', 5), ('cont', 0.0, 100.0, 10), ('cat', 3)]

        groups : list of lists
            Grouping specification for variables with associated rank constraints.
            Each inner list contains:

            - Variable indices: tuple of int
                Indices of variables that belong to this group
            - max_rank: int
                Maximum rank constraint for this variable group

            Structure: [[(i, j, k, ...), max_rank], [(l, v, ...), max_rank], ...]

            Example:
                [[(0, 1, 2), 2], [(3, 4), 3]]
                # Group 1: variables at indices 0,1,2 with max rank 2
                # Group 2: variables at indices 3,4 with max rank 3

        Returns
        -------
        return_type
            Description of what the function returns

        Examples
        --------
        >>> input_types = [('cat', 4), ('cont', -10.0, 10.0, 8)]
        >>> groups = [[(0, 1), 3]]
        """
        super().__init__(**kwargs)
        self.input_types = input_types
        self.groups = groups or []

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        assert len(self.input_types) == self.input_dim, "input_types length mismatch"
        self.cat_embeddings = {}
        for i, t in enumerate(self.input_types):
            if t[0] == "cat":
                num_categories = t[1]
                if num_categories is None:
                    raise ValueError(f"Missing category count for categorical feature {i}")
                self.cat_embeddings[i] = layers.CategoryEncoding(num_tokens=num_categories, output_mode="one_hot")

        super().build(input_shape)

    def scale_to_minus_one_to_one(self, x, min_val, max_val):
        return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

    def encode_sigmoid(self, x, min_val, max_val, num_knots):
        x = self.encode_linear(x, min_val, max_val)
        knots = ops.linspace(0, 1, num_knots)  # shape (K,)
        diffs = x - knots[None, :]  # shape (B, K)
        sharpness = 10.0
        mask = ops.sigmoid(sharpness * diffs)  # shape (B, K)

        new_poly = ops.concatenate(
            [
                ops.multiply(x, mask),
                ops.multiply(ops.ones(shape=x.shape), mask)
            ],
            axis=-1)


        return new_poly

    def encode_linear(self, x, min_val, max_val):
        x = ops.clip(x, min_val, max_val)
        return (x - min_val) / (max_val - min_val + 1e-6)

    def encode_poly(self, x, min_val, max_val, degree=2):
        x = self.encode_linear(x, min_val, max_val)
        return ops.concatenate([x ** d for d in range(1, degree + 1)], axis=-1)

    def encode_inputs(self, x):
        encoded = []
        for i in range(self.input_dim):
            xi = x[:, i:i+1]
            if self.input_types[i][0] == "cont":
                #encoded.append(xi)
                _, min_val, max_val, k_knots = self.input_types[i]
                emb = self.encode_sigmoid(xi, min_val, max_val, k_knots)
            elif self.input_types[i][0] == "cat":
                emb = self.cat_embeddings[i](ops.cast(xi, "int32"))
            encoded.append(emb)
        return encoded  # list of tensors (B, D_i)

    def combine_features(self, tensors, max_order):
        combined = []
        for order in range(1, max_order + 1):
            for idxs in combinations(range(len(tensors)), order):
                # Outer product style multiplication
                feat = tensors[idxs[0]]
                for i in idxs[1:]:
                    feat = ops.expand_dims(feat, -1) * ops.expand_dims(tensors[i], -2)
                    feat = ops.reshape(feat, (feat.shape[0], -1))
                combined.append(feat)
        return combined

    def call(self, inputs):
        x = inputs
        encoded_inputs = self.encode_inputs(x)
        out_features = []

        # Grouped interactions
        used_indices = set()
        for group, max_order in self.groups:
            group_inputs = [encoded_inputs[i] for i in group]
            used_indices.update(group)
            interactions = self.combine_features(group_inputs, max_order)

            out_features.extend(interactions)

        # Remaining (ungrouped) inputs
        remaining = [i for i in range(self.input_dim) if i not in used_indices]
        if remaining:
            remaining_inputs = [encoded_inputs[i] for i in remaining]
            interactions = layers.concatenate(remaining_inputs, axis = -1)
            out_features.append(interactions)

        #exit()

        # Concatenate all features

        return ops.concatenate(out_features, axis=-1)
# --- Data generation ---
def generate_data(num_samples, n_binary_inputs, groups, group_ops):
    X = np.random.randint(0, 2, size=(num_samples, n_binary_inputs))
    y = []
    groups = [group[0] for group in groups]
    for x in X:
        vals = []
        for (g, op) in zip(groups, group_ops):
            a, b, c = x[g[0]], x[g[1]], x[g[2]]
            if op == 'and':
                vals.append(a & b & c)
            elif op == 'or':
                vals.append(a | b | c)
            elif op == 'xor':
                vals.append(a ^ b ^ c)
        y.append(np.array(vals).sum() >= 2)
    return X.astype("float32"), np.array(y, dtype=np.float32).reshape(-1, 1)

# --- Settings ---


if __name__ == '__main__':
    n_binary_inputs = 40
    groups = [[(0, 1, 2), 3], [(3, 4, 5), 3], [(6, 7, 8), 3], [(9, 10, 11), 3]]
    group_ops = ['and', 'and', 'or', 'xor']
    test_size = 0.2
    rank = 12

    # --- Generate data ---
    X_train, y_train = generate_data(1000, n_binary_inputs, groups, group_ops)

    X_test, y_test = generate_data(10000, n_binary_inputs, groups, group_ops)

    print(X_train.shape, y_train.shape)

    # --- MLP model ---
    inputs_mlp = layers.Input(shape=(n_binary_inputs,))
    x = layers.Dense(64, activation="relu")(inputs_mlp)
    x = layers.Dense(32, activation="relu")(x)
    outputs_mlp = layers.Dense(1)(x)
    mlp_model = models.Model(inputs=inputs_mlp, outputs=outputs_mlp)
    mlp_model.compile(optimizer="AdamW", loss="mse", metrics=["accuracy"])

    mlp_model.summary()

    mlp_model.fit(X_train, y_train, epochs=2000, batch_size=302, validation_split=0.1, verbose=0,callbacks = [TqdmCallback()])



    # --- Spline interaction model ---
    inputs_int = layers.Input(shape=(n_binary_inputs,))
    x = FeatureInteractionLayer( input_types=[("cat", 2)] * n_binary_inputs,
                                     groups=groups)(inputs_int)

    x = layers.Dense(1)(x)

    spline_model = models.Model(inputs=inputs_int, outputs=x)
    spline_model.compile(optimizer="AdamW", loss="mse", metrics=["accuracy"])

    spline_model.summary()

    spline_model.fit(X_train, y_train,
                     epochs=2000, batch_size=302, validation_split=0.1, verbose=0, callbacks = [TqdmCallback()])



    # --- Evaluation ---
    mlp_preds = mlp_model.predict(X_test) > 0.5
    spline_preds = spline_model.predict(X_test) > 0.5
    mlp_acc = accuracy_score(y_test, mlp_preds)
    spline_acc = accuracy_score(y_test, spline_preds)

    print(f"MLP accuracy: {mlp_acc:.4f}")
    print(f"Feature Interaction Model accuracy: {spline_acc:.4f}")
