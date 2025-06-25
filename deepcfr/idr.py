import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, Ridge
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from collections import defaultdict, Counter
import lightgbm as lgb
import shap

# --- Synthetic data ---
np.random.seed(0)
n_samples, n_features = 5000, 20
X = np.random.randint(0, 2, size=(n_samples, n_features))
X = np.random.randn(n_samples, n_features)
y = X[:, 1] * X[:, 2] + X[:, 3] * X[:, 4] + X[:, 0] * X[:, 2] * X[:, 3] + 0.1 * np.random.randn(n_samples)

# --- Split into train/val/test ---
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

reg_linear = Ridge
class InteractionDiscoveryRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, categorical_indices=None, encoding="onehot", coef_threshold=1e-5, max_iter=10, top_k=10, max_order=3, method="tree"):
        self.categorical_indices = categorical_indices if categorical_indices is not None else []
        self.encoding = encoding
        self.coef_threshold = coef_threshold
        self.max_iter = max_iter
        self.top_k = top_k
        self.max_order = max_order
        self.method = method

    def _encode(self, X):
        if self.encoding == "onehot":
            self.encoder = ColumnTransformer([
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), self.categorical_indices)
            ], remainder='passthrough')
        elif self.encoding == "target":
            raise NotImplementedError("Target encoding is not implemented in this snippet.")
        return self.encoder.fit_transform(X)

    def _transform_new(self, X):
        return self.encoder.transform(X) if self.encoding == "onehot" else X

    def _extract_tree_interactions(self, model, X):
        interactions = Counter()
        booster = model.booster_
        for tree in booster.dump_model()["tree_info"]:
            splits = tree["tree_structure"]
            stack = [splits]
            features_in_tree = set()
            while stack:
                node = stack.pop()
                if "split_feature" in node:
                    features_in_tree.add(node["split_feature"])
                    stack.append(node["left_child"])
                    stack.append(node["right_child"])
            for pair in self._get_combinations(features_in_tree):
                interactions[pair] += 1
        return [pair for pair, _ in interactions.most_common(self.top_k)]

    def _extract_shap_interactions(self, model, X):
        explainer = shap.TreeExplainer(model)
        shap_interactions = explainer.shap_interaction_values(X)
        pair_scores = defaultdict(float)
        n_features = X.shape[1]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                score = np.abs(shap_interactions[:, i, j]).mean()
                if score > 0:
                    pair_scores[(i, j)] = score
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: -x[1])[:self.top_k]
        return [tuple(sorted(pair)) for pair, _ in sorted_pairs]

    def _get_combinations(self, features):
        from itertools import combinations
        combs = []
        for r in range(2, min(self.max_order, len(features)) + 1):
            combs.extend(combinations(sorted(features), r))
        return combs

    def _create_interactions(self, X, origins):
        return np.column_stack([np.prod(X[:, list(o)], axis=1) for o in origins])

    def _print_feature_names(self, origins):
        print("New features added:")
        for origin in origins:
            print(" * ".join(f"x_{i}" for i in origin))

    def fit(self, X, y, sample_weight = None):
        X = np.asarray(X)
        X_enc = self._encode(X)
        n_features = X_enc.shape[1]
        self.feature_origins_ = [(i,) for i in range(n_features)]
        self.seen_interactions_ = set(self.feature_origins_)
        X_current = X_enc.copy()

        #print(X.shape[1])

        if((X.shape[0]) < 500):
            self.final_model_ = reg_linear()
            self.final_model_.fit(X, y, sample_weight)
            return self


        for iteration in range(self.max_iter):
            #print(f"\n--- Iteration {iteration+1} ---")
            lasso = reg_linear()
            lasso.fit(X_current, y, sample_weight)
            y_pred = lasso.predict(X_current)
            r2 = r2_score(y, y_pred)
            #print(f"Lasso R^2: {r2:.4f}")

            residuals = y - y_pred
            model = lgb.LGBMRegressor(verbose=-1, n_jobs =20, n_estimators=1000)
            model.fit(X_current, residuals, sample_weight)

            if self.method == "shap":
                new_interactions = self._extract_shap_interactions(model, X_current)
            elif self.method == "tree":
                new_interactions = self._extract_tree_interactions(model, X_current)
            else:
                raise ValueError("Unknown method for interaction extraction.")

            final_interactions = []
            for combo in new_interactions:
                origin_combo = tuple(sorted(set().union(*(self.feature_origins_[i] for i in combo))))
                if origin_combo not in self.seen_interactions_:
                    self.seen_interactions_.add(origin_combo)
                    final_interactions.append(origin_combo)

            if not final_interactions:
                print("No new unique interactions. Stopping.")
                break

            #self._print_feature_names(final_interactions)
            new_X_feats = self._create_interactions(X_enc, final_interactions)
            X_current = np.hstack([X_current, new_X_feats])
            self.feature_origins_ += final_interactions

        print("\n--- Final Lasso Fit ---")
        all_feats = self._create_interactions(X_enc, self.feature_origins_)
        self.final_model_ = reg_linear()
        self.final_model_.fit(all_feats, y, sample_weight)

        nz_mask = np.abs(self.final_model_.coef_) > self.coef_threshold
        print("\nSelected features and weights:")
        for origin, coef, keep in zip(self.feature_origins_, self.final_model_.coef_, nz_mask):
            if keep:
                name = " * ".join(f"x_{i}" for i in origin)
                print(f"{name}: {coef:.4f}")

        self.selected_origins_ = [o for o, keep in zip(self.feature_origins_, nz_mask) if keep]
        return self

    def _create_all_features(self, X):
        return self._create_interactions(X, self.feature_origins_)

    def transform(self, X):
        X = self._transform_new(X)
        return self._create_interactions(X, self.feature_origins_)

    def predict(self, X):
        return self.final_model_.predict(self.transform(X))

# --- Run the pipeline ---
if __name__ == "__main__":
    # reg = InteractionDiscoveryRegressor(categorical_indices=[], encoding="onehot", alpha=0.01)
    # reg.fit(X_train, y_train)
    # y_pred = reg.predict(X_test)
    # print("\n--- Final Test R^2 ---")
    # print(f"R^2: {r2_score(y_test, y_pred):.4f}")
    #
    # --- New Synthetic Dataset with Categorical Features ---
    np.random.seed(42)
    n_samples = 5050
    n_cat = 3  # number of categorical features
    n_cont = 5  # number of continuous features

    # Categorical features: values in {0, 1, 2}
    X_cat = np.random.randint(0, 3, size=(n_samples, n_cat))

    # Continuous features
    X_cont = np.random.randn(n_samples, n_cont)

    # Combine
    X_full = np.hstack([X_cat, X_cont])

    # Ground truth target with interaction between categorical and continuous
    # Example: interaction between cat_0 == 2 and cont_1 + interaction between cat_1 * cat_2
    y = ((X_cat[:, 0] == 2).astype(float)) * X_cont[:, 1] + X_cat[:, 1] * X_cat[:, 2] + 0.1 * np.random.randn(n_samples)

    # --- Split ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_full, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

    # --- Fit the model ---
    reg = InteractionDiscoveryRegressor(categorical_indices=[0, 1, 2], encoding="onehot", coef_threshold=0.05)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("\n--- Final Test R^2 with Categorical Data ---")
    print(f"R^2: {r2_score(y_test, y_pred):.4f}")

