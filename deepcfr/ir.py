import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, OrthogonalMatchingPursuitCV, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from itertools import combinations, product

# --- Data Generation ---
def generate_data(n_samples=1000, n_features=6):
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    y = (
        (1 - X[:, 1]) * X[:, 2]
        + 2 * X[:, 3] * X[:, 4]
        - 3.0 * X[:, 0] * X[:, 2] * X[:, 3]
        + 0.1 * np.random.randn(n_samples)
    )
    flip_mask = np.random.rand(n_samples) < 0.3
    y[flip_mask] = -y[flip_mask]
    return X, y

# --- Extract conjunctions ---
def extract_conjunctions_from_sklearn_tree(tree, feature_names):
    conjunctions = []
    def recurse(node_id=0, path=[]):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            conjunctions.append(frozenset(path))
            return
        feat_idx = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        feat_name = feature_names[feat_idx]
        if thresh <= 0.5:
            left_cond = (feat_name, 0)
            right_cond = (feat_name, 1)
        else:
            left_cond = (feat_name, "<=")
            right_cond = (feat_name, ">")
        recurse(tree.children_left[node_id], path + [left_cond])
        recurse(tree.children_right[node_id], path + [right_cond])
    recurse(0, [])
    return conjunctions

def extract_conjunctions_from_tree(tree, feature_names):
    def recurse(node, path):
        if 'leaf_index' in node:
            conjunctions.append(frozenset(path))
            return
        idx = node['split_feature']
        threshold = node['threshold']
        feat = feature_names[idx]
        if threshold <= 0.5:
            left_cond = (feat, 0)
            right_cond = (feat, 1)
        else:
            left_cond = (feat, "<=")
            right_cond = (feat, ">")
        recurse(node['left_child'], path + [left_cond])
        recurse(node['right_child'], path + [right_cond])
    conjunctions = []
    recurse(tree['tree_structure'], [])
    return conjunctions

def extract_all_conjunctions(model):
    conjunctions = []
    if hasattr(model, "booster_"):  # LightGBM
        booster = model.booster_
        model_dict = booster.dump_model()
        feature_names = booster.feature_name()
        for tree in model_dict['tree_info']:
            conjunctions.extend(extract_conjunctions_from_tree(tree, feature_names))
    elif hasattr(model, "estimators_"):  # RandomForest
        feature_names = [f"x_{i}" for i in range(model.n_features_in_)]
        for tree in model.estimators_:
            conjunctions.extend(extract_conjunctions_from_sklearn_tree(tree.tree_, feature_names))
    elif hasattr(model, "tree_"):  # DecisionTree
        feature_names = [f"x_{i}" for i in range(model.n_features_in_)]
        conjunctions.extend(extract_conjunctions_from_sklearn_tree(model.tree_, feature_names))
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    return conjunctions

# --- Feature transformation ---
def apply_conjunctions_binary_product(X, conjunctions):
    n = X.shape[0]
    feats = []
    for conj in conjunctions:
        prod = np.ones(n)
        skip = False
        for fstr, v in conj:
            idx = int(fstr.split('_')[1])
            if v == 1:
                prod *= X[:, idx]
            elif v == 0:
                prod *= (1 - X[:, idx])
            else:
                skip = True
                break
        if not skip:
            feats.append(prod)
    if feats:
        return np.stack(feats, axis=1)
    return np.zeros((n, 0))

def extract_interacting_feature_sets(conjunctions):
    sets = set()
    for conj in conjunctions:
        if all(v in (0, 1) for _, v in conj):
            features = frozenset(f for f, _ in conj)
            if len(features) > 1:
                sets.add(features)
    return sets

def apply_targeted_poly_features(X, interacting_sets):
    n = X.shape[0]
    feats = []
    for fset in interacting_sets:
        idxs = sorted([int(f.split('_')[1]) for f in fset])
        for k in range(2, len(idxs) + 1):
            for combo in combinations(idxs, k):
                feats.append(np.prod(X[:, combo], axis=1))
    if feats:
        return np.stack(feats, axis=1)
    return np.zeros((n, 0))

def apply_signed_interaction_features(X, interacting_sets):
    n = X.shape[0]
    feats = []
    for fset in interacting_sets:
        idxs = sorted([int(f.split('_')[1]) for f in fset])
        for signs in product([1, -1], repeat=len(idxs)):
            prod = np.ones(n)
            for i, s in zip(idxs, signs):
                if s == 1:
                    prod *= X[:, i]
                else:
                    prod *= (1 - X[:, i])
            feats.append(prod)
    if feats:
        return np.stack(feats, axis=1)
    return np.zeros((n, 0))

class InteractionFeatureRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model='random_forest', interaction_type='poly', verbose=1):
        self.base_model = base_model
        self.interaction_type = interaction_type
        self.verbose = verbose

    def fit(self, X, y, sample_weight = None):
        X = np.array(X)
        self.X_ = X
        self.y_ = y
        if X.shape[0] < 100:
            if self.verbose:
                print("[SKIP] Not enough samples, using RidgeCV")
            self.model_ = RidgeCV()
            self.model_.fit(X, y)
            return self

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.base_model == 'random_forest':
                self.tree_model_ = RandomForestRegressor(n_estimators=100, max_depth=4)
                self.tree_model_.fit(X, y)
            elif self.base_model == 'decision_tree':
                temp_tree = DecisionTreeRegressor(random_state=0)
                path = temp_tree.cost_complexity_pruning_path(X, y)
                alphas = path.ccp_alphas[:-1]
                alphas = [max(alpha, 0.0) for alpha in alphas]
                param_grid = {
                           #   'ccp_alpha': alphas,
                    "criterion" : ["squared_error", "friedman_mse", "absolute_error"],
                              'min_samples_leaf':[1,2,3,4, 5,10,20,30, 50, 100],
                    'max_depth': list(range(2, 4))

                }
                search = GridSearchCV(
                    DecisionTreeRegressor(random_state=0),
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=KFold(n_splits=5, shuffle=True, random_state=0),
                    error_score='raise',
                    n_jobs=-1
                )
                search.fit(X, y)
                self.tree_model_ = search.best_estimator_
                if self.verbose:
                    print(f"[INFO] Best params: {search.best_params_}")
            else:
                raise ValueError("Unsupported base_model")

        conjunctions = extract_all_conjunctions(self.tree_model_)
        interacting_sets = extract_interacting_feature_sets(conjunctions)

        if self.interaction_type == 'conjunctions':
            X_new = apply_conjunctions_binary_product(X, conjunctions)
        elif self.interaction_type == 'poly':
            #print(interacting_sets)
            X_new = apply_targeted_poly_features(X, interacting_sets)
        elif self.interaction_type == 'signed':
            X_new = apply_signed_interaction_features(X, interacting_sets)
        else:
            raise ValueError("Unknown interaction_type")

        X_combined = np.hstack([X, X_new])

        if self.verbose:
            print(f"Fitting with {X_combined.shape[1]} features on {X_combined.shape[0]} samples using method: {self.interaction_type}")

        self.model_ = OrthogonalMatchingPursuitCV()
        #self.model_ = Ridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X_combined, y)

        self.X_new_ = X_combined
        return self

    def predict(self, X):
        X = np.array(X)
        if hasattr(self, 'tree_model_'):
            conjunctions = extract_all_conjunctions(self.tree_model_)
            interacting_sets = extract_interacting_feature_sets(conjunctions)
            if self.interaction_type == 'conjunctions':
                X_new = apply_conjunctions_binary_product(X, conjunctions)
            elif self.interaction_type == 'poly':
                X_new = apply_targeted_poly_features(X, interacting_sets)
            elif self.interaction_type == 'signed':
                X_new = apply_signed_interaction_features(X, interacting_sets)
            else:
                raise ValueError("Unknown interaction_type")
            X_combined = np.hstack([X, X_new])
        else:
            X_combined = X
        return self.model_.predict(X_combined)

# Example usage
def main():
    X, y = generate_data(8000, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    for method in ['random_forest', 'decision_tree']:
        for itype in ['conjunctions', 'poly', 'signed']:
            print(f"\nMethod: {method}, Interaction: {itype}")
            reg = InteractionFeatureRegressor(base_model=method, interaction_type=itype, verbose=True)
            reg.fit(X_train, y_train)
            r2 = r2_score(y_test, reg.predict(X_test))
            print(f"R² score: {r2:.4f}")

    print("\n=== Baseline Models ===")
    for baseline_model in [RidgeCV(), LassoCV(cv=5), RandomForestRegressor()]:
        name = baseline_model.__class__.__name__
        baseline_model.fit(X_train, y_train)
        r2_base = baseline_model.score(X_test, y_test)
        print(f"{name} R²: {r2_base:.4f}")

    print("\n=== Perfect Feature Baseline (Ridge) ===")
    X_perfect = np.stack([
        (1 - X_train[:, 1]) * X_train[:, 2],
        X_train[:, 3] * X_train[:, 4],
        X_train[:, 0] * X_train[:, 2] * X_train[:, 3]
    ], axis=1)
    X_perfect_test = np.stack([
        (1 - X_test[:, 1]) * X_test[:, 2],
        X_test[:, 3] * X_test[:, 4],
        X_test[:, 0] * X_test[:, 2] * X_test[:, 3]
    ], axis=1)
    ridge_perf = Ridge()
    ridge_perf.fit(X_perfect, y_train)
    r2_perf = ridge_perf.score(X_perfect_test, y_test)
    print(f"Perfect features Ridge R²: {r2_perf:.4f}")

if __name__ == "__main__":
    main()