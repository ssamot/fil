import json
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import _tree
from itertools import combinations
from collections import defaultdict
from pysr import PySRRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from open_spiel.python import policy
import pyspiel
from policies import policy_manipulation_and_conversion as conversion
from open_spiel.python.algorithms import exploitability

POLICY_TYPE = "cat"

if POLICY_TYPE == "cat":
    j_file = "data/cfr_categorical_policy_leduc_poker.json"
elif POLICY_TYPE == "sane":
    j_file = "data/cfr_sane_policy_leduc_poker.json"
else:
    raise ValueError("Policy type must be cat or sane, it is ", POLICY_TYPE)

key_map = {}

def state_to_key(state, game_name):
    infostate_string = state.information_state_string(state.current_player())
    if  infostate_string in key_map:
        return key_map[infostate_string]
    if POLICY_TYPE == "cat":
        ret = conversion.convert_one_hot_to_cat(state.information_state_tensor(state.current_player()), game_name)
    elif POLICY_TYPE == "sane":
        ret = conversion.convert_categorical_to_sane(conversion.convert_one_hot_to_cat(state.information_state_tensor(state.current_player()), game_name), game_name)
    else:
        raise ValueError("Policy type must be cat or sane, it is ", POLICY_TYPE)
    key_map[infostate_string] = ret
    return ret

with open(j_file) as f:
    d = json.load(f)


action_X = []
action_y = []

for key in d:
    X = key
    #print(X)
    actions = []
    for v in d[key]:
        action = v[0]
        prob = v[1]
        #print(X)
        #print(type(X))
        #exit()
        actions.append(float(prob))
    X_inter = X.strip("()").split(",")  # ['1', ' 3', 'f ']
    X_inter = [l.rstrip() for l in X_inter]
    #exit()
    int_list = [float(item.strip()) for item in X_inter]  # [1, 3]
    #int_list +=[action]
    action_X.append(int_list)
    action_y.append(actions)

        #print(int_list, float(prob))



X = np.array(action_X)
y = np.array(action_y)



from fil.fil_main import FeatureInteractionTransformer, FeatureInteractionLayer


cat_real_dims=[("cat",2),
               ("cat",6),
               ("cat",7),
               ("cat",3),
               ("cat",3),
               ("cat",3),
               ("cat",3),
               ("cat",3),
               ("cat",3),
               ("cat",3),
               ("cat",3),
               ("cat",3)]


from lightgbm import LGBMRegressor


clf = LGBMRegressor()

from tqdm import tqdm



import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMClassifier

#X = OneHotEncoder().fit_transform((X))
# Initialize regressor and CV
#clf = MultiOutputRegressor(LGBMRegressor(verbose = -1, n_estimators=1000, n_jobs=40))
from sklearn.calibration import CalibratedClassifierCV


#clf = LGBMClassifier(n_jobs=40, n_estimators=1000)

clf = RandomForestRegressor(n_jobs = 40, n_estimators=100)
#cv = KFold(n_splits=20, shuffle=True, random_state=42)
cv = LeaveOneOut()
#cv = LeavePOut(p = 2)

print("Cross-Validation Error Analysis (Regression)")
print("=" * 45)
# Initialize global worst tracking
global_worst_error = -np.inf
global_worst_sample_idx = None
global_worst_true = None
global_worst_pred = None
global_nash_conv = None

with tqdm(total=cv.get_n_splits(X, y), desc="CV") as pbar:
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_full = clf.predict(X)

        game = pyspiel.load_game("leduc_poker")
        tabular_policy = policy.TabularPolicy(game)
        for state in tabular_policy.states:
            infoset_key = state_to_key(state, "leduc_poker")
            probs = y_full[np.where(np.all(np.isclose(X, infoset_key), axis=1))][0]
            mask = np.zeros_like(probs, dtype=bool)
            mask[state.legal_actions()] = True
            probs = np.clip(probs, 0.0, 1.0)
            probs[~mask] = 0.0
            tabular_policy.action_probability_array[tabular_policy.state_lookup[tabular_policy._state_key(state, state.current_player())]] = probs

        nash_conv = exploitability.nash_conv(game, tabular_policy)

        # Compute per-sample errors
        if y.ndim == 1:
            errors = np.abs(y_val - y_pred)
        else:
            errors = np.mean(np.abs(y_val - y_pred), axis=1)

        # Find worst in current fold
        local_worst_idx = np.argmax(errors)
        local_worst_error = errors[local_worst_idx]
        local_sample_idx = val_idx[local_worst_idx]
        local_true = y_val[local_worst_idx]
        local_pred = y_pred[local_worst_idx]
        local_features = X[local_sample_idx]

        # Update global worst if needed
        if local_worst_error > global_worst_error:
            global_worst_error = local_worst_error
            global_worst_sample_idx = local_sample_idx
            global_worst_true = local_true
            global_worst_pred = local_pred
            global_worst_features = local_features
            global_nash_conv = nash_conv

        # Update progress bar with global worst so far
        pbar.set_postfix({
            "GlobalWorstErr": f"{global_worst_error:.4f}",
            "Sample": X[global_worst_sample_idx],
            "True": f"{np.round(global_worst_true, 2)}",
            "Pred": f"{np.round(global_worst_pred, 2)}",
            "Expl": f"{np.round(global_nash_conv, 3)}"
        })
        pbar.update(1)

# Final report
print("\n" + "=" * 45)
print("Global Worst Prediction Across All Folds")
print("=" * 45)
print(f"Sample Index: {global_worst_sample_idx}")
print(f"Features    : {global_worst_features}")
print(f"True Value  : {np.round(global_worst_true, 3)}")
print(f"Predicted   : {np.round(global_worst_pred, 3)}")
print(f"Error       : {np.round(global_worst_error, 4)}")

