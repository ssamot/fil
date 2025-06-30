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
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import NuSVR
import warnings
warnings.filterwarnings("ignore")
clf = RandomForestRegressor(n_jobs = 40, n_estimators=100)
clf = MultiOutputRegressor(LGBMRegressor(verbose = -1, n_estimators=100, n_jobs=40))
clf = MultiOutputRegressor(NuSVR())
from sklearn.preprocessing import OneHotEncoder


j_file = "data/cfr_categorical_policy_leduc_poker.json"

with open(j_file) as f:
    d = json.load(f)


action_X = []
action_y = []

for key in d:
    X = key
    actions = []
    for v in d[key]:
        action = v[0]
        prob = v[1]
        #print(X)
        #print(type(X))
        #exit()
        actions.append(float(prob))
    X_inter = X.strip("()").split(",")  # ['1', ' 3', 'f ']
    int_list = [int(item.strip()) for item in X_inter if item.strip().isdigit()]  # [1, 3]
    #int_list +=[action]
    action_X.append(int_list)
    action_y.append(actions)

        #print(int_list, float(prob))



X = np.array(action_X)
y = np.array(action_y)

import os
import numpy as np
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

# Hyperparameters
x_fraction = 0.3  # Fraction (0 < x_fraction ≤ 1) of worst val samples to move to training
n_splits = 1000
test_size = 0.8
save_path = "./data/best_train_indices_fold.csv"

cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
best_local_worst_error = float("inf")
save_path = f"./data/best_train_indices_fold.csv"

X = OneHotEncoder(sparse_output=False).fit_transform(X)

with tqdm(total=cv.get_n_splits(X, y), desc="Cross-Validation") as pbar:
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Initial train/val split
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # First round fit/predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        # Error on output[0]
        if y.ndim == 1:
            errors = np.abs(y_val - y_pred)
        else:
            errors = np.abs(y_val[:, 0] - y_pred[:, 0])

        # Select worst x% in validation
        num_to_select = max(1, int(np.ceil(len(errors) * x_fraction)))
        worst_indices_in_val = np.argsort(errors)[-num_to_select:]
        val_aug_indices = val_idx[worst_indices_in_val]  # global indices

        # Augment training set
        X_aug = X[val_aug_indices]
        y_aug = y[val_aug_indices]
        X_train_aug = np.concatenate([X_train, X_aug], axis=0)
        y_train_aug = np.concatenate([y_train, y_aug], axis=0)
        train_aug_indices = np.concatenate([train_idx, val_aug_indices], axis=0)

        # Remove augmented samples from validation
        real_val_mask = ~np.isin(val_idx, val_aug_indices)
        val_idx_real = val_idx[real_val_mask]
        X_val_real = X[val_idx_real]
        y_val_real = y[val_idx_real]

        if len(val_idx_real) == 0:
            pbar.set_postfix({"Note": "Skipped (no val left)"})
            pbar.update(1)
            continue

        # Retrain and re-evaluate
        clf.fit(X_train_aug, y_train_aug)
        y_val_pred = clf.predict(X_val_real)

        if y.ndim == 1:
            final_errors = np.abs(y_val_real - y_val_pred)
        else:
            final_errors = np.abs(y_val_real[:, 0] - y_val_pred[:, 0])

        local_worst_idx = np.argmax(final_errors)
        local_worst_error = final_errors[local_worst_idx]
        local_sample_idx = val_idx_real[local_worst_idx]
        local_true = y_val_real[local_worst_idx]
        local_pred = y_val_pred[local_worst_idx]

        if local_worst_error < best_local_worst_error:
            best_local_worst_error = local_worst_error
            best_sample_idx = local_sample_idx
            best_true = local_true
            best_pred = local_pred
            np.savetxt(save_path, train_aug_indices.reshape(-1, 1), fmt="%d", delimiter=",")

        real_test_pct = 100 * len(val_idx_real) / len(X)

        # Update progress bar
        pbar.set_postfix({
            "BestLocalWorst": f"{best_local_worst_error:.4f}",
            "Sample": int(best_sample_idx),
            "True": np.round(best_true, 2),
            "Pred": np.round(best_pred, 2),
            "RealTest%": f"{real_test_pct:.1f}%"
        })
        pbar.update(1)
