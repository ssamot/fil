import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
from fil_main import generate_data
import numpy as np
from itertools import combinations, chain
from collections import defaultdict

# 1. Generate synthetic binary function: f(x) = XOR(x1, x2) ^ x3

# 2. Generate data
n_binary_inputs = 12
groups = [[(0, 1, 2), 3], [(3, 4, 5), 2], [(6, 7, 8), 2], [(9, 10, 11), 3]]
group_ops = ['and', 'and', 'or', 'xor']
test_size = 0.2
rank = 12

# --- Generate data ---
X, y = generate_data(1000, n_binary_inputs, groups, group_ops)

#X_test, y_test = generate_data(10000, n_binary_inputs, groups, group_ops)


# 3. Estimate conditional variances V_S = Var(E[f | x_S])
def estimate_V_S(X, y, subset):
    """Estimate variance contribution of variable subset S."""
    # Extract values for subset
    keys = [tuple(x[s] for s in subset) for x in X]
    key_to_vals = defaultdict(list)

    for key, val in zip(keys, y):
        key_to_vals[key].append(val)

    conditional_means = [np.mean(vals) for vals in key_to_vals.values()]
    V_S = np.var(conditional_means)
    return V_S


# 4. Analyze subsets (up to 3-way groups)
subset_V = {}
max_order = 3
variables_to_check = list(range(6))  # focus on first 6 variables

for r in range(1, max_order + 1):
    for subset in combinations(variables_to_check, r):
        V_S = estimate_V_S(X, y, subset)
        subset_V[subset] = V_S


# 5. Compute interaction strength
def get_non_additive_strength(subset):
    V_S = subset_V[subset]
    total_additive = 0
    for r in range(1, len(subset)):
        for sub in combinations(subset, r):
            total_additive += subset_V.get(sub, 0)
    return V_S - total_additive


interaction_strengths = []
for subset in subset_V:
    if len(subset) >= 2:
        strength = get_non_additive_strength(subset)
        if strength > 0.005:  # threshold for significance
            interaction_strengths.append((subset, strength))

# 6. Report results
interaction_strengths.sort(key=lambda x: -x[1])
print("Detected Non-Additive Interactions:")
for subset, strength in interaction_strengths:
    vars_str = ', '.join(f'x{i}' for i in subset)
    print(f"Group: [{vars_str}] → interaction strength = {strength:.4f}")