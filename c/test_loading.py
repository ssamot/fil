import numpy as np

# Number of 7-card hands from a 52-card deck
num_combos = 133784560  # C(52, 7)

# Load from binary file
hand_ranks = np.fromfile("hand_ranks.bin", dtype=np.int32)

print(f"Loaded {hand_ranks.size} evaluations.")
print(f"First few values: {hand_ranks[:10]}")