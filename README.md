# FIL: Feature Interaction Layers for Deep CFR

This repository implements [Deep CFR](https://arxiv.org/abs/1811.00164) and several regressors that replace the original neural networks. The main experimental comparison is between **DeepCFR** (PyTorch MLP baseline) and **LUGL-DeepCFR** variants that use simpler, structured function approximators.

Precomputed exploitability curves for all paper experiments are stored in [`results/`](results/). Plotting scripts reproduce the figures with mean ± 95% confidence intervals across random seeds.

## Methods

| Label | Entry point | Regressor | Description |
|-------|-------------|-----------|-------------|
| **DeepCFR** | `mains/main_deep_cfr_pytorch.py` | PyTorch MLP | Standard Deep CFR with fully connected advantage and strategy networks on the full information-state tensor. |
| **LUGL-DeepCFR-LightGBM** | `mains/main_deep_cfr_lgb.py` | LightGBM | Same Deep CFR loop, but advantage and strategy targets are fit with LightGBM regressors on the full info state. |
| **LUGL-DeepCFR-Multi-S** | `mains/main_deep_cfr_public_table_splines.py` | Per-public-state splines | Splits the info state into a **public table key** (betting history) and a **hand-strength** feature; fits a separate spline regressor per public state. |
| **LUGL-DeepCFR-Multi-D** | `mains/main_deep_cfr_public_table_decision_tree.py` | Per-public-state decision trees | Same public-table decomposition as Multi-S, but uses decision-tree regressors. |

All methods share the same outer Deep CFR loop (external sampling traversals, reservoir buffers, regret matching, periodic NashConv logging). They differ only in how advantage and strategy values are stored and regressed.

The `fil/` package additionally implements **Feature Interaction Layers (FIL)** for neural Deep CFR (`mains/main_deep_cfr_fil.py`), which is separate from the LUGL experiments above.

## Experiments

Each run logs **exploitability** (OpenSpiel `nash_conv`) every iteration. Lower is better. A horizontal **Random** baseline is drawn on each plot for reference.

| Game | Config | Methods compared | Seeds |
|------|--------|------------------|-------|
| Kuhn poker | `configs/kuhn.py` | DeepCFR, LUGL-LightGBM | 10 |
| Leduc poker | `configs/leduc.py` | DeepCFR, LUGL-LightGBM, Multi-S, Multi-D | 10–11 |
| Leduc (ignore Jack+King) | `configs/leduc_ignore_jack_king.py` | DeepCFR, Multi-S, Multi-D | 10 |
| Leduc (ignore Jack+Queen) | `configs/leduc_ignore_jack_queen.py` | DeepCFR, Multi-S, Multi-D | 10 |
| Leduc (ignore Queen+King) | `configs/leduc_ignore_queen_king.py` | DeepCFR, Multi-S, Multi-D | 10 |
| Leduc (ignore Queens) | `configs/leduc_ignore_queens.py` | DeepCFR, Multi-S, Multi-D | 10–11 |
| Goofspiel (4 cards) | `configs/goofspiel_4.py` | DeepCFR, LUGL-LightGBM | 10 |
| Liar's Dice (4 sides) | `configs/liars_dice.py` | DeepCFR, LUGL-LightGBM | 10 |

### Results summary (final-iteration exploitability, mean ± std)

Values below are computed from the JSON files in `results/` at iteration 999 (or the last logged iteration).

| Game | DeepCFR | LUGL-LightGBM | Multi-S | Multi-D |
|------|---------|---------------|---------|---------|
| Kuhn | 0.104 ± 0.037 | **0.016 ± 0.006** | — | — |
| Leduc | 0.286 ± 0.053 | 0.122 ± 0.005 | **0.066 ± 0.008** | **0.066 ± 0.008** |
| Goofspiel-4 | 0.094 ± 0.022 | **0.011 ± 0.004** | — | — |
| Liar's Dice | 0.070 ± 0.010 | **0.016 ± 0.002** | — | — |

On the Leduc **ignore-cards** ablations, the public-table LUGL variants (Multi-S / Multi-D) consistently reach lower exploitability than the DeepCFR MLP baseline (~0.06 vs ~0.30–0.40), matching the full Leduc trend. The ignore-Queens variant is harder for all methods; see the plots for the full learning curves.

**Takeaway:** LUGL-DeepCFR reaches substantially lower exploitability than standard DeepCFR on every game tested. On Leduc, the public-table decompositions (Multi-S / Multi-D) outperform both the MLP baseline and the global LightGBM variant.

## Setup

### Requirements

- Python 3.10+ (3.11 on Windows per `Makefile`)
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) (`pyspiel`, `open_spiel`) — not pinned in `requirements.txt`; install separately before running experiments
- CUDA optional (PyTorch runs on CPU by default)

```bash
# Create and activate a virtual environment
make create_environment
# Windows: .\.venv\Scripts\activate
# Unix:    source ./.venv/bin/activate

make requirements
# Then install OpenSpiel, e.g.:
# pip install open_spiel
```

### Environment variable

All training scripts expect the repo root on `PYTHONPATH`:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."

# Unix
export PYTHONPATH=.
```

## Reproducing experiments

Configs in [`configs/`](configs/) define game, iteration count, traversals, network sizes, batch sizes, and the results output prefix. Pass `--index N` to write seed-specific files (`*_N.json`); omit it for a single unnumbered file.

### DeepCFR (PyTorch baseline)

```bash
PYTHONPATH=. python mains/main_deep_cfr_pytorch.py --config=configs/kuhn.py --index=0
PYTHONPATH=. python mains/main_deep_cfr_pytorch.py --config=configs/leduc.py --index=0
```

### LUGL-DeepCFR-LightGBM

```bash
PYTHONPATH=. python mains/main_deep_cfr_lgb.py --config=configs/kuhn.py --index=0
PYTHONPATH=. python mains/main_deep_cfr_lgb.py --config=configs/leduc.py --index=0
```

### LUGL-DeepCFR-Multi-S / Multi-D

```bash
PYTHONPATH=. python mains/main_deep_cfr_public_table_splines.py --config=configs/leduc.py --index=0
PYTHONPATH=. python mains/main_deep_cfr_public_table_decision_tree.py --config=configs/leduc.py --index=0
```

### Makefile shortcuts

```bash
make kuhn_torch      # DeepCFR on Kuhn
make leduc_torch     # DeepCFR on Leduc
make kuhn_fil_torch  # FIL variant on Kuhn
```

### Output format

Results are JSON objects mapping iteration number (string key) to exploitability (float), written under `results/`:

```
results/kuhn_pytorch_0.json
results/kuhn_lgb_0.json
results/leduc_pt_splines_0.json
results/leduc_pt_dt_0.json
```

The prefix comes from `results_file_base` in each config (e.g. `../results/kuhn_`).

### Running all seeds

Example loop for 10 seeds on Kuhn:

```bash
for i in $(seq 0 9); do
  PYTHONPATH=. python mains/main_deep_cfr_pytorch.py --config=configs/kuhn.py --index=$i
  PYTHONPATH=. python mains/main_deep_cfr_lgb.py --config=configs/kuhn.py --index=$i
done
```

On Windows PowerShell:

```powershell
0..9 | ForEach-Object {
  $env:PYTHONPATH = "."
  python mains/main_deep_cfr_pytorch.py --config=configs/kuhn.py --index=$_
  python mains/main_deep_cfr_lgb.py --config=configs/kuhn.py --index=$_
}
```

Leduc and the larger games use 1000 iterations with 1500 traversals per iteration and can take hours per seed.

## Plotting results

Precomputed results are already in `results/`. To regenerate figures:

```bash
# Single game
PYTHONPATH=. python plotting/plot_paper_expriment_results_kuhn.py
PYTHONPATH=. python plotting/plot_paper_expriment_results_leduc.py

# Log-scale y-axis
PYTHONPATH=. python plotting/plot_paper_expriment_results_kuhn.py --log

# All games that have complete result files
PYTHONPATH=. python plotting/plot_paper_all.py
```

Figures are saved as PDFs in `results/`:

- `results/fig_<game>.pdf` — linear scale
- `results/fig_<game>_log.pdf` — log scale

Plot styling (colors, labels, confidence level) is centralized in [`plotting/plot_paper_experiment_config.py`](plotting/plot_paper_experiment_config.py).

For ad-hoc comparison of individual JSON files:

```bash
python plotting/plot_results.py results/kuhn_pytorch_0.json results/kuhn_lgb_0.json
```

## Repository layout

```
configs/          Experiment hyperparameters (one .py file per game/variant)
deepcfr/          Deep CFR solver implementations
fil/              Feature Interaction Layer modules
mains/            Training entry points (one per method)
plotting/         Paper figure scripts
results/          Exploitability JSON logs and generated PDFs
our_cfr/          Tabular MCCFR baselines
policies/         Policy utilities
analysis/         Post-hoc analysis scripts
```

## References

- Brown, N., & Sandholm, T. (2019). [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1811.00164). (Deep CFR)
- Lanctot, M., et al. (2019). [OpenSpiel: A Framework for Reinforcement Learning in Games](https://arxiv.org/abs/2008.09404).
