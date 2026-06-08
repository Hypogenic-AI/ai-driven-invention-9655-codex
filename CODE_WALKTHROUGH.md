# Code Walkthrough

## Code Structure Overview

`src/run_discovery_experiments.py` contains the complete workflow:

1. Environment and workspace validation.
2. Materials Project feature extraction and caching.
3. Data audit for Materials Project and GNoME.
4. PyTorch MLP surrogate training.
5. Active-discovery loop across acquisition policies.
6. Statistical tests and GNoME transfer ranking.
7. Figure and metadata generation.

The script is intentionally self-contained so it can be rerun without notebook state.

## Environment Setup

```bash
source .venv/bin/activate
python src/run_discovery_experiments.py
```

Dependencies are tracked in `pyproject.toml` and installed with `uv add`.

## Key Functions

### `load_or_build_mp_features(force=False)`

Loads cached Materials Project descriptors from `results/mp_features.npz`, or builds them from `datasets/materials_project_hf/extracted/data.hdf5`.

It excludes rows where `energy_above_hull` is missing and records the exclusion count in `results/data_audit.json`.

### `composition_feature_from_counts(counts_by_z)`

Converts atomic counts into 118 element-fraction features plus summary descriptors such as composition entropy, atom count, and weighted atomic-number statistics.

### `train_mlp(X, y, seed, config, epochs=None)`

Trains the small PyTorch surrogate classifier. It uses CUDA when available, mixed precision on GPU, class-weighted binary cross entropy, and a fixed training batch size of 128.

### `run_active_discovery(X_struct, X_comp, y, config)`

Runs the budgeted discovery benchmark. Each seed uses a 25,000-row candidate pool, 400 initial observations, and 8 acquisition rounds of 250 candidates.

### `summarize_active_results(trace)`

Computes final policy means and paired policy-vs-random statistical tests, including Holm-adjusted p-values.

### `run_gnome_transfer(model, scaler, config)`

Applies the Materials Project composition-only model to GNoME R2SCAN summaries and evaluates top-K on-hull enrichment.

## Data Pipeline

Raw data:

`Materials Project HDF5 -> descriptor extraction -> label filtering -> MLP training -> active acquisition -> statistical analysis`

External ranking:

`GNoME composition strings -> composition descriptors -> MP-trained model score -> top-K enrichment analysis`

## Outputs

| Output | Meaning |
|---|---|
| `results/data_audit.json` | Dataset counts, missingness, label prevalence |
| `results/predictive_metrics.csv` | Classifier metrics on held-out Materials Project rows |
| `results/active_learning_trace.csv` | Per-round results for each policy and seed |
| `results/active_learning_summary.csv` | Final policy-level means and standard deviations |
| `results/stat_tests.csv` | Paired statistical tests against random |
| `results/gnome_transfer_summary.csv` | Top-K GNoME enrichment metrics |
| `results/environment.json` | Hardware, versions, config, runtime |
| `figures/*.png` | Plots used by `REPORT.md` |

## Reproducibility

The script sets Python, NumPy, and PyTorch seeds. It also runs a deterministic mini-rerun of the greedy policy and writes the result to `results/reproducibility_check.json`.

The corrected full run completed in 164.5 seconds on one RTX A6000 GPU.
