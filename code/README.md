# Cloned Repositories

Repository count: 8

All repositories are shallow clones under `code/`. MatterGen was cloned with `GIT_LFS_SKIP_SMUDGE=1` to avoid downloading large checkpoints and datasets automatically.

## google-deepmind/materials_discovery

- URL: https://github.com/google-deepmind/materials_discovery
- Location: `code/materials_discovery_gnome/`
- Purpose: GNoME dataset access, model definitions, and notebooks for exploring stable materials and decomposition energies.
- Key files:
  - `README.md`
  - `DATASET.md`
  - `scripts/download_data_wget.py`
  - `model/gnome.py`
  - `notebooks/Compute_Decomposition_Energies.ipynb`
- Notes: Summary CSVs were downloaded separately to `datasets/gnome_summaries/`. Full structure ZIPs are available through this repo's downloader but were not pulled because of size.

## janosh/matbench-discovery

- URL: https://github.com/janosh/matbench-discovery
- Location: `code/matbench-discovery/`
- Purpose: Benchmark framework and leaderboard code for crystal stability prediction and discovery-oriented metrics.
- Key files:
  - `readme.md`
  - `matbench_discovery/data.py`
  - `matbench_discovery/energy.py`
  - `matbench_discovery/modeling-tasks.yml`
  - `data/datasets.yml`
- Notes: Strong candidate for the experiment runner's primary evaluation harness.

## microsoft/mattergen

- URL: https://github.com/microsoft/mattergen
- Location: `code/mattergen/`
- Purpose: Official MatterGen implementation for unconditional and property-conditioned inorganic crystal generation.
- Key files:
  - `README.md`
  - `mattergen/generator.py`
  - `sampling_conf/default.yaml`
  - `sampling_conf/csp.yaml`
  - `data-release/README.md`
- Notes: Requires Python 3.10+ and likely CUDA for practical generation. Checkpoints can be pulled from Hugging Face or Git LFS when needed.

## CederGroupHub/chgnet

- URL: https://github.com/CederGroupHub/chgnet
- Location: `code/chgnet/`
- Purpose: CHGNet pretrained universal charge-informed neural network potential.
- Key files:
  - `README.md`
  - `chgnet/model/`
  - `examples/`
- Notes: Useful for fast relaxation, molecular dynamics, magnetic moment inference, and baseline MLIP evaluation.

## facebookresearch/fairchem

- URL: https://github.com/facebookresearch/fairchem
- Location: `code/fairchem/`
- Purpose: FAIR Chemistry codebase for UMA and related pretrained atomistic models.
- Key files:
  - `README.md`
  - `packages/fairchem-core/`
- Notes: UMA model access requires a Hugging Face account/token and accepted model access. Provides ASE calculator workflows for `omat`, `omol`, `oc20`, `odac`, and other tasks.

## ACEsuit/mace-foundations

- URL: https://github.com/ACEsuit/mace-foundations
- Location: `code/mace-foundations/`
- Purpose: Registry and training scripts for MACE foundation models including MACE-MP, MACE-OMAT, MACE-MATPES, and MACE-MH.
- Key files:
  - `README.md`
  - `mace_mp_0a/`
- Notes: Install the main MACE package separately. This repo is mainly a model catalog and reference point.

## materialsvirtuallab/matgl

- URL: https://github.com/materialsvirtuallab/matgl
- Location: `code/matgl/`
- Purpose: Materials Graph Library implementing M3GNet, CHGNet-like models, TensorNet, QET, and Hugging Face model loading.
- Key files:
  - `README.md`
  - `src/matgl/`
  - `pyproject.toml`
- Notes: Current MatGL v3 targets PyTorch Geometric and loads canonical pretrained models from the `materialyze` Hugging Face organization.

## Fung-Lab/LLMatDesign

- URL: https://github.com/Fung-Lab/LLMatDesign
- Location: `code/LLMatDesign/`
- Purpose: LLM-driven autonomous material modification loop for target property design.
- Key files:
  - `README.md`
  - `llmatdesign/utils.py`
  - `notebooks/band_gap_example.ipynb`
  - `requirements.txt`
- Notes: Requires MatDeepLearn setup. Useful as an agentic design-loop reference rather than a plug-and-play universal experiment harness.
