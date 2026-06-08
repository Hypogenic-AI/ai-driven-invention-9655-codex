# Resources Catalog

## Summary

This document catalogs resources gathered for "Beyond Trial and Error: AI-Driven Invention (Atomic Modelling)."

- Papers downloaded: 14
- Datasets downloaded or partially mirrored: 5
- Code repositories cloned: 8
- Environment: fresh local `uv` venv at `.venv/`
- Dependencies recorded in `pyproject.toml`: `pypdf`, `requests`, `arxiv`, `datasets`, `pandas`, `h5py`

## Papers

| Title | Authors | Year | File | Key info |
|---|---|---:|---|---|
| Towards Automated Discovery | Babu, Gouvea, Rignanese | 2026 | `papers/2606.02507_automated_inverse_materials_design_review.pdf` | Latest review of generative, multimodal, closed-loop inverse design |
| OMat24 | Barroso-Luque, Shuaibi, Fu, Wood, et al. | 2024 | `papers/2410.12771_omat24_open_materials_dataset_models.pdf` | Large open inorganic DFT dataset and models |
| UMA | Wood, Dzamba, Fu, Gao, et al. | 2025 | `papers/2506.23971_uma_universal_models_for_atoms.pdf` | Cross-domain universal atomistic model |
| MACE-MP | Batatia, Benner, Chiang, Elena, et al. | 2024 | `papers/2401.00096_mace_mp_atomistic_materials_chemistry_foundation_model.pdf` | Foundation MLIP for broad atomistic simulations |
| CHGNet | Deng, Zhong, Jun, Riebesell, et al. | 2023 | `papers/2302.14231_chgnet_charge_informed_atomistic_modeling.pdf` | Charge-informed universal potential |
| M3GNet | Chen, Ong | 2022 | `papers/2202.02450_m3gnet_universal_graph_interatomic_potential.pdf` | Universal graph IAP and high-throughput screening |
| Matbench Discovery | Riebesell, Goodall, Benner, Chiang, et al. | 2023 | `papers/2308.14920_matbench_discovery_crystal_stability_benchmark.pdf` | Stability-prediction benchmark |
| MatterGen | Zeni, Pinsler, Zuegner, Fowler, et al. | 2023 | `papers/2312.03687_mattergen_generative_inorganic_materials_design.pdf` | Diffusion generation of inorganic materials |
| LLMatDesign | Jia, Zhang, Fung | 2024 | `papers/2406.13163_llmatdesign_autonomous_materials_discovery_llms.pdf` | LLM-guided material modification loop |
| Diffusion Transformers | Takahara, Shibata, Mizoguchi | 2024 | `papers/2406.09263_diffusion_transformers_crystal_inverse_design.pdf` | Transformer diffusion for crystal inverse design |
| Atomistic Simulation Perspective | Yuan, Liu, Chen, Zhong, et al. | 2025 | `papers/2503.10538_foundation_models_atomistic_simulation_perspective.pdf` | Foundation-model perspective and caveats |
| GNoME | Merchant, Batzner, Schoenholz, Aykol, Cheon, Cubuk | 2023 | `papers/s41586-023-06735-9_gnome_scaling_deep_learning_materials_discovery.pdf` | Active-learning graph networks for materials discovery |
| A-Lab | Szymanski, Rendy, Fei, Kumar, et al. | 2023 | `papers/s41586-023-06734-w_alab_autonomous_laboratory_inorganic_materials.pdf` | Robotic closed-loop inorganic synthesis |
| A-Lab Correction | Nature correction | 2026 | `papers/s41586-025-09992-y_alab_nature_correction_2026.pdf` | Correction on A-Lab novelty/interpretation claims |

See `papers/README.md` for details.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|---|---|---:|---|---|---|
| GNoME summaries | Google Cloud public bucket | 554,054 + 139,541 rows | Candidate stability screening | `datasets/gnome_summaries/` | Structure ZIPs not downloaded due size |
| Materials Project HF | Hugging Face `materials-toolkits/materials-project` | 133,420 structures | Formation/stability property modeling | `datasets/materials_project_hf/` | Extracted HDF5 validated |
| JARVIS-DFT 3D | Hugging Face `colabfit/JARVIS_DFT_3D_8_18_2021` | 56,627 rows | Crystal properties, energy, force/stress | `datasets/jarvis_dft_3d_hf/` | Parquet validated |
| OMat24 references | Hugging Face `facebook/OMAT24` | 89 reference entries | Element references/corrections | `datasets/omat24_references/` | Full OMat24 not mirrored |
| QM9 | Hugging Face `yairschiff/qm9` | 133,885 molecules | Molecular atomistic property benchmark | `datasets/qm9_hf/` | Saved with `datasets.save_to_disk` |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| GNoME materials discovery | https://github.com/google-deepmind/materials_discovery | Dataset access, GNoME model definitions | `code/materials_discovery_gnome/` | Includes full data downloader |
| Matbench Discovery | https://github.com/janosh/matbench-discovery | Stability benchmark and leaderboard code | `code/matbench-discovery/` | Recommended evaluation harness |
| MatterGen | https://github.com/microsoft/mattergen | Generative inorganic material design | `code/mattergen/` | LFS skipped; pull checkpoints as needed |
| CHGNet | https://github.com/CederGroupHub/chgnet | Charge-informed MLIP | `code/chgnet/` | Fast relaxation/MD baseline |
| fairchem | https://github.com/facebookresearch/fairchem | UMA and FAIR atomistic models | `code/fairchem/` | UMA requires HF access |
| MACE foundations | https://github.com/ACEsuit/mace-foundations | MACE model registry and scripts | `code/mace-foundations/` | Install MACE separately |
| MatGL | https://github.com/materialsvirtuallab/matgl | M3GNet/CHGNet/QET graph library | `code/matgl/` | Current PyG-based graph potential toolkit |
| LLMatDesign | https://github.com/Fung-Lab/LLMatDesign | LLM material-design loop | `code/LLMatDesign/` | Depends on MatDeepLearn |

See `code/README.md` for details.

## Search Strategy

The prescribed paper-finder service was tried first but did not return results before timeout. Manual fallback searched:

- arXiv for recent model, benchmark, and review papers.
- Nature/Nature Computational Science for GNoME, A-Lab, MatterGen, OMat24, and corrections.
- Hugging Face for datasets and model cards.
- GitHub for official implementations and benchmark code.

Selection prioritized papers with runnable code, released datasets, benchmark relevance, or direct bearing on autonomous material discovery.

## Challenges Encountered

- Paper-finder timed out in both diligent and fast mode.
- Full OMat24 and full GNoME structure archives are too large for blind mirroring in this phase.
- UMA model use requires Hugging Face access approval.
- MatterGen checkpoints and data are LFS/Hugging Face assets and were intentionally not pulled during shallow clone.
- A-Lab claims require careful interpretation because Nature issued a 2026 correction related to novelty and material identification.

## Recommendations for Experiment Design

1. Primary dataset: start with `datasets/materials_project_hf/` plus `datasets/jarvis_dft_3d_hf/` for manageable property prediction, then use `datasets/gnome_summaries/` for candidate screening.
2. Primary benchmark: use `code/matbench-discovery/` and report stability F1, precision, recall, energy-above-hull MAE, and discovery acceleration factor.
3. Candidate generation: use MatterGen if GPU/checkpoints are available; otherwise use substitution/prototype enumeration from known structures as a transparent baseline.
4. Surrogate relaxation: start with CHGNet or MACE-MP; add UMA only after HF model access is configured.
5. Validation: do not claim invention from MLIP scores alone. Deduplicate, check database novelty, verify stability with convex hulls, and reserve higher-fidelity DFT or experimental characterization for top candidates.

## Research Execution Artifacts

The automated execution phase used the gathered resources to run an offline active-discovery benchmark:

- Plan: `planning.md`
- Main code: `src/run_discovery_experiments.py`
- Report: `REPORT.md`
- Overview: `README.md`
- Code notes: `CODE_WALKTHROUGH.md`
- Results: `results/`
- Figures: `figures/`

The final corrected run excluded 3,798 Materials Project rows with missing `energy_above_hull`, used 129,622 labeled rows, and evaluated budgeted active discovery over five random seeds. The best structured policy, diverse UCB, found 1,933.2 stable or near-hull candidates on average at a 2,400-candidate validation budget, compared with 1,364.8 for random selection.
