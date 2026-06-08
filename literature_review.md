# Literature Review: AI-Driven Invention Through Atomic Modeling

## Review Scope

### Research Question

Can an intelligent system autonomously combine generative models, atomistic surrogate models, benchmarked stability predictors, and closed-loop validation to discover novel materials more systematically than trial-and-error experimentation?

### Inclusion Criteria

- Methods that operate on atoms, molecules, crystals, or periodic structures.
- Work on materials discovery, inverse design, autonomous labs, ML interatomic potentials, or foundation atomistic models.
- Papers with datasets, code, benchmarks, or clear experimental workflows useful to a downstream experiment runner.
- Recent work from 2022-2026, plus core benchmark/foundation papers.

### Exclusion Criteria

- Purely conceptual AI-discovery commentary without usable methods, data, or benchmarks.
- Non-atomic design tasks where structure, energy, stability, or molecular/crystal properties are absent.
- Datasets too large to download blindly without a benchmark-sized subset or reproducible partial-access path.

### Search Log

| Date | Query | Source | Results | Notes |
|---|---|---:|---:|---|
| 2026-06-08 | "AI-driven materials discovery autonomous experimentation atomic modeling" | paper-finder | 0 usable | Diligent mode stalled; process was stopped. |
| 2026-06-08 | "AI-driven materials discovery atomic modeling" | paper-finder | 0 usable | Fast mode timed out after 30 seconds. |
| 2026-06-08 | AI-driven materials discovery, foundation MLIPs, autonomous labs, generative crystal inverse design | arXiv/Nature/HF/GitHub | 14 papers | Manual fallback search. |

## Research Area Overview

The area is converging on a modular discovery stack. Candidate proposals come from substitution/enumeration, generative crystal models, or LLM-guided modification agents. Candidate evaluation is increasingly handled by universal or foundation machine-learned interatomic potentials (MLIPs), then calibrated against DFT or experimental characterization. Benchmarks such as Matbench Discovery shift evaluation from generic formation-energy regression to the task that matters for discovery: whether an unrelaxed candidate can be triaged as stable relative to a DFT convex hull.

The hypothesis is directionally supported, but with a major qualification: current AI systems are strongest as accelerators and prioritizers, not autonomous arbiters of novelty or synthesizability. The A-Lab correction and broader critique around automated phase identification show that closed-loop systems need robust independent verification, especially when claiming "new" materials.

## Key Papers

### GNoME: Scaling Deep Learning for Materials Discovery

- Authors: Merchant, Batzner, Schoenholz, Aykol, Cheon, Cubuk
- Year: 2023
- Source: Nature
- Key contribution: Scaled graph networks plus active learning to search inorganic crystal space, reporting 2.2M structures below the convex hull and a released subset of highly stable candidates.
- Methodology: Candidate generation from structural/compositional transformations, GNN filtering, DFT validation, iterative retraining, and additional r2SCAN checks.
- Datasets used: Materials Project, OQMD, WBM, ICSD snapshots; released GNoME summaries and structures.
- Results: Improved discovery efficiency by about an order of magnitude; released 381k initially and later more than 520k near-hull materials in the repository.
- Code/data: `code/materials_discovery_gnome/`; summaries downloaded to `datasets/gnome_summaries/`.
- Relevance: Best example of large-scale AI-directed material search with active-learning feedback.

### A-Lab: Autonomous Laboratory for Inorganic Materials

- Authors: Szymanski, Rendy, Fei, Kumar, He, Milsted, McDermott, et al.
- Year: 2023, correction 2026
- Source: Nature
- Key contribution: Robotic solid-state synthesis loop that plans, executes, characterizes, and updates synthesis experiments.
- Methodology: Candidate selection from phase-stability databases, NLP-based synthesis recipe suggestions, robotic powder handling, automated XRD phase analysis, and active learning over synthesis recipes.
- Datasets used: Materials Project, Google DeepMind/GNoME-derived candidates, text-mined synthesis literature.
- Results: Original paper reported 36/57 realized targets over 17 days; the 2026 correction clarifies novelty and interpretation concerns.
- Code/data: Target-screening algorithm link in paper; no full robotic system code cloned.
- Relevance: Shows why autonomous discovery must couple prediction with audited experiment interpretation.

### Matbench Discovery

- Authors: Riebesell, Goodall, Benner, Chiang, Deng, Ceder, Asta, Lee, Jain, Persson
- Year: 2023/2024
- Source: arXiv; later Nature Machine Intelligence
- Key contribution: Discovery-oriented benchmark for ML energy models used as DFT prefilters.
- Methodology: Predict relaxed energies from unrelaxed WBM test structures, classify thermodynamic stability against a DFT convex hull, and rank models by task metrics.
- Datasets used: MP v2022.10.28 as allowed training set; WBM unrelaxed structures as test set.
- Baselines: random forests, CGCNN, MEGNet, ALIGNN, M3GNet, CHGNet, MACE, SevenNet, Orb, EquiformerV2/eSEN variants.
- Results: Universal interatomic potentials outperform older property predictors; OMat24-derived models now dominate leaderboard-style metrics.
- Code/data: `code/matbench-discovery/`.
- Relevance: Recommended primary benchmark for evaluating material discovery triage.

### OMat24

- Authors: Barroso-Luque, Shuaibi, Fu, Wood, Dzamba, Gao, Rizvi, Zitnick, et al.
- Year: 2024, updated 2026
- Source: arXiv; Nature Computational Science version appeared in 2026
- Key contribution: Open inorganic materials dataset with over 110M DFT calculations and pretrained models.
- Methodology: Generate diverse non-equilibrium configurations via rattling, relaxation sampling, AIMD, and split by in-domain and out-of-domain composition/element/prototype criteria.
- Datasets used: Materials Project, MPtrj, Alexandria, OMat generation pipelines.
- Results: OMat24-pretrained models achieve state-of-the-art Matbench Discovery results, with reported F1 above 0.9 and energy-above-hull MAE around 18-20 meV/atom for top models.
- Code/data: OMat24 references downloaded to `datasets/omat24_references/`; full dataset not mirrored due size.
- Relevance: Strongest open data source for training or fine-tuning current inorganic MLIPs.

### MatterGen

- Authors: Zeni, Pinsler, Zuegner, Fowler, Horton, Fu, Shysheya, Crabbé, et al.
- Year: 2023/2025
- Source: arXiv/Nature
- Key contribution: Diffusion model for directly generating stable, unique, novel inorganic crystals with optional property, chemistry, and symmetry conditioning.
- Methodology: Joint diffusion over atom types, fractional coordinates, and lattice, with equivariant score network and property-adapter fine-tuning.
- Datasets used: Alex-MP, MP-20, ICSD/reference sets, DFT relaxation evaluation.
- Baselines: CDVAE, G-SchNet, FTCP, substitution, random structure search.
- Metrics: S.U.N. rate (stable, unique, novel), RMSD to DFT-relaxed structures, novelty/uniqueness, target-property success.
- Code/data: `code/mattergen/`.
- Relevance: Best candidate generator for experiments that move beyond screening known enumerations.

### LLMatDesign

- Authors: Jia, Zhang, Fung
- Year: 2024
- Source: arXiv
- Key contribution: LLM-agent loop for interpretable material modification toward user-specified property targets.
- Methodology: LLM proposes addition/removal/substitution/exchange operations, tools relax structures and predict properties, and self-reflection is fed back into the next iteration.
- Datasets/tools used: Materials Project queries, ML force fields, ML property predictors.
- Results: GPT-4o and Gemini variants outperform random modification baselines on band-gap and formation-energy tasks, but results depend on tool quality and target setup.
- Code: `code/LLMatDesign/`.
- Relevance: Useful blueprint for agent orchestration and hypothesis-driven material modification.

### M3GNet, CHGNet, MACE-MP, and UMA

- M3GNet introduced a universal graph IAP with three-body interactions trained on Materials Project relaxation trajectories, then used it to screen 31M hypothetical structures.
- CHGNet adds magnetic-moment regularization to infer charge-relevant information and supports energy, forces, stress, magnetic moments, relaxation, and MD.
- MACE-MP demonstrates broad zero-shot MD and chemistry/materials use from MPtrj-trained MACE foundation potentials, while highlighting that fine-tuning remains important for reaction barriers and specialized domains.
- UMA scales cross-domain atomistic modeling to roughly 500M unique 3D atomic structures and uses task conditioning plus mixture-of-linear-experts routing to support materials, molecules, catalysts, MOFs, and molecular crystals.

Together these papers define the current surrogate-model baseline set for atomic experiments.

## Common Methodologies

- Graph neural network potentials: M3GNet, CHGNet, MACE, Equiformer/eSEN, UMA, Orb, SevenNet. Used for fast relaxation, MD, energy/force/stress prediction, and DFT prefiltering.
- Active learning: GNoME and A-Lab iteratively add high-value calculations or experiments to improve future decisions.
- Generative crystal modeling: MatterGen and diffusion-transformer work generate structures directly rather than only filtering enumerated candidates.
- LLM-guided agents: LLMatDesign uses language models to choose chemically motivated modifications and reflect on prior failures.
- Convex-hull stability filtering: central for GNoME, A-Lab target selection, and Matbench Discovery evaluation.

## Standard Baselines

- Random selection or random structure search: lower-bound discovery acceleration.
- Substitution/prototype enumeration: strong classical materials-discovery baseline.
- CGCNN, MEGNet, ALIGNN: older supervised crystal-property predictors.
- M3GNet, CHGNet, MACE-MP, SevenNet, Orb, Equiformer/eSEN, UMA: modern MLIP baselines.
- CDVAE, G-SchNet, FTCP, SyMat: generative crystal baselines.

## Evaluation Metrics

- Energy MAE/RMSE: useful but insufficient for discovery if used alone.
- Energy above hull MAE: more task-aligned for stability triage.
- Stability F1, precision, recall/TPR, TNR: core classification metrics in Matbench Discovery.
- Discovery acceleration factor: measures enrichment of stable candidates compared with random selection.
- S.U.N. rate: stable, unique, novel material fraction for generative models.
- RMSD to DFT-relaxed structures: measures whether generated structures are near equilibrium.
- Experimental synthesis/characterization success: necessary for final claims, but hard to automate reliably.

## Datasets in the Literature

- Materials Project/MPtrj: canonical training source for M3GNet, CHGNet, MACE-MP, and Matbench Discovery.
- WBM: out-of-distribution test set for discovery from elemental substitutions.
- GNoME: large catalog of predicted stable/near-hull materials.
- OMat24: current high-scale open DFT training data for inorganic MLIPs.
- JARVIS-DFT: independent computed-property repository useful for validation.
- QM9: molecule-scale atomistic property benchmark for chemistry/medicine-adjacent tests.

## Gaps and Opportunities

- Novelty claims need stricter automated checking against disorder, polymorphism, and database updates.
- Formation-energy regression does not guarantee useful discovery; use convex-hull and downstream task metrics.
- MLIPs can be systematically soft or unreliable out of distribution; fine-tuning or uncertainty calibration is needed.
- Full DFT validation remains the bottleneck; experiment runners should budget validation calls carefully.
- Generators still require post-generation relaxation, duplicate detection, charge/oxidation sanity checks, and stability verification.
- Autonomous labs need audited characterization and independent confirmation before claims of "new" materials.

## Recommendations for Experiments

- Primary benchmark: Matbench Discovery with WBM-style unrelaxed structures and stability F1/DAF.
- Primary local datasets: Materials Project HF snapshot, JARVIS-DFT, GNoME summaries; use OMat24 references and selectively pull OMat24 subsets only if needed.
- Candidate generator: MatterGen for property-conditioned crystal proposals; substitution/prototype enumeration as a classical baseline.
- Surrogate models: start with CHGNet or MACE-MP because they are easier to run locally than restricted UMA; include UMA if Hugging Face model access is available.
- Verification flow: generate or enumerate candidates, relax with at least one MLIP, estimate energy above hull, deduplicate/novelty-check, then reserve DFT or higher-fidelity validation for top candidates.
- Agentic loop: LLMatDesign is a useful pattern for hypothesis/action/reflection, but property predictors and structure validators should be treated as tools with explicit uncertainty and failure modes.
