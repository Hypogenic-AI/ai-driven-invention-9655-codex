# Planning: AI-Driven Invention Through Atomic Materials Discovery

## Motivation & Novelty Assessment

### Why This Research Matters
Systematized invention in atomic materials requires more than generating many combinations: it needs a way to prioritize which combinations deserve costly validation. If an autonomous loop can learn from prior atomic structures and select better candidates than random trial-and-error, it would support the practical core of AI-driven invention for materials, energy, and medicine-adjacent molecular design.

### Gap in Existing Work
The literature review shows strong progress in GNoME, Matbench Discovery, MatterGen, CHGNet, MACE-MP, OMat24, and A-Lab, but also highlights a gap between large-scale candidate production and audited autonomous discovery. Existing work often demonstrates powerful screening or generation, while novelty, synthesizability, and validation remain bottlenecks. A compact offline experiment can test the narrower mechanism: whether structured autonomous selection increases discovery yield under a fixed validation budget.

### Our Novel Contribution
This study tests a budgeted, active discovery loop on local real atomic-materials datasets. The contribution is not a claim of new physical materials, but an empirical measurement of how much a structured surrogate-guided loop improves stable-material discovery relative to random search and simpler acquisition rules when the hidden oracle is real energy-above-hull data.

### Experiment Justification
- Experiment 1: Data audit and feature validation are needed to ensure the offline oracle is real, labels are not leaked, and Materials Project/JARVIS/GNoME resources can support atomic-discovery evaluation.
- Experiment 2: Budgeted active-learning discovery on Materials Project is needed to test whether a structured autonomous loop finds stable or near-hull materials faster than trial-and-error.
- Experiment 3: External screening on GNoME summaries is needed to test whether the learned prioritization transfers to a distinct candidate catalog and can rank high-stability candidates.
- Experiment 4: Error and subgroup analysis are needed because autonomous invention systems can fail systematically on rare chemistries, larger unit cells, or out-of-distribution combinations.

## Research Question
Can an intelligent, structured experimental loop use atomic composition and structure descriptors to discover stable or near-hull inorganic materials more efficiently than random trial-and-error under a fixed validation budget?

## Background and Motivation
The user hypothesis proposes that inventions arise from recombining existing natural ingredients, and asks whether intelligent systems can search those combinations systematically. In atomic materials discovery, this maps naturally to candidate generation, surrogate prediction, active selection, and validation against thermodynamic stability. The literature indicates that foundation atomistic models and active-learning loops can accelerate screening, but rigorous discovery claims require careful offline benchmarking and independent validation.

## Hypothesis Decomposition
- H1: Composition and structure descriptors contain enough signal to predict whether a candidate is stable or near-hull.
- H2: A model-guided acquisition policy discovers stable candidates at a higher rate than random selection with the same number of oracle evaluations.
- H3: Diversity-aware acquisition avoids overly narrow exploitation and improves cumulative discovery coverage across chemistries.
- H4: A model trained on Materials Project labels can at least partially rank GNoME candidates by stability, but external transfer will be weaker than in-domain active learning.

Independent variables are the selection policy, random seed, budget, and candidate dataset. Dependent variables are stable discoveries, precision among selected candidates, discovery acceleration factor, recall of available stable candidates, ROC/PR metrics, and subgroup/error patterns. Stable or near-hull is defined as energy above hull <= 0.05 eV/atom where available.

## Proposed Methodology

### Approach
Use an offline active-learning benchmark. Each candidate has descriptors available to the autonomous system and a hidden stability label from precomputed DFT or high-throughput calculations. The loop starts with a small random seed set, trains a surrogate classifier, selects the next batch to "validate," reveals labels, and repeats until the budget is exhausted.

This approach is chosen because real DFT or robotic synthesis is infeasible in a short automated run, but precomputed stability labels let us evaluate discovery efficiency objectively without simulating outcomes.

### Experimental Steps
1. Verify environment, GPU availability, dataset schema, missingness, and label prevalence.
2. Build leakage-controlled descriptors from atomic numbers, atom counts, cell geometry, composition diversity, and simple element statistics. Exclude energy-per-atom, formation energy, and energy-above-hull from model inputs.
3. Split Materials Project into a candidate pool and a held-out test set with fixed random seed.
4. Train baseline supervised classifiers and evaluate predictive signal with ROC-AUC, PR-AUC, F1, precision, and recall.
5. Run budgeted discovery loops across multiple random seeds for random selection, greedy surrogate exploitation, uncertainty-weighted acquisition, and diversity-aware acquisition.
6. Analyze cumulative stable discoveries, discovery acceleration factor, final precision, and confidence intervals across seeds.
7. Apply the trained ranking model to GNoME summaries using compatible features and evaluate whether top-ranked candidates have lower decomposition energy than random candidates.
8. Perform error analysis by number of elements, cell size, composition families, and uncertainty.

### Baselines
- Random selection: lower-bound trial-and-error baseline.
- Uncertainty sampling: selects candidates near model uncertainty; tests whether exploration alone is enough.
- Greedy exploitation: selects highest predicted stability probability.
- Diversity-aware acquisition: proposed structured policy that combines predicted stability, uncertainty, and chemical diversity.

### Evaluation Metrics
- Stable discovery count and stable precision among selected candidates.
- Discovery acceleration factor: selected stable precision divided by pool stable prevalence.
- Recall of all stable candidates in the searchable pool.
- ROC-AUC and average precision for surrogate ranking.
- Bootstrap or across-seed 95% confidence intervals for discovery outcomes.
- Effect sizes comparing proposed policy to random baseline.

### Statistical Analysis Plan
Use paired comparisons across the same random seeds and budgets. For final cumulative stable discoveries, use paired t-tests when differences are approximately normal and Wilcoxon signed-rank tests otherwise. Report p-values, confidence intervals, and Cohen's d. Use alpha = 0.05. Since multiple acquisition policies are compared to random, apply Holm correction for the main family of comparisons.

## Expected Outcomes
The hypothesis is supported if model-guided and diversity-aware policies discover substantially more stable candidates than random search, with statistically significant improvements and meaningful discovery acceleration. The hypothesis is weakened if all structured policies perform near random, if improvements only come from leakage-prone features, or if external transfer to GNoME ranking is negligible.

## Timeline and Milestones
- Setup and data audit: 10-20 minutes.
- Descriptor and model implementation: 30-45 minutes.
- Active-learning experiments and external ranking: 45-75 minutes.
- Statistical analysis and figures: 20-35 minutes.
- Final report and reproducibility validation: 20-30 minutes.

## Potential Challenges
- Label imbalance may make accuracy misleading, so PR-AUC and discovery precision are primary.
- Energy columns can leak the target, so they are excluded from model inputs.
- GNoME summaries do not include full structures locally, so external transfer will use composition and coarse summary descriptors only.
- ML stability labels are not experimental synthesis, so conclusions are about computational triage, not final invention.
- Heavy atomistic models may require additional setup; this study uses lightweight descriptors for reliable completion in one session.

## Success Criteria
The research succeeds if it produces a reproducible offline benchmark, actual active-learning results, statistical comparisons against baselines, visualizations, and a clear conclusion about whether structured autonomous search improves stable-material discovery under budget constraints.
