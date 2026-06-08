# AI-Driven Invention: Atomic Modeling Benchmark

This workspace tests whether structured autonomous selection can discover stable inorganic materials more efficiently than random trial-and-error. The experiment uses real Materials Project stability labels as an offline oracle and evaluates transfer ranking on GNoME R2SCAN summaries.

## Key Findings

- A diversity-aware surrogate policy found 1,933 stable or near-hull candidates on average, versus 1,365 for random selection at the same 2,400-candidate budget.
- Composition plus cell descriptors achieved ROC-AUC 0.874 and average precision 0.891 for stability prediction.
- Model-guided policies were statistically stronger than random after Holm correction.
- Pure uncertainty sampling underperformed random, showing that exploration alone is not a discovery objective.
- GNoME transfer ranking improved top on-hull rates, but gains were modest because GNoME is already highly enriched for stable candidates.

See `REPORT.md` for the full methodology, results, figures, and limitations.

## Reproduce

The project uses an isolated `uv` environment in `.venv`.

```bash
source .venv/bin/activate
python src/run_discovery_experiments.py
```

To rebuild the Materials Project feature cache:

```bash
source .venv/bin/activate
python src/run_discovery_experiments.py --force-feature-cache
```

Expected full runtime on this machine was about 165 seconds using one NVIDIA RTX A6000 GPU.

## File Structure

| Path | Description |
|---|---|
| `planning.md` | Preregistered research plan |
| `src/run_discovery_experiments.py` | Main experiment script |
| `results/` | CSV/JSON outputs and cached features |
| `figures/` | Generated plots |
| `REPORT.md` | Full research report |
| `CODE_WALKTHROUGH.md` | Code structure and reproduction notes |
| `literature_review.md` | Pre-gathered literature synthesis |
| `resources.md` | Resource catalog and execution notes |

## Important Scope Note

This benchmark supports AI-guided computational triage, not a final claim of newly invented or synthesized materials. Physical invention claims require DFT validation, novelty checks, synthesis planning, and experimental characterization.
