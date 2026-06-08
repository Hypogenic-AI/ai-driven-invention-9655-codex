# Downloaded Datasets

This directory contains local datasets for the AI-driven atomic modeling resource phase. Large data files are intentionally excluded from Git by `datasets/.gitignore`; documentation and small samples remain trackable.

## Summary

| Dataset | Local path | Local size | Format | Main use |
|---|---:|---:|---|---|
| GNoME summaries | `datasets/gnome_summaries/` | 82 MB | CSV | Candidate stable crystal screening and convex-hull context |
| Materials Project HF snapshot | `datasets/materials_project_hf/` | 89 MB | HDF5 archive plus extracted HDF5 | Formation energy and energy-above-hull property modeling |
| JARVIS-DFT 3D HF snapshot | `datasets/jarvis_dft_3d_hf/` | 39 MB | Parquet | Crystal property/force/stress baseline data |
| OMat24 reference metadata | `datasets/omat24_references/` | 44 KB | YAML/JSON | Element reference corrections for OMat24-trained models |
| QM9 HF | `datasets/qm9_hf/` | 160 MB | Hugging Face Arrow | Molecular atomistic property baseline |

## Dataset 1: GNoME Summaries

### Overview

- Source: `https://storage.googleapis.com/gdm_materials_discovery/gnome_data/`
- Local files:
  - `stable_materials_summary.csv`: 554,054 rows, 24 columns
  - `stable_materials_r2scan.csv`: 139,541 rows, 22 columns
- Format: CSV
- Task: stable inorganic crystal candidate analysis, convex-hull filtering, candidate ranking
- License: GNoME data is documented by Google DeepMind as CC BY-NC 4.0
- Sample files:
  - `datasets/gnome_summaries/samples/stable_materials_summary.csv.sample.json`
  - `datasets/gnome_summaries/samples/stable_materials_r2scan.csv.sample.json`

### Download Instructions

The local copy contains summary CSVs only. Full structure ZIPs were not downloaded because each structure archive is about 455-472 MB.

```bash
mkdir -p datasets/gnome_summaries
wget https://storage.googleapis.com/gdm_materials_discovery/gnome_data/stable_materials_summary.csv -O datasets/gnome_summaries/stable_materials_summary.csv
wget https://storage.googleapis.com/gdm_materials_discovery/gnome_data/stable_materials_r2scan.csv -O datasets/gnome_summaries/stable_materials_r2scan.csv
```

For full CIF structures, use the cloned GNoME downloader:

```bash
python code/materials_discovery_gnome/scripts/download_data_wget.py --data_dir datasets/gnome_full
```

### Loading

```python
import pandas as pd

summary = pd.read_csv("datasets/gnome_summaries/stable_materials_summary.csv")
r2scan = pd.read_csv("datasets/gnome_summaries/stable_materials_r2scan.csv")
```

## Dataset 2: Materials Project Hugging Face Snapshot

### Overview

- Source: `materials-toolkits/materials-project` on Hugging Face
- Local files:
  - `materials-project.tar.gz`
  - `extracted/data.hdf5`
  - `extracted/batching.json`
- Size: 133,420 structures; extracted HDF5 is 104.48 MB
- Format: HDF5 arrays
- Task: formation-energy and energy-above-hull prediction from periodic structures
- Sample files:
  - `datasets/materials_project_hf/samples/archive_members_sample.json`
  - `datasets/materials_project_hf/samples/hdf5_schema_sample.json`

### Download Instructions

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="materials-toolkits/materials-project",
    repo_type="dataset",
    local_dir="datasets/materials_project_hf",
)
```

Extract:

```bash
tar -xzf datasets/materials_project_hf/materials-project.tar.gz -C datasets/materials_project_hf/extracted
```

### Loading

```python
import h5py

with h5py.File("datasets/materials_project_hf/extracted/data.hdf5", "r") as h5:
    material_ids = h5["data/material_id"][:]
    energy_above_hull = h5["data/energy_above_hull"][:]
    atomic_numbers = h5["data/z"][:]
    positions = h5["data/pos"][:]
```

## Dataset 3: JARVIS-DFT 3D Hugging Face Snapshot

### Overview

- Source: `colabfit/JARVIS_DFT_3D_8_18_2021` on Hugging Face
- Local files:
  - `co/co_0.parquet`: 56,627 configuration/property rows
  - `ds.parquet`: dataset metadata row
- Format: Parquet
- Task: crystal property prediction, formation energy, energy above hull, force/stress data inspection
- Sample files:
  - `datasets/jarvis_dft_3d_hf/samples/co_0.sample.json`
  - `datasets/jarvis_dft_3d_hf/samples/ds.sample.json`

### Download Instructions

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="colabfit/JARVIS_DFT_3D_8_18_2021",
    repo_type="dataset",
    local_dir="datasets/jarvis_dft_3d_hf",
)
```

### Loading

```python
import pandas as pd

df = pd.read_parquet("datasets/jarvis_dft_3d_hf/co/co_0.parquet")
```

## Dataset 4: OMat24 Reference Metadata

### Overview

- Source: `facebook/OMAT24` on Hugging Face
- Local files:
  - `README.md`
  - `references/element-references.yaml`
  - `references/omat-elemental-reference-compounds.json.gz`
- Size: 89 elemental reference entries
- Format: YAML and gzipped JSON
- Task: correct reference handling for OMat24-trained models, not a full OMat24 dataset mirror
- Sample file: `datasets/omat24_references/samples/elemental_reference_compounds_sample.json`

### Download Instructions

The full OMat24 dataset contains over 110M DFT calculations and should be pulled selectively from Hugging Face only when the experiment requires it.

```python
from huggingface_hub import hf_hub_download

for filename in [
    "README.md",
    "references/element-references.yaml",
    "references/omat-elemental-reference-compounds.json.gz",
]:
    hf_hub_download("facebook/OMAT24", filename, repo_type="dataset")
```

## Dataset 5: QM9 Hugging Face

### Overview

- Source: `yairschiff/qm9` on Hugging Face
- Local rows: 133,885 molecules
- Format: Hugging Face Arrow
- Task: small-molecule atomistic property prediction baseline, useful for medicine/chemistry-adjacent experiments
- Sample file: `datasets/qm9_hf/samples/qm9_first5.json`

### Download Instructions

```python
from datasets import load_dataset

dataset = load_dataset("yairschiff/qm9", split="train")
dataset.save_to_disk("datasets/qm9_hf")
```

### Loading

```python
from datasets import load_from_disk

qm9 = load_from_disk("datasets/qm9_hf")
```

## Validation Notes

- `datasets/validation_summary.json` contains the local validation summary but is excluded from Git by design.
- No full GNoME structure ZIPs, full OMat24 training shards, or model checkpoints were downloaded because they are large and not needed for an initial experiment runner.
- For experiments requiring full structures, prioritize targeted downloads by chemical system or benchmark split rather than bulk archive mirroring.
