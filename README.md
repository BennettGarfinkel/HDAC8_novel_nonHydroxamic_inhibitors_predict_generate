# hHDAC8 Non-Hydroxamate Inhibitor Discovery Pipeline

Computational drug discovery pipeline for finding potent, drug-like, isoform-selective
non-hydroxamate inhibitors of human HDAC8 (hHDAC8). It combines ML-based QSAR/docking-score
prediction with genetic-algorithm generative chemistry, with docking-based selectivity over
HDAC1/HDAC6 built in from the start.

## Project constraints and goals

- **Hard constraint: no hydroxamic acid zinc-binding groups (ZBGs).** Most other
  non-hydroxamate ZBG chemotype are fine.
- **Goal:** find candidates with real potency (aiming for ≤150–500 nM depending on how
  strict the triage stage is), acceptable drug-likeness (Lipinski/Veber), synthetic
  accessibility, no reactive/toxic liabilities, and evidence of HDAC8 selectivity over
  HDAC1/HDAC6 — both in predicted IC50 and in predicted docking score.
- **Why this matters:** HDAC8-selective inhibition avoids the toxicity that comes with
  pan-HDAC inhibitors, and is relevant to pediatric T-cell ALL, neuroblastoma, Cornelia
  de Lange syndrome (SMC3/cohesin deacetylation), and antiparasitic applications.

## Architecture

Cap–linker–ZBG (L-shaped) design paradigm. Key PDB structures: 1T64 (HDAC8), 4BKX
(HDAC1), 5EEM (HDAC6 CD2).

## Pipeline components

| File | Purpose |
|---|---|
| `scripts/pipeline.py` | ZBG whitelist + Tier-1 hard gates, liability engine, data cleaning, IC50/docking model training (`train_model_from_df` generic trainer), single/batch SMILES prediction, ZBG-precedent tracking, potency tiering, 2D L-shape proxy |
| `scripts/ga.py` | ZBG-locked mutation/crossover, NSGA-II Pareto GA with docking-selectivity objective + threshold gate, IC50-selectivity flag, lineage-protected elite selection |
| `scripts/data_prep.py` | Consolidates the 73k-row merged dataset into clean per-target training sets |
| `scripts/train_all_models.py` | **Optional retrain entry point**: retrains all six models from `data/` and rebuilds `models/run_state.pkl` |
| `scripts/run_docking_selective.py` | Generation run: reuses trained models, applies docking gate + IC50 selectivity, exports candidates |
| `scripts/make_figures.py` | Generates all nine figures from the candidate CSVs + model bundles |
| `notebooks/hHDAC8_docking_selective.ipynb` | Full runnable notebook: (optional retrain) → load models → (optional regenerate) GA → assembly → figures → `predict_from_smiles()` |
| `models/run_state.pkl` | Trained model bundles (HDAC8 IC50/docking, HDAC1/HDAC6 IC50/docking) + `ic50_df` |
| `models/docking_selective_run_state.pkl` | Full state from the generation run (hits_df + all model bundles) |
| `candidates/*.csv` | GA output: raw hits, balanced primary deliverable, cluster representatives, strict clean screen |
| `figures/*.png` | Nine diagnostic visualizations (fig1–fig9) |

A fresh `run_docking_selective.py` / `make_figures.py` run always writes to top-level
`candidates/` and `figures/` (created if they don't exist yet). Once I've reviewed a run
— and ideally spot-checked it against real docking, like in `Run 1/Run1.md` — I archive
its `candidates/` and `figures/` output into a numbered `Run N/` folder (`Run N
Generation/`, `Run N Figures/`, `Run N.md` summarizing what changed and any real
validation results), so I can compare generation runs side by side. See `Run 1/`,
`Run 2/`, `Run 3/` for the archived runs so far.

## Docking + IC50 selectivity gate

The GA uses a **docking-score-based isoform selectivity** system:

1. **Docking gate (hard, defines a qualifying hit):** HDAC8 predicted docking must be
   **≤ -7** (strong on-target binding, and not below a -13 applicability-domain floor),
   while HDAC1 and HDAC6 predicted docking must each be **> -7** (weak off-target
   binding). 

2. **IC50 selectivity (reported and preferred, but not a hard hit gate):** off-target
   predicted IC50 ≥ 1000 nM and HDAC8 ≥ 0.7 log more potent than each off-target sets a
   `passes_ic50_selectivity` flag. I tried making this a hard gate and it collapsed
   yield down to about 1 hit per lineage, since HDAC8's own median training IC50 is
   ~920 nM to begin with. So instead the `selectivity` Pareto objective pushes the
   search toward IC50-selective molecules, and the flag itself is only enforced only when a
   fully-selective subset is  wanted.

3. **Pareto objective:** `docking_selectivity = min(dock_HDAC1, dock_HDAC6) - dock_HDAC8`
   is maximized alongside pIC50, HDAC8 docking, liability score, and IC50-based
   selectivity.

The docking gate is deliberately *not* a hard rejection inside the GA loop itself —
that would freeze out any lineage whose seeds start outside the target region (this was
a real bug I found and fixed). It only gates what counts as a returned hit; the Pareto
objective is what actually drives the population toward selectivity across generations.

## ZBG whitelist (safety-audited)

| ZBG | Status | Rationale |
|---|---|---|
| Hydroxamate | Hard-excluded | Project constraint |
| Hydrazide | Removed | N-N bond, oxidative cleavage, hydrazine-release precedent |
| Thiol (free) | Removed | Oxidizes to reactive disulfide |
| Benzisothiazolone | Removed | Ring-opens to reactive electrophile |
| 8-Hydroxyquinoline | Removed | Promiscuous non-Zn metal chelation; neurotoxicity precedent |
| Salicylamide | Kept | Established HDAC8 ZBG |
| Ortho-aminoanilide | Kept | Clinically precedented (entinostat/mocetinostat class) |
| Acylurea | Kept | Current FragBreed lead chemotype |
| 3-HPT | Kept, flagged | Pyrithione-adjacent risk; open SAR area |
| Carboxylate | Added | Metabolically inert, weak/reversible chelator |
| Trifluoromethyl-ketone | Added | Reversible hydrate-forming ZBG |
| Cyclic-thione | Added | Generalizes 3-HPT; 72+ real examples |
| Triazolopyridine | Added | Published non-hydroxamate HDAC8 ZBG |
| Triazolo[4,3-a]quinoline | Added | Published non-hydroxamate HDAC8 ZBG |

## Data

- `data/HDAC_Docking_Inhibition.csv` — merged 73,467-row dataset
- `data/hdac8_ic50_clean_MERGED.csv` — 6,082 unique HDAC8 IC50 compounds
- `data/hdac1_ic50_clean.csv` / `data/hdac6_ic50_clean.csv` — 10,340 / 9,035 off-target IC50
- `data/hdac1_dock_clean.csv` / `data/hdac6_dock_clean.csv` — 946 / 1,003 off-target docking

Model performance (scaffold-grouped CV R²): HDAC8 IC50 0.54, HDAC8 docking 0.43, HDAC1
IC50 0.51, HDAC6 IC50 0.50. HDAC1/HDAC6 docking models are modest (0.13/0.15) due to
small training sets with high scaffold diversity — treat as directional, not precise.

## Output columns

Every candidate CSV includes: `SMILES`, `zbg_tag`, `pIC50_pred`, `IC50_nM_est`,
`IC50_tier`, `docking_pred`, `docking_hdac1_pred`, `docking_hdac6_pred`,
`docking_selectivity`, `pIC50_hdac1_pred`, `pIC50_hdac6_pred`, `selectivity_vs_hdac1/6`,
`passes_ic50_selectivity`, `MW`, `cLogP`, `HBD`, `HBA`, `TPSA`, `RotB`, `sa_score`,
`pains_brenk_flagged`, plus applicability-domain and precedent flags.

## Setup and running the notebook

### 1. Folder structure

The notebook navigates via `../`, so it expects this layout relative to the repo root:

```
repo-root/
├── data/                     # required if RETRAIN=True
│   ├── HDAC_Docking_Inhibition.csv
│   ├── hdac1_dock_clean.csv
│   ├── hdac1_ic50_clean.csv
│   ├── hdac6_dock_clean.csv
│   ├── hdac6_ic50_clean.csv
│   ├── hdac8_dock_clean_MERGED.csv
│   └── hdac8_ic50_clean_MERGED.csv
├── models/                   # required if RETRAIN=False / REGENERATE=False
│   ├── run_state.pkl
│   └── docking_selective_run_state.pkl
├── scripts/
│   ├── pipeline.py
│   ├── ga.py
│   ├── data_prep.py
│   ├── train_all_models.py
│   ├── run_docking_selective.py
│   └── make_figures.py
├── notebooks/
│   └── hHDAC8_docking_selective.ipynb
├── candidates/                # not in repo, created automatically on first run
└── figures/                   # not in repo, created automatically on first run
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib "scikit-learn==1.8.0" rdkit jupyter ipykernel
```

Pin `scikit-learn==1.8.0`. The shipped `models/*.pkl` files were pickled under 1.8.0, and
newer sklearn versions restructure `HistGradientBoostingRegressor`'s internal loss
classes, which breaks `pickle.load()` on the shipped models. `sascorer` needs no separate
install; it's pulled from RDKit's own `Contrib/SA_Score`.

### 3. Launch and run

```bash
cd notebooks
jupyter notebook hHDAC8_docking_selective.ipynb
# Run All
```

The notebook has two optional switches near the top, both `False` by default:

- **`RETRAIN`**: retrain all six models from `../data/` and rebuild
  `../models/run_state.pkl` (a few minutes).
- **`REGENERATE`**: re-run the full nine-island GA (about 25 minutes).

With both left `False`, the notebook loads the cached models and the cached GA hit pool,
rebuilds all the deliverables and all nine figures, and defines `predict_from_smiles()`.
The whole run finishes in under a minute and reproduces the shipped results. It reads
from `../models/`, imports from `../scripts/`, and writes out to `../candidates/` and
`../figures/`.

To retrain or regenerate from the command line instead:

```bash
cd scripts
python3 train_all_models.py        # rebuild ../models/run_state.pkl from ../data
python3 run_docking_selective.py   # re-run the GA and export candidates
python3 make_figures.py            # regenerate ../figures from the candidate CSVs
```

## Known limitations
- Lack of training data results in lwo R^2 values, especially for HDAC 1 & 6 docking
- Values should be tested using software such as Maestro before in-vitro assays
