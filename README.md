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
| `scripts/ga.py` | ZBG-locked mutation/crossover, ZBG-periphery mutation, NSGA-II Pareto GA with docking-selectivity objective + threshold gate, IC50-selectivity flag, lineage-protected elite selection |
| `scripts/data_prep.py` | Consolidates the 73k-row merged dataset into clean per-target training sets |
| `scripts/train_all_models.py` | **Optional retrain entry point**: retrains all six models from `data/` and rebuilds `models/run_state.pkl` |
| `scripts/run_docking_selective.py` | Generation run: reuses trained models, applies docking gate + IC50 selectivity, runs one or more execution modes, exports candidates |
| `scripts/make_figures.py` | Generates all nine figures from the candidate CSVs + model bundles |
| `notebooks/hHDAC8_docking_selective.ipynb` | Full runnable notebook: (optional retrain) → load models → (optional regenerate) GA → assembly → figures → `predict_from_smiles()` |
| `models/run_state.pkl` | Trained model bundles (HDAC8 IC50/docking, HDAC1/HDAC6 IC50/docking) + `ic50_df`. Shared by every execution mode |
| `models/docking_selective_run_state.pkl` | Generation-run results for all modes, keyed by mode name. Each slot holds `hits_df`, `balanced`, `reps`, `final_screen`, `saved_at`. No model bundles (see below) |
| `candidates/<mode>/*.csv` | GA output per mode: raw hits, balanced primary deliverable, cluster representatives, strict clean screen |
| `figures/<mode>/*.png` | Nine diagnostic visualizations (fig1–fig9) per mode |

A fresh `run_docking_selective.py` / `make_figures.py` run writes to
`candidates/<mode>/` and `figures/<mode>/` (created if they don't exist yet), one
subfolder per execution mode, so modes never overwrite each other. Once I've reviewed a
run — and ideally spot-checked it against real docking, like in `Run 1/Run1.md` — I
archive its `candidates/` and `figures/` output into a numbered `Run N/` folder (`Run N
Generation/`, `Run N Figures/`, `Run N.md` summarizing what changed and any real
validation results), so I can compare generation runs side by side. See `Run 1/`,
`Run 2/`, `Run 3/` for the archived runs so far.

## Execution modes

The generation run has three modes, differing only in how the GA proposes new molecules.
Everything else (Tier-1 gates, Pareto selection, docking/IC50 selectivity, AD gating,
clustering, export) is identical across all three.

| Mode | Crossover | ZBG-periphery mutation | Lineages searched |
|---|---|---|---|
| `baseline` | 0.5 | off | all 8 |
| `separate_zbg` | disabled | always | the 5 in `ZBG_PERIPHERY_ELIGIBLE` |
| `integrated` | 0.5 | 5% of mutation events | all 8, periphery only on eligible |

**ZBG-periphery mutation** restricts substitution to atoms bonded to a ZBG-matched atom
but not part of the ZBG match themselves, meaning the anchor positions where cap and
linker chemistry attaches to the warhead. Normal `mutate()` picks any free-H atom in the
molecule; this one targets the region that actually determines how the cap sits relative
to the zinc. `ZBG_PERIPHERY_ELIGIBLE` is restricted to the five ZBGs with real measured
HDAC8 data (Ortho-aminoanilide 384 rows, Carboxylate 136, Cyclic-thione 50,
alpha-amino-amide 29, 3-HPT 27), since periphery mutation on a chemotype with no
measured precedent has nothing to anchor to.

`separate_zbg` runs periphery mutation in isolation to see what it produces on its own.
`integrated` blends it into the normal search. `baseline` is the pre-existing behavior,
kept so the other two have something to compare against; passing the default dial
settings reproduces the pre-refactor RNG stream exactly.

Set the switches at the top of `scripts/run_docking_selective.py`:

```python
RUN_ALL = True             # default: run all three back to back
RUN_BASELINE = False       # only consulted when RUN_ALL is False
RUN_SEPARATE_ZBG = False
RUN_INTEGRATED = False
```

`RUN_ALL = True` runs all three in order. To run a subset, set `RUN_ALL = False` and turn
on the individual switches you want. Each mode reseeds `RANDOM_SEED` before it starts, so
any mode is reproducible on its own regardless of which others ran or in what order.

Results go into one shared `models/docking_selective_run_state.pkl`, keyed by mode and
updated in place after each mode finishes, rather than one file per mode. A mode's save
replaces only its own slot and leaves the others alone, and the write goes through a temp
file so an interrupted save can't destroy results from modes that already completed.
Model bundles are deliberately not stored there: no mode retrains anything, so all six
are identical across modes and already live in `run_state.pkl`, which is where
`make_figures.py` reads them from. A pre-refactor pickle is migrated into the `baseline`
slot automatically on first read.

If a mode produces no surviving candidates, it prints why, skips its export and figures,
and the run continues to the next mode rather than aborting.

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
| Acylurea | Kept, flagged | Current FragBreed lead chemotype; ZBG mechanism unconfirmed in literature (see pipeline.py) |
| 3-HPT | Kept, flagged | Pyrithione-adjacent risk; open SAR area |
| Carboxylate | Added | Metabolically inert, weak/reversible chelator |
| Trifluoromethyl-ketone | Added | Reversible hydrate-forming ZBG |
| Cyclic-thione | Added | Generalizes 3-HPT; 101 HDAC8 IC50 rows (50 unique compounds) |
| alpha-amino-amide | Added | Whitehead 2011/Greenwood 2020 bidentate ZBG; 72 HDAC8 IC50 rows (60 unique compounds) |

Counts are HDAC8 IC50 rows in `data/HDAC_Docking_Inhibition.csv`. After deduplication,
hydroxamate removal, and measurement averaging, the cleaned training set
(`hdac8_ic50_clean_MERGED.csv`) carries fewer per chemotype: Ortho-aminoanilide 384,
Carboxylate 136, Cyclic-thione 50, alpha-amino-amide 29, 3-HPT 27,
Trifluoromethyl-ketone 23, Acylurea 8, Salicylamide 6. Those cleaned numbers are what
`ZBG_PERIPHERY_ELIGIBLE` in `ga.py` is based on.

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
│   ├── baseline/
│   ├── separate_zbg/
│   └── integrated/
└── figures/                   # not in repo, created automatically on first run
    ├── baseline/
    ├── separate_zbg/
    └── integrated/
```

Mode subfolders appear only for modes that have actually been run.

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib "scikit-learn==1.9.0" rdkit jupyter ipykernel
```

Pin `scikit-learn==1.9.0`. The shipped `models/*.pkl` files were pickled under 1.9.0, and
newer sklearn versions restructure `HistGradientBoostingRegressor`'s internal loss
classes, which breaks `pickle.load()` on the shipped models. `sascorer` needs no separate
install; it's pulled from RDKit's own `Contrib/SA_Score`.

### 3. Launch and run

```bash
cd notebooks
jupyter notebook hHDAC8_docking_selective.ipynb
# Run All
```

The notebook has three switches near the top:

- **`RETRAIN`** (`False`): retrain all six models from `../data/` and rebuild
  `../models/run_state.pkl` (a few minutes).
- **`REGENERATE`** (`False`): re-run the full eight-island GA (about 25 minutes).
- **`RUN_MODE`** (`None`): which execution mode's results to work with. Left as `None`,
  it loads whichever mode was saved most recently, so running the script and then running
  the notebook needs no edits. Set it to `'baseline'`, `'separate_zbg'`, or `'integrated'`
  to pin one. When regenerating with `RUN_MODE = None` it defaults to `'baseline'`, since
  there is no completed run to infer intent from.

With `RETRAIN` and `REGENERATE` left `False`, the notebook loads the cached models and the
cached GA hit pool for the selected mode, rebuilds all the deliverables and all nine
figures, and defines `predict_from_smiles()`. The whole run finishes in under a minute and
reproduces the shipped results. It reads from `../models/`, imports from `../scripts/`,
and writes out to `../candidates/<mode>/` and `../figures/<mode>/`.

The notebook imports `load_mode_run_state` and `save_mode_run_state` from
`run_docking_selective.py` rather than reimplementing them, so the notebook and the script
cannot drift apart on the run-state format.

To retrain or regenerate from the command line instead:

```bash
cd scripts
python3 train_all_models.py        # rebuild ../models/run_state.pkl from ../data
python3 run_docking_selective.py   # run the enabled modes, export candidates + figures
python3 make_figures.py            # regenerate figures from candidate CSVs (defaults to
                                   # the un-suffixed ../candidates and ../figures paths;
                                   # pass cand_dir/fig_dir for a specific mode)
```

`run_docking_selective.py` generates each enabled mode's figures itself, so
`make_figures.py` only needs running standalone if you want to regenerate figures without
re-running the GA.

## Known limitations
- Lack of training data results in lwo R^2 values, especially for HDAC 1 & 6 docking
- Values should be tested using software such as Maestro before in-vitro assays
