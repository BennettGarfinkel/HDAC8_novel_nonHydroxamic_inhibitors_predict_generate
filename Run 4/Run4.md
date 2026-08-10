## Run 4

Retrained models and archived the Run 1-3 versions to `models/Models Runs 1-3/`. Fixed a bug
where `run_docking_selective.py` wrote its CSVs and pickle to the scripts folder instead of
`../candidates/` and `../models/`.

Raw GA pool only produced hits for 5 of 9 ZBG lineages this run (3-HPT, Acylurea,
Cyclic-thione, Triazolopyridine, Trifluoromethyl-ketone) -- Salicylamide, Ortho-aminoanilide,
Carboxylate, and Triazoloquinoline came back empty. Run 3 had hits across at least 8 lineages,
so this is a bigger drop than expected and needs checking, not just retrain noise.

`final_screened_candidates.csv` came back with 0 rows (Run 1 had 6, Run 3 had 20).
