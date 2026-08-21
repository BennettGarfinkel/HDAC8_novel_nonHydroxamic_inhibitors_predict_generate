# data_prep provenance report

## Inputs

| source | format | status | rows read | rows kept | censored | compounds |
|---|---|---|---|---|---|---|
| hdac_docking_inhibition | long_format_csv | ok | 94,906 | 94,662 | 13,537 | 27,230 |

## Rows dropped, by input and reason

- **hdac_docking_inhibition**: target_unmapped 244

## Provenance (per source_file inside the master)

| source_file | rows | censored | duplicate rows dropped | targets | types | compounds contributed |
|---|---|---|---|---|---|---|
| pubchem_bioassay_hdac1.csv | 18,859 | 3,319 | 11,001 | HDAC1 | EC50, IC50, Kd, Ki | 10,883 |
| pubchem_bioassay_hdac6.csv | 16,330 | 2,133 | 9,855 | HDAC6 | EC50, IC50, Kd, Ki | 9,795 |
| bindingdb_hdac1.tsv | 13,708 | 2,566 | 564 | HDAC1 | EC50, IC50, Kd, Ki | 8,558 |
| bindingdb_hdac6.tsv | 12,306 | 1,596 | 327 | HDAC6 | EC50, IC50, Kd, Ki | 7,709 |
| hdac8_htvs_fragbreed_glide.csv | 11,036 | 0 | 0 | HDAC8 | docking_score | 11,036 |
| pubchem_bioassay_hdac8.csv | 9,921 | 1,771 | 4,974 | HDAC8 | EC50, IC50, Kd, Ki | 5,692 |
| bindingdb_hdac8.tsv | 5,780 | 1,200 | 233 | HDAC8 | EC50, IC50, Kd, Ki | 3,397 |
| chembl_hdac8_ic50_raw.csv | 4,773 | 952 | 4,126 | HDAC8 | IC50 | 3,018 |
| hdac6_docking_scores.sdf | 1,003 | 0 | 0 | HDAC6 | docking_score | 1,003 |
| hdac1_docking_scores.sdf | 946 | 0 | 0 | HDAC1 | docking_score | 946 |

31,080 rows were the same measurement reported by more than one source (PubChem mirrors ChEMBL and BindingDB). Each is counted once, so n_measurements reflects distinct measurements rather than distinct files. Rows, censored and compounds above are counted before that collapse.

## Redundant / superseded sources

- **REDUNDANT: `chembl_hdac8_ic50_raw.csv` inside `pubchem_bioassay_hdac8.csv` (HDAC8 IC50)** -- 2,904 of 3,018 compounds (96.2%) also appear in `pubchem_bioassay_hdac8.csv` (5,692 compounds); median |delta value| 0.000.
    - The identical measurements are already counted once (see the duplicate column above), so n_measurements is not inflated by this. What is left is a judgment call: `chembl_hdac8_ic50_raw.csv` adds little beyond `pubchem_bioassay_hdac8.csv`. Nothing was removed automatically.

## Outputs

| file | compounds | mean n_measurements | baseline | change |
|---|---|---|---|---|
| hdac8_ic50_clean_MERGED.csv | 6,198 | 1.39 | 6,198 | unchanged |
| hdac1_ic50_clean.csv | 11,566 | 1.45 | 11,566 | unchanged |
| hdac6_ic50_clean.csv | 10,369 | 1.50 | 10,369 | unchanged |
| hdac8_dock_clean_MERGED.csv | 11,036 | 1.00 | 11,036 | unchanged |
| hdac1_dock_clean.csv | 946 | 1.00 | 946 | unchanged |
| hdac6_dock_clean.csv | 1,003 | 1.00 | 1,003 | unchanged |

