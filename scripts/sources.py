"""
Declarative registry of data sources for data_prep.py.

Two modes, deliberately separated:

1. NORMAL RUN (`python data_prep.py`) reads exactly ONE file,
   data/HDAC_Docking_Inhibition.csv, and builds the six training CSVs from it.
   Nothing else has to exist on disk. That file is the long-format master: one row
   per measurement, with the provenance of every row in its source_file column.

2. REBUILD (`python data_prep.py --rebuild-master --raw-dir <dir>`) regenerates the
   master file from the original vendor downloads (BindingDB TSVs, PubChem bioassay
   exports plus a CID->SMILES map, ChEMBL exports, Glide SDF/CSV docking output).
   Those raw files are NOT kept in the repo. Run this only when new data arrives.

Adding a new source means adding a RawSource(...) entry below, or dropping a file
into the raw directory under a subfolder whose glob already maps to a format. It
never means editing the aggregation logic in data_prep.py.
"""
import os
import glob as globmod
from dataclasses import dataclass
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')          # optional; only used by --rebuild-master
MASTER_FILE = os.path.join(DATA_DIR, 'HDAC_Docking_Inhibition.csv')


# --------------------------------------------------------------------------
# Target canonicalization
# --------------------------------------------------------------------------
# Every target label a source can emit maps to one of the three canonical targets,
# or to nothing (in which case its rows are dropped and counted as 'target_unmapped'
# in the report, never silently discarded).
TARGET_ALIASES = {
    'hdac1': 'HDAC1',
    'histone deacetylase 1': 'HDAC1',
    'histone deacetylase 1 (human)': 'HDAC1',
    'hdac6': 'HDAC6',
    'histone deacetylase 6': 'HDAC6',
    'histone deacetylase 6 [482-800]': 'HDAC6',
    'histone deacetylase 6 (human)': 'HDAC6',
    'hdac8': 'HDAC8',
    'histone deacetylase 8': 'HDAC8',
    'histone deacetylase 8 (human)': 'HDAC8',
    'hdac8 - histone deacetylase 8 (human)': 'HDAC8',
    'chain a, histone deacetylase 8 (human)': 'HDAC8',
    # Docking runs against the HDAC8 structure (1T64), labelled by which screen
    # produced them. Both belong to HDAC8.
    'htvs': 'HDAC8',
    'fragbreed': 'HDAC8',
    # Deliberately absent: 'histone deacetylase 1/8'. BindingDB uses that label for
    # dual HDAC1/HDAC8 patent assays, where one IC50 cannot be assigned to either
    # target. Those rows are dropped as target_unmapped, on purpose.
}

# PubChem bioassay exports carry the target as an NCBI Gene ID, not a name.
GENE_ID_TARGETS = {
    3065: 'HDAC1',
    10013: 'HDAC6',
    55869: 'HDAC8',
}

SUPPORTED_ACTIVITY_TYPES = ('IC50', 'Ki', 'Kd', 'EC50', 'docking_score')
VALID_QUALIFIERS = ('=', '>', '<', '>=', '<=', '~')


# --------------------------------------------------------------------------
# Known-bad inputs (documentation only -- nothing here is enforced in code)
# --------------------------------------------------------------------------
# Kept as an audit trail so the history is not lost, and so nobody re-adds these by
# accident. The pipeline no longer blocks anything by filename; the redundancy check
# in data_prep.py is what catches duplicate and superseded data automatically.
#
#   BindingDB_HDAC1_HDAC6_HDAC8_combined.tsv
#       Not a real BindingDB export. Produced by an LLM scraping BindingDB, and it
#       contains at least one confirmed fabricated/mismatched structure. Use the
#       direct BindingDB downloads instead.
#   combined thing.csv / combined_thing.csv
#       Hand-assembled precursor of HDAC_Docking_Inhibition.csv with no
#       activity_qualifier column at all, so every censored value in it reads as an
#       exact measurement.
#   hdac8_ic50_clean_VERIFIED.csv
#       Strict subset of chembl_hdac8_ic50_raw.csv: all 2,975 of its compounds are in
#       chembl_raw, median pIC50 difference 0.0. Merging both double-counted
#       n_measurements. The redundancy check now catches this class on its own.
#   bgarfinkel74_claytonschools.net1371.tsv / .net1500.tsv / .net1502.tsv
#       Stale BindingDB re-downloads: 100%, 99.8% and 99.9% subsets of the 1370
#       (HDAC8) and 1373 (HDAC6) downloads that are actually used.


# --------------------------------------------------------------------------
# Source descriptors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    """One input file (or glob) and everything the pipeline needs to know about it."""
    name: str
    path: str                    # file or glob; relative paths resolve under base_dir
    fmt: str                     # key into parsers.PARSERS
    # 'from_column'   -> the parser reads the target out of the file
    # 'column_or_filename' -> use the file's column, falling back to the filename when
    #                    the column is blank; a disagreement between the two is a
    #                    target_conflict drop, never a guess
    # 'from_filename' -> canonical target inferred from hdac1/hdac6/hdac8 in the name
    # ('HDAC8',)      -> every row in the file belongs to these targets
    targets: object = 'from_column'
    enabled: bool = True         # trust flag: sources are opt-in
    needs_resolution: bool = False    # needs an external ID -> SMILES lookup
    resolution_status: str = 'n/a'    # 'n/a' | 'raw' (not yet resolved) | 'resolved'
    resolver: Optional[str] = None    # path to the ID -> SMILES map, relative to base_dir
    # What a blank/missing qualifier means FOR THIS SOURCE. None means "no policy":
    # blank-qualifier rows are dropped and counted, never assumed exact.
    qualifier_blank_means: Optional[str] = None
    expect_zero_contribution: bool = False   # documented, so it is not read as a bug
    notes: str = ''
    discovered: bool = False     # True if picked up by glob rather than registered
    base_dir: str = DATA_DIR

    def resolved_path(self):
        return self.path if os.path.isabs(self.path) else os.path.join(self.base_dir, self.path)

    def resolver_path(self):
        if not self.resolver:
            return None
        if os.path.isabs(self.resolver):
            return self.resolver
        return os.path.join(self.base_dir, self.resolver)


# --------------------------------------------------------------------------
# The only source a normal run needs
# --------------------------------------------------------------------------
MASTER_SOURCE = Source(
    name='hdac_docking_inhibition',
    path=MASTER_FILE,
    fmt='long_format_csv',
    targets='from_column',
    qualifier_blank_means=None,
    notes='Long-format master: one row per measurement, columns compound_name, smiles, '
          'target, activity_type, activity_value, activity_units, source_file, '
          'activity_qualifier. Every row carries an explicit qualifier, so the blank '
          'policy is None (a blank qualifier here means something went wrong upstream '
          'and the row is dropped and counted, not assumed exact). Regenerate with '
          '--rebuild-master when new data arrives; the per-row source_file column is '
          'what the provenance report and the redundancy check group on.',
)


# --------------------------------------------------------------------------
# Raw vendor downloads -- only read by --rebuild-master
# --------------------------------------------------------------------------
def _raw(**kwargs):
    kwargs.setdefault('base_dir', RAW_DIR)
    return Source(**kwargs)


RAW_SOURCES = [
    _raw(
        name='bindingdb_hdac8',
        path='bindingdb/bindingdb_hdac8.tsv',
        fmt='bindingdb_tsv',
        targets='from_column',
        qualifier_blank_means='=',
        notes='BindingDB direct download, Histone deacetylase 8. BindingDB carries '
              'censoring inside the value string ("<1", ">10000"), so a bare number is an '
              "exact measurement by BindingDB's own export convention: that is why blank "
              "means '=' here. 1,189 of its 5,575 HDAC8 IC50 values are censored and were "
              'being merged as exact before this pipeline existed.',
    ),
    _raw(
        name='bindingdb_hdac1',
        path='bindingdb/bindingdb_hdac1.tsv',
        fmt='bindingdb_tsv',
        qualifier_blank_means='=',
        notes='BindingDB direct download, Histone deacetylase 1.',
    ),
    _raw(
        name='bindingdb_hdac6',
        path='bindingdb/bindingdb_hdac6.tsv',
        fmt='bindingdb_tsv',
        qualifier_blank_means='=',
        notes='BindingDB direct download, Histone deacetylase 6. Includes the '
              'catalytic-domain construct "Histone deacetylase 6 [482-800]", mapped to HDAC6.',
    ),
    _raw(
        name='bindingdb_patent_hdac1_8',
        path='bindingdb/bindingdb_patent_hdac1_8.tsv',
        fmt='bindingdb_tsv',
        qualifier_blank_means='=',
        expect_zero_contribution=True,
        notes='Patent-derived BindingDB rows labelled "Histone deacetylase 1/8" (dual '
              'assay). Kept for provenance, but every row maps to no canonical target and '
              'is dropped as target_unmapped: a dual HDAC1/8 IC50 cannot be assigned to '
              'either model. Contributing 0 compounds is expected here, and is flagged as '
              'documented rather than as the silent-zero bug class.',
    ),
    _raw(
        name='pubchem_hdac1',
        path='pubchem/pubchem_bioassay_hdac1.csv',
        fmt='pubchem_bioassay_csv',
        targets='column_or_filename',
        needs_resolution=True,
        resolution_status='raw',
        resolver='pubchem/pubchem_cid_smiles.tsv',
        qualifier_blank_means='=',
        notes='PubChem protein-bioactivity export, Gene ID 3065. Structures arrive as CIDs '
              'only and are resolved through pubchem_cid_smiles.tsv. The export is one '
              'file per protein, and a few thousand rows have a blank Target GeneID, so '
              'the target falls back to the one the file was downloaded for.  BLANK-QUALIFIER '
              'DECISION: a blank Activity Qualifier is read as an exact value, on the '
              'standard scientific-reporting convention that an omitted censoring symbol '
              'means an exact measurement. That is a judgment call about this export, NOT a '
              'PubChem rule verified against PubChem documentation, and it does not '
              'generalize to other sources. Roughly a quarter of the IC50 rows here have a '
              'blank qualifier, so it is load-bearing.',
    ),
    _raw(
        name='pubchem_hdac6',
        path='pubchem/pubchem_bioassay_hdac6.csv',
        fmt='pubchem_bioassay_csv',
        targets='column_or_filename',
        needs_resolution=True,
        resolution_status='raw',
        resolver='pubchem/pubchem_cid_smiles.tsv',
        qualifier_blank_means='=',
        notes='PubChem protein-bioactivity export, Gene ID 10013. Same blank-qualifier '
              'judgment call as pubchem_hdac1.',
    ),
    _raw(
        name='pubchem_hdac8',
        path='pubchem/pubchem_bioassay_hdac8.csv',
        fmt='pubchem_bioassay_csv',
        targets='column_or_filename',
        needs_resolution=True,
        resolution_status='raw',
        resolver='pubchem/pubchem_cid_smiles.tsv',
        qualifier_blank_means='=',
        notes='PubChem protein-bioactivity export, Gene ID 55869. Same blank-qualifier '
              'judgment call as pubchem_hdac1.',
    ),
    _raw(
        name='chembl_hdac8',
        path='chembl/chembl_hdac8_ic50_raw.csv',
        fmt='chembl_export_csv',
        targets='from_column',
        qualifier_blank_means=None,
        notes="ChEMBL export, semicolon separated, Standard Relation quoted as \"'='\". A "
              'blank relation is NOT assumed exact here: in this export every blank-relation '
              'row also has a blank Standard Value, so those rows carry no measurement and '
              'are dropped. 952 of its 4,947 rows are censored (900 of them ">") and were '
              'being merged as exact before this pipeline existed.',
    ),
    _raw(
        name='hdac1_docking_sdf',
        path='docking/hdac1_docking_scores.sdf',
        fmt='docking_sdf',
        targets='from_filename',
        qualifier_blank_means='=',
        notes='Glide SDF, r_i_docking_score property, kcal/mol. Docking scores are point '
              'estimates and are never censored.',
    ),
    _raw(
        name='hdac6_docking_sdf',
        path='docking/hdac6_docking_scores.sdf',
        fmt='docking_sdf',
        targets='from_filename',
        qualifier_blank_means='=',
        notes='Glide SDF, r_i_docking_score property, kcal/mol.',
    ),
    _raw(
        name='hdac8_docking_glide',
        path='docking/hdac8_htvs_fragbreed_glide.csv',
        fmt='glide_csv',
        targets='from_column',
        qualifier_blank_means='=',
        notes='HTVS and FragBreed Glide runs against the HDAC8 1T64 structure. The run label '
              'in the "source" column is kept as the target string and maps to HDAC8.',
    ),
]


# --------------------------------------------------------------------------
# Auto-discovery for the rebuild path: glob -> format, once per FORMAT
# --------------------------------------------------------------------------
# A new file dropped into the raw directory under one of these patterns is ingested
# by --rebuild-master with no code change.
FORMAT_GLOBS = [
    ('bindingdb/*.tsv', 'bindingdb_tsv', dict(
        targets='from_column', qualifier_blank_means='=',
        notes='BindingDB direct download; censoring lives inside the value string.')),
    ('bgarfinkel74_*.tsv', 'bindingdb_tsv', dict(
        targets='from_column', qualifier_blank_means='=',
        notes='BindingDB direct download under its default export filename.')),
    ('pubchem/pubchem_bioassay_*.csv', 'pubchem_bioassay_csv', dict(
        targets='column_or_filename', needs_resolution=True, resolution_status='raw',
        resolver='pubchem/pubchem_cid_smiles.tsv', qualifier_blank_means='=',
        notes='See the registered PubChem entries for the blank-qualifier reasoning.')),
    ('chembl/*.csv', 'chembl_export_csv', dict(
        targets='from_column', qualifier_blank_means=None,
        notes='ChEMBL web-interface export (semicolon separated).')),
    ('docking/*.sdf', 'docking_sdf', dict(
        targets='from_filename', qualifier_blank_means='=',
        notes='Glide output SDF.')),
    ('docking/*_glide.csv', 'glide_csv', dict(
        targets='from_column', qualifier_blank_means='=',
        notes='Flat Glide results CSV.')),
    ('long_format/*.csv', 'long_format_csv', dict(
        targets='from_column', qualifier_blank_means=None,
        notes='Another long-format file in the master schema.')),
]


def discover_raw_sources(raw_dir=None, registered=None):
    """Registered raw sources plus anything new sitting in the raw directory."""
    raw_dir = raw_dir or RAW_DIR
    registered = [s if s.base_dir == raw_dir else _rebase(s, raw_dir)
                  for s in (RAW_SOURCES if registered is None else registered)]
    known = {os.path.normcase(os.path.abspath(s.resolved_path())) for s in registered}
    found = []
    for pattern, fmt, defaults in FORMAT_GLOBS:
        for hit in sorted(globmod.glob(os.path.join(raw_dir, pattern))):
            key = os.path.normcase(os.path.abspath(hit))
            if key in known:
                continue
            known.add(key)
            rel = os.path.relpath(hit, raw_dir).replace('\\', '/')
            found.append(Source(name=os.path.splitext(os.path.basename(hit))[0],
                                path=rel, fmt=fmt, discovered=True,
                                base_dir=raw_dir, **defaults))
    return registered + found


def _rebase(source, raw_dir):
    from dataclasses import replace
    return replace(source, base_dir=raw_dir)


def target_from_filename(path):
    name = os.path.basename(path).lower()
    for key in ('hdac1', 'hdac6', 'hdac8'):
        if key in name:
            return key.upper()
    return None


def canonical_target(raw):
    if raw is None:
        return None
    return TARGET_ALIASES.get(str(raw).strip().lower())
