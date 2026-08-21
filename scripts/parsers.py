"""
Format-specific parsers. Each one translates a single raw file into the standard
intermediate schema and does nothing else -- no filtering by potency, no
canonicalization, no aggregation, no target-specific special cases.

Standard intermediate schema (one row per measurement):
    smiles              raw structure string, canonicalized later by shared code
    target              raw target label, mapped later by sources.canonical_target
    activity_type       IC50 / Ki / Kd / EC50 / docking_score
    activity_value      float, nM for affinities, kcal/mol for docking scores
    activity_qualifier  '=', '>', '<', '>=', '<=', '~', or None (meaning the file
                        left it blank -- the source's qualifier_blank_means policy
                        decides what that means, the parser never assumes)
    compound_name       free-text name where the source has one, else None
    source_file         provenance label (see SOURCE_FILE_LABELS)
    n_measurements      always 1 here; aggregation happens downstream
    parse_note          optional drop reason the shared validator should use
                        instead of its generic one (e.g. unresolved_pubchem_cid)

Every parser returns (DataFrame, rows_read) where rows_read counts records in the
file, so the report can show "read 5,781 -> kept 4,386" per source.
"""
import os
import re
import numpy as np
import pandas as pd

STANDARD_COLUMNS = ['compound_name', 'smiles', 'target', 'activity_type', 'activity_value',
                    'activity_qualifier', 'source_file', 'n_measurements', 'parse_note']

_QUALIFIED_VALUE = re.compile(r'^\s*(>=|<=|>|<|~)?\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$')


def empty_frame():
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def parse_qualified_value(raw):
    """Split a possibly-censored value such as '>10000' into ('>', 10000.0).

    Returns (None, None) if the string carries no usable number. A bare number
    yields (None, value): the caller's source policy decides whether a missing
    qualifier means 'exact' or means 'unknown'.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, None
    if isinstance(raw, (int, float)):
        return None, float(raw)
    m = _QUALIFIED_VALUE.match(str(raw).replace(',', ''))
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


TEXT_COLUMNS = ['compound_name', 'smiles', 'target', 'activity_type',
                'activity_qualifier', 'parse_note']


def _frame(rows, path):
    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS) if rows else empty_frame()
    df['source_file'] = os.path.basename(path)
    df['n_measurements'] = 1
    # Missing text is None, never NaN: downstream code and the tests both read
    # "this source left it blank" off a single sentinel.
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


# --------------------------------------------------------------------------
# BindingDB direct download (TSV, ~640 columns, one row per curated entry)
# --------------------------------------------------------------------------
BINDINGDB_VALUE_COLUMNS = {
    'IC50 (nM)': 'IC50',
    'Ki (nM)': 'Ki',
    'Kd (nM)': 'Kd',
    'EC50 (nM)': 'EC50',
}


def parse_bindingdb_tsv(path, source=None):
    wanted = ['Ligand SMILES', 'Target Name', 'BindingDB Ligand Name'] + list(BINDINGDB_VALUE_COLUMNS)
    df = pd.read_csv(path, sep='\t', low_memory=False, dtype=str,
                     usecols=lambda c: c in wanted)
    rows_read = len(df)
    for col in wanted:
        if col not in df.columns:
            df[col] = np.nan

    rows = []
    for col, act_type in BINDINGDB_VALUE_COLUMNS.items():
        sub = df[df[col].notna()]
        for smi, tgt, name, raw in zip(sub['Ligand SMILES'], sub['Target Name'],
                                       sub['BindingDB Ligand Name'], sub[col]):
            qual, val = parse_qualified_value(raw)
            rows.append({
                'compound_name': name,
                'smiles': smi, 'target': tgt, 'activity_type': act_type,
                'activity_value': val, 'activity_qualifier': qual,
                'parse_note': None if val is not None else 'unparseable_value',
            })
    return _frame(rows, path), rows_read


# --------------------------------------------------------------------------
# PubChem protein-bioactivity export (CID only; needs an ID -> SMILES map)
# --------------------------------------------------------------------------
_PUBCHEM_TYPES = {'IC50', 'Ki', 'Kd', 'EC50'}
_resolver_cache = {}


def load_cid_smiles_map(path):
    """CID -> SMILES map, as downloaded from PubChem's ID exchange (TSV, no header)."""
    key = os.path.abspath(path)
    if key not in _resolver_cache:
        m = pd.read_csv(path, sep='\t', header=None, names=['cid', 'smiles'],
                        dtype={'cid': 'Int64', 'smiles': str})
        m = m.dropna(subset=['cid', 'smiles'])
        _resolver_cache[key] = dict(zip(m['cid'].astype(int), m['smiles']))
    return _resolver_cache[key]


def parse_pubchem_bioassay_csv(path, source=None):
    from sources import GENE_ID_TARGETS

    df = pd.read_csv(path, low_memory=False)
    rows_read = len(df)

    resolver_path = source.resolver_path() if source is not None else None
    cid_map = (load_cid_smiles_map(resolver_path)
               if resolver_path and os.path.exists(resolver_path) else {})

    sub = df[df['Activity Name'].isin(_PUBCHEM_TYPES)]
    rows = []
    for _, r in sub.iterrows():
        cid = r.get('CID')
        smi, note = None, None
        if pd.isna(cid):
            note = 'missing_pubchem_cid'
        else:
            smi = cid_map.get(int(cid))
            if smi is None:
                note = 'unresolved_pubchem_cid'
        gene = r.get('Target GeneID')
        target = GENE_ID_TARGETS.get(int(gene)) if pd.notna(gene) else None
        qual, val_um = parse_qualified_value(r.get('Activity Value [uM]'))
        raw_qual = r.get('Activity Qualifier')
        if isinstance(raw_qual, str) and raw_qual.strip():
            qual = raw_qual.strip()
        rows.append({
            'compound_name': r.get('Compound_Name'),
            'smiles': smi,
            'target': target,
            'activity_type': r['Activity Name'],
            # PubChem reports these assays in micromolar; the standard schema is nM.
            'activity_value': None if val_um is None else val_um * 1000.0,
            'activity_qualifier': qual,
            'parse_note': note,
        })
    return _frame(rows, path), rows_read


# --------------------------------------------------------------------------
# ChEMBL web-interface export (semicolon separated, quoted relation strings)
# --------------------------------------------------------------------------
def parse_chembl_export_csv(path, source=None):
    df = pd.read_csv(path, sep=';', low_memory=False, dtype=str)
    rows_read = len(df)
    rows = []
    for _, r in df.iterrows():
        qual = r.get('Standard Relation')
        if isinstance(qual, str):
            qual = qual.strip().strip("'\"").strip() or None
        else:
            qual = None
        units = r.get('Standard Units')
        _, val = parse_qualified_value(r.get('Standard Value'))
        note = None
        if val is not None and (not isinstance(units, str) or units.strip() != 'nM'):
            # Only nM is convertible without guessing; anything else is reported,
            # not silently rescaled.
            val, note = None, 'unsupported_units'
        rows.append({
            'compound_name': r.get('Molecule Name') or r.get('Compound Key'),
            'smiles': r.get('Smiles'),
            'target': r.get('Target Name'),
            'activity_type': r.get('Standard Type'),
            'activity_value': val,
            'activity_qualifier': qual,
            'parse_note': note,
        })
    return _frame(rows, path), rows_read


# --------------------------------------------------------------------------
# Glide docking output
# --------------------------------------------------------------------------
def parse_docking_sdf(path, source=None):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(path, sanitize=True)
    rows, rows_read = [], 0
    for i, mol in enumerate(supplier):
        rows_read += 1
        if mol is None:
            rows.append({'smiles': None, 'target': None, 'activity_type': 'docking_score',
                         'activity_value': None, 'activity_qualifier': '=',
                         'parse_note': f'rdkit_sdf_record_{i}_unreadable'})
            continue
        score = mol.GetProp('r_i_docking_score') if mol.HasProp('r_i_docking_score') else None
        _, val = parse_qualified_value(score)
        rows.append({
            'compound_name': mol.GetProp('_Name') if mol.HasProp('_Name') else None,
            'smiles': Chem.MolToSmiles(mol),
            'target': None,
            'activity_type': 'docking_score',
            'activity_value': val,
            'activity_qualifier': '=',
            'parse_note': None if val is not None else 'missing_docking_score',
        })
    return _frame(rows, path), rows_read


def parse_glide_csv(path, source=None):
    df = pd.read_csv(path, low_memory=False)
    rows_read = len(df)
    rows = []
    for _, r in df.iterrows():
        _, val = parse_qualified_value(r.get('r_i_docking_score'))
        rows.append({
            'compound_name': r.get('title'),
            'smiles': r.get('SMILES'),
            # HTVS / FragBreed: which screen produced the pose. Both map to HDAC8.
            'target': r.get('source'),
            'activity_type': 'docking_score',
            'activity_value': val,
            'activity_qualifier': '=',
            'parse_note': None if val is not None else 'missing_docking_score',
        })
    return _frame(rows, path), rows_read


# --------------------------------------------------------------------------
# Already-normalized long-format CSV (the schema this pipeline itself emits)
# --------------------------------------------------------------------------
def parse_long_format_csv(path, source=None):
    df = pd.read_csv(path, low_memory=False)
    rows_read = len(df)
    if 'activity_qualifier' not in df.columns:
        # A long-format file with no qualifier column tracked no censoring at all.
        # Leaving it blank hands the decision to the source's policy instead of
        # quietly promoting every censored row to an exact measurement.
        df['activity_qualifier'] = None
    rows = []
    for _, r in df.iterrows():
        units = r.get('activity_units')
        units = units.strip() if isinstance(units, str) else None
        _, val = parse_qualified_value(r.get('activity_value'))
        note = None
        if val is not None and units not in ('nM', 'kcal/mol', None):
            val, note = None, 'unsupported_units'
        qual = r.get('activity_qualifier')
        qual = qual.strip() if isinstance(qual, str) and qual.strip() else None
        rows.append({
            'compound_name': r.get('compound_name'),
            'smiles': r.get('smiles'),
            'target': r.get('target'),
            'activity_type': r.get('activity_type'),
            'activity_value': val,
            'activity_qualifier': qual,
            'parse_note': note,
        })
    out = _frame(rows, path)
    if 'source_file' in df.columns and len(df):
        # Keep each row's original provenance label; the master file is a merge of
        # many downloads, and collapsing them to "the master file" would hide
        # exactly the per-source detail the report exists to show.
        out['source_file'] = df['source_file'].values
    if 'n_measurements' in df.columns and len(df):
        counts = pd.to_numeric(df['n_measurements'], errors='coerce').fillna(1)
        out['n_measurements'] = counts.astype(int).values
    return out, rows_read


PARSERS = {
    'bindingdb_tsv': parse_bindingdb_tsv,
    'pubchem_bioassay_csv': parse_pubchem_bioassay_csv,
    'chembl_export_csv': parse_chembl_export_csv,
    'docking_sdf': parse_docking_sdf,
    'glide_csv': parse_glide_csv,
    'long_format_csv': parse_long_format_csv,
}
