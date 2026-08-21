"""
Builds the six training CSVs in data/ from data/HDAC_Docking_Inhibition.csv.

A normal run needs exactly one input file. HDAC_Docking_Inhibition.csv is the
long-format master: one row per measurement, with compound_name, smiles, target,
activity_type, activity_value, activity_units, source_file and activity_qualifier.
Every row carries its own provenance in source_file, which is what the report and
the redundancy check group on.

Pipeline shape:
    master file  ->  format parser (translate only)  ->  shared validation  ->
    redundancy check  ->  shared weighted-mean aggregation  ->  six CSVs + report

Nothing after the parser knows or cares which source a row came from.

When new data arrives, regenerate the master from the original vendor downloads:
    python data_prep.py --rebuild-master --raw-dir /path/to/raw
The raw files are not kept in the repo. Registering a new one is a RawSource entry
in sources.py, or a drop-in file matching a FORMAT_GLOBS pattern; either way no
aggregation code changes.

Usage:
    python data_prep.py                     # rebuild the six CSVs, write the report
    python data_prep.py --dry-run           # report only, write nothing
    python data_prep.py --rebuild-master --raw-dir DIR
    python data_prep.py --update-baseline   # accept current counts as the drift baseline
"""
import os
import sys
import json
import glob as globmod
import argparse
import warnings
from collections import Counter
from dataclasses import replace

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parsers
from sources import (DATA_DIR, MASTER_FILE, MASTER_SOURCE, RAW_DIR,
                     SUPPORTED_ACTIVITY_TYPES, TARGET_SOURCE_EXCLUSIONS,
                     VALID_QUALIFIERS, canonical_target, discover_raw_sources,
                     target_from_filename)

RDLogger.DisableLog('rdApp.*')

# Output spec. Filenames and column names are contractual: train_all_models.py
# reads exactly these.
OUTPUTS = [
    ('hdac8_ic50_clean_MERGED.csv', 'HDAC8', 'IC50', 'pIC50'),
    ('hdac1_ic50_clean.csv', 'HDAC1', 'IC50', 'pIC50'),
    ('hdac6_ic50_clean.csv', 'HDAC6', 'IC50', 'pIC50'),
    ('hdac8_dock_clean_MERGED.csv', 'HDAC8', 'docking_score', 'r_i_docking_score'),
    ('hdac1_dock_clean.csv', 'HDAC1', 'docking_score', 'r_i_docking_score'),
    ('hdac6_dock_clean.csv', 'HDAC6', 'docking_score', 'r_i_docking_score'),
]

MASTER_COLUMNS = ['compound_name', 'smiles', 'target', 'activity_type', 'activity_value',
                  'activity_units', 'source_file', 'activity_qualifier']

# Only exact measurements train a regression model. Everything else is reported as
# censored and dropped, in one place, for every source alike.
USABLE_QUALIFIER = '='

DEFAULT_OVERLAP_THRESHOLD = 0.95   # fraction of A's compounds also present in B
DEFAULT_VALUE_TOLERANCE = 0.10     # median |delta value| below this counts as "agrees"
MIN_OVERLAP_COMPOUNDS = 25         # below this, overlap is noise, not redundancy
DRIFT_TOLERANCE = 0.01             # 1% change from the stored baseline is flagged

BASELINE_FILE = os.path.join(DATA_DIR, 'data_prep_baseline.json')
REPORT_FILE = os.path.join(DATA_DIR, 'data_prep_report.md')

STANDARD_SCHEMA = ['compound_name', 'canonical_smiles', 'target', 'activity_type',
                   'activity_value', 'activity_qualifier', 'source_file', 'source_name',
                   'n_measurements']


# --------------------------------------------------------------------------
# SMILES handling (shared, source-agnostic)
# --------------------------------------------------------------------------
def strip_stereo_annotation(smi):
    """Some rows carry SMILES extension annotations like ' |w:8.7,2.1|' or ' |r|'
    (enhanced stereochemistry markup) that RDKit's MolFromSmiles chokes on unless
    using the V3000/extended parser. Strip anything from the first ' |' onward."""
    if not isinstance(smi, str):
        return None
    idx = smi.find(' |')
    return smi[:idx] if idx != -1 else smi


_canon_cache = {}


def canon(smi):
    """RDKit-canonical SMILES, or None if the structure will not parse."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    if smi in _canon_cache:
        return _canon_cache[smi]
    s = strip_stereo_annotation(smi)
    mol = Chem.MolFromSmiles(s) if s else None
    out = Chem.MolToSmiles(mol) if mol is not None else None
    _canon_cache[smi] = out
    return out


# --------------------------------------------------------------------------
# Per-input ingestion report
# --------------------------------------------------------------------------
class SourceReport:
    def __init__(self, source):
        self.source = source
        self.status = 'ok'
        self.status_detail = ''
        self.rows_read = 0
        self.rows_parsed = 0
        self.rows_kept = 0
        self.drops = Counter()
        self.bad_smiles = []          # (raw string, reason), capped at 5
        self.censored_rows = 0
        self.compounds_contributed = 0

    @property
    def name(self):
        return self.source.name

    def record_bad(self, raw, reason):
        if len(self.bad_smiles) < 5:
            self.bad_smiles.append((str(raw)[:120], reason))


# --------------------------------------------------------------------------
# Shared validation: identical for every source
# --------------------------------------------------------------------------
def validate(df, source, report, drop_unmapped=True):
    """Turn a parser's output into standard-schema rows, counting every drop.

    drop_unmapped=False keeps rows whose target label has no canonical mapping,
    under their original label. That is what --rebuild-master wants: the master file
    stays a superset of everything ingested, and the normal run is where unmapped
    rows get dropped and reported.
    """
    if df.empty:
        return pd.DataFrame(columns=STANDARD_SCHEMA)
    rows = []
    explicit_targets = source.targets if isinstance(source.targets, (tuple, list)) else None
    filename_target = (target_from_filename(source.resolved_path())
                       if source.targets in ('from_filename', 'column_or_filename') else None)

    for r in df.itertuples(index=False):
        note = getattr(r, 'parse_note', None)
        if isinstance(note, str) and note.strip():
            report.drops[note] += 1
            if 'smiles' in note or 'unreadable' in note or 'cid' in note:
                report.record_bad(r.smiles, note)
            continue

        act_type = r.activity_type
        if not isinstance(act_type, str) or act_type not in SUPPORTED_ACTIVITY_TYPES:
            report.drops['unsupported_activity_type'] += 1
            continue

        if not isinstance(r.smiles, str) or not r.smiles.strip():
            report.drops['missing_smiles'] += 1
            continue

        value = r.activity_value
        if value is None or (isinstance(value, float) and np.isnan(value)):
            report.drops['missing_value'] += 1
            continue
        value = float(value)
        if act_type != 'docking_score' and value <= 0:
            report.drops['nonpositive_value'] += 1
            continue

        qualifier = r.activity_qualifier
        if not isinstance(qualifier, str) or not qualifier.strip():
            if source.qualifier_blank_means is None:
                # No documented meaning for a blank qualifier in this source, so the
                # row is dropped and counted rather than assumed exact.
                report.drops['blank_qualifier_no_policy'] += 1
                continue
            qualifier = source.qualifier_blank_means
        qualifier = qualifier.strip()
        if qualifier not in VALID_QUALIFIERS:
            report.drops['unrecognized_qualifier'] += 1
            continue

        if explicit_targets is not None:
            targets = list(explicit_targets)
        elif source.targets == 'from_filename':
            targets = [filename_target]
        else:
            mapped = canonical_target(r.target)
            if mapped is None and filename_target is not None:
                # 'column_or_filename': the file was downloaded for one target, so a
                # row whose in-file target column is blank still belongs to it.
                if isinstance(r.target, str) and r.target.strip():
                    report.drops['target_unmapped'] += 1
                    continue
                mapped = filename_target
            elif (mapped is not None and filename_target is not None
                  and mapped != filename_target):
                # The file says one target and the row says another. Never guess.
                report.drops['target_conflict'] += 1
                continue
            if mapped is None:
                if drop_unmapped:
                    report.drops['target_unmapped'] += 1
                    continue
                mapped = r.target
            targets = [mapped]

        canonical = canon(r.smiles)
        if canonical is None:
            report.drops['rdkit_parse_failed'] += 1
            report.record_bad(r.smiles, 'rdkit_parse_failed')
            continue

        name = getattr(r, 'compound_name', None)
        for target in targets:
            rows.append((name, canonical, target, act_type, value, qualifier,
                         r.source_file, source.name,
                         int(getattr(r, 'n_measurements', 1) or 1)))

    out = pd.DataFrame(rows, columns=STANDARD_SCHEMA)
    report.rows_kept += len(out)
    return out


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def expand_paths(source):
    pattern = source.resolved_path()
    if any(ch in pattern for ch in '*?['):
        return sorted(globmod.glob(pattern))
    return [pattern]


def ingest(source_list, drop_unmapped=True):
    """Parse and validate every source in the list. Returns (frame, reports)."""
    frames, reports = [], []
    for source in source_list:
        report = SourceReport(source)
        reports.append(report)

        if not source.enabled:
            report.status = 'disabled'
            report.status_detail = 'enabled=False in the registry'
            continue

        paths = [p for p in expand_paths(source) if os.path.exists(p)]
        if not paths:
            report.status = 'MISSING FILE'
            report.status_detail = f'no file at {source.resolved_path()}'
            continue

        parser = parsers.PARSERS.get(source.fmt)
        if parser is None:
            report.status = 'UNKNOWN FORMAT'
            report.status_detail = f'no parser registered for format {source.fmt!r}'
            continue

        for path in paths:
            try:
                parsed, rows_read = parser(path, source)
            except Exception as exc:                      # noqa: BLE001
                report.status = 'PARSE ERROR'
                report.status_detail = f'{type(exc).__name__}: {exc}'
                continue
            report.rows_read += rows_read
            report.rows_parsed += len(parsed)
            clean = validate(parsed, source, report, drop_unmapped=drop_unmapped)
            if len(clean):
                frames.append(clean)

    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=STANDARD_SCHEMA))
    for report in reports:
        if len(frame):
            annotate_report(frame[frame['source_name'] == report.name], report)
    return frame, reports


def annotate_report(sub, report):
    if sub is None or sub.empty:
        return
    report.censored_rows = int((sub['activity_qualifier'] != USABLE_QUALIFIER).sum())
    usable = usable_rows(sub)
    report.compounds_contributed = int(usable['canonical_smiles'].nunique()) if len(usable) else 0


def usable_rows(frame):
    """Rows that actually reach an output: exact measurement, mapped target, used type."""
    if frame.empty:
        return frame
    wanted_types = {act for _, _, act, _ in OUTPUTS}
    wanted_targets = {tgt for _, tgt, _, _ in OUTPUTS}
    return frame[(frame['activity_qualifier'] == USABLE_QUALIFIER) &
                 frame['activity_type'].isin(wanted_types) &
                 frame['target'].isin(wanted_targets)]


def provenance_breakdown(frame, duplicate_counts=None):
    """Per source_file summary, so a provenance label contributing 0 is visible.

    The master file collapses many original downloads into one input, so this is
    where "a source silently contributed nothing" actually shows up.
    """
    columns = ['source_file', 'rows', 'censored', 'duplicates', 'targets',
               'activity_types', 'compounds']
    duplicate_counts = duplicate_counts or {}
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for label, sub in frame.groupby('source_file'):
        use = usable_rows(sub)
        rows.append({
            'source_file': label,
            'rows': len(sub),
            'censored': int((sub['activity_qualifier'] != USABLE_QUALIFIER).sum()),
            'duplicates': int(duplicate_counts.get(label, 0)),
            'targets': ', '.join(sorted(sub['target'].dropna().astype(str).unique())),
            'activity_types': ', '.join(sorted(sub['activity_type'].dropna().unique())),
            'compounds': int(use['canonical_smiles'].nunique()) if len(use) else 0,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        'rows', ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Cross-source duplicate measurements
# --------------------------------------------------------------------------
# PubChem aggregates ChEMBL and BindingDB, so one experiment reaches this pipeline
# through two or three files. Counting it once per file inflates n_measurements and
# lets a mirrored value outvote an independent one in the weighted mean. Two rows are
# the same measurement when compound, target, activity type, qualifier and value all
# match. Repeats INSIDE one source are left alone: those can be genuine replicates
# from different papers.
DUPLICATE_KEY = ['canonical_smiles', 'target', 'activity_type', 'activity_qualifier',
                 'rounded_value']


def _rounded(values):
    """Six significant figures, so 86.00000000000001 nM and 86 nM are one value."""
    return values.astype(float).map(lambda v: float(format(v, '.6g')))


def deduplicate_measurements(frame):
    """Collapse a measurement mirrored across sources. Returns (frame, per-source counts).

    Within one key group the source reporting that value the most times wins and keeps
    all of its rows; the other sources' copies are dropped. A source that lists a value
    twice while another lists it once really did contribute two measurements, and that
    survives.
    """
    if frame.empty:
        return frame, Counter()
    work = frame.copy()
    work['rounded_value'] = _rounded(work['activity_value'])
    per_source = work.groupby(DUPLICATE_KEY + ['source_file'], dropna=False).size()
    winners = per_source.reset_index(name='n').sort_values(
        ['n', 'source_file'], ascending=[False, True]).groupby(
        DUPLICATE_KEY, dropna=False).head(1)[DUPLICATE_KEY + ['source_file']]
    winners['_keep'] = True
    merged = work.merge(winners, on=DUPLICATE_KEY + ['source_file'], how='left')
    dropped = merged['_keep'].isna()
    counts = Counter(merged.loc[dropped, 'source_file'].astype(str))
    kept = merged.loc[~dropped, list(frame.columns)]
    return kept.reset_index(drop=True), counts


# --------------------------------------------------------------------------
# Redundant / superseded source detection
# --------------------------------------------------------------------------
def to_output_value(frame, activity_type):
    """pIC50 for affinities (values are nM), raw score for docking."""
    if activity_type == 'docking_score':
        return frame['activity_value'].astype(float)
    return 9 - np.log10(frame['activity_value'].astype(float))


def find_redundant_sources(frame, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD,
                           value_tolerance=DEFAULT_VALUE_TOLERANCE,
                           min_compounds=MIN_OVERLAP_COMPOUNDS, group_col='source_file'):
    """Flag any source whose compounds are almost entirely contained in another source
    for the same target, with closely agreeing values.

    Returns a list of finding dicts. Nothing is removed: a human decides.
    """
    findings = []
    if frame.empty:
        return findings
    usable = usable_rows(frame)
    if usable.empty:
        return findings
    for (target, act_type), block in usable.groupby(['target', 'activity_type']):
        per_source = {}
        for name, sub in block.groupby(group_col):
            vals = sub.assign(value=to_output_value(sub, act_type))
            per_source[name] = vals.groupby('canonical_smiles')['value'].mean()
        names = sorted(per_source)
        for a in names:
            for b in names:
                if a == b:
                    continue
                sa, sb = per_source[a], per_source[b]
                if len(sa) < min_compounds or len(sa) > len(sb):
                    continue
                shared = sa.index.intersection(sb.index)
                containment = len(shared) / len(sa)
                if containment < overlap_threshold:
                    continue
                median_diff = float(np.median(np.abs(sa[shared] - sb[shared])))
                findings.append({
                    'target': target,
                    'activity_type': act_type,
                    'subset_source': a,
                    'superset_source': b,
                    'subset_compounds': int(len(sa)),
                    'superset_compounds': int(len(sb)),
                    'overlap_compounds': int(len(shared)),
                    'containment': containment,
                    'median_abs_diff': median_diff,
                    'values_agree': bool(median_diff < value_tolerance),
                })
    findings.sort(key=lambda f: (-f['containment'], str(f['subset_source'])))
    return findings


# --------------------------------------------------------------------------
# Aggregation and output (shared, source-agnostic)
# --------------------------------------------------------------------------
def apply_target_source_exclusions(sub, target, activity_type, exclusion_log=None):
    """Drop rows from sources excluded for THIS target only (TARGET_SOURCE_EXCLUSIONS).

    Narrower than disabling a source in the registry: the rows stay in the master, stay
    in the provenance report, and stay available to every other target. Applied to IC50
    rows only -- the exclusions were validated on IC50 models, and docking must not
    inherit them by accident if a docking source is ever added to the dict.
    """
    excluded = TARGET_SOURCE_EXCLUSIONS.get(target, []) if activity_type == 'IC50' else []
    if not excluded:
        return sub
    mask = sub['source_file'].isin(excluded)
    if not mask.any():
        return sub
    kept = sub[~mask]
    if exclusion_log is not None:
        before = sub['canonical_smiles'].nunique()
        exclusion_log.append({
            'target': target,
            'activity_type': activity_type,
            'sources': sorted(set(sub.loc[mask, 'source_file'])),
            'rows_removed': int(mask.sum()),
            'compounds_removed': int(before - kept['canonical_smiles'].nunique()),
        })
    return kept.copy()


def aggregate(frame, target, activity_type, value_col, exclusion_log=None):
    """Weighted-mean aggregation across every source, weighted by n_measurements."""
    empty = pd.DataFrame(columns=['canonical_smiles', value_col, 'n_measurements'])
    if frame.empty:
        return empty
    sub = frame[(frame['target'] == target) &
                (frame['activity_type'] == activity_type) &
                (frame['activity_qualifier'] == USABLE_QUALIFIER)].copy()
    sub = apply_target_source_exclusions(sub, target, activity_type, exclusion_log)
    if sub.empty:
        return empty
    sub['value'] = to_output_value(sub, activity_type)
    sub['weighted'] = sub['value'] * sub['n_measurements']
    agg = sub.groupby('canonical_smiles').agg(
        weighted_sum=('weighted', 'sum'), n_measurements=('n_measurements', 'sum')
    ).reset_index()
    agg[value_col] = agg['weighted_sum'] / agg['n_measurements']
    return agg[['canonical_smiles', value_col, 'n_measurements']]


def build_outputs(frame, out_dir, write=True):
    results, exclusions = {}, []
    for filename, target, activity_type, value_col in OUTPUTS:
        agg = aggregate(frame, target, activity_type, value_col, exclusion_log=exclusions)
        results[filename] = agg
        if write:
            os.makedirs(out_dir, exist_ok=True)
            agg.to_csv(os.path.join(out_dir, filename), index=False)
    return results, exclusions


# --------------------------------------------------------------------------
# Master file regeneration (only when new raw data arrives)
# --------------------------------------------------------------------------
def rebuild_master(raw_dir, master_path=MASTER_FILE, write=True):
    """Re-derive the long-format master file from the original vendor downloads."""
    source_list = discover_raw_sources(raw_dir=raw_dir)
    print(f'--rebuild-master: {len(source_list)} raw sources '
          f'({sum(1 for s in source_list if s.discovered)} auto-discovered) under {raw_dir}')
    frame, reports = ingest(source_list, drop_unmapped=False)
    out = frame.rename(columns={'canonical_smiles': 'smiles'})
    out['activity_units'] = np.where(out['activity_type'] == 'docking_score',
                                     'kcal/mol', 'nM')
    out = out[MASTER_COLUMNS]
    if write:
        out.to_csv(master_path, index=False)
        print(f'Wrote {master_path}  ({len(out):,} rows)')
    return out, reports


# --------------------------------------------------------------------------
# Drift baseline
# --------------------------------------------------------------------------
def check_drift(results, baseline_path=BASELINE_FILE, update=False, tolerance=DRIFT_TOLERANCE):
    counts = {name: int(len(df)) for name, df in results.items()}
    baseline, notes = None, []
    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            baseline = json.load(fh).get('counts', {})
    if baseline:
        for name, n in counts.items():
            old = baseline.get(name)
            if old is None:
                notes.append((name, None, n, 'new output, no baseline'))
                continue
            drift = (n - old) / old if old else 0.0
            notes.append((name, old, n, f'{drift:+.2%}' if n != old else 'unchanged'))
    if update or not baseline:
        with open(baseline_path, 'w') as fh:
            json.dump({'counts': counts, 'tolerance': tolerance}, fh, indent=2)
    return counts, baseline, notes


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def format_report(reports, provenance, findings, results, drift_notes,
                  exclusions=(), tolerance=DRIFT_TOLERANCE):
    lines = ['# data_prep provenance report', '']

    lines += ['## Inputs', '',
              '| source | format | status | rows read | rows kept | censored | compounds |',
              '|---|---|---|---|---|---|---|']
    for r in reports:
        tag = r.name + (' *(auto-discovered)*' if r.source.discovered else '')
        lines.append(f'| {tag} | {r.source.fmt} | {r.status} | {r.rows_read:,} | '
                     f'{r.rows_kept:,} | {r.censored_rows:,} | {r.compounds_contributed:,} |')
    lines.append('')

    lines += ['## Rows dropped, by input and reason', '']
    for r in reports:
        if r.status != 'ok':
            lines.append(f'- **{r.name}**: {r.status} -- {r.status_detail}')
            continue
        if not r.drops:
            lines.append(f'- **{r.name}**: no rows dropped ({r.rows_parsed:,} parsed rows '
                         f'all valid)')
            continue
        detail = ', '.join(f'{reason} {n:,}'
                           for reason, n in sorted(r.drops.items(), key=lambda kv: str(kv[0])))
        lines.append(f'- **{r.name}**: {detail}')
        for raw, reason in r.bad_smiles:
            lines.append(f'    - `{raw}` ({reason})')
    lines.append('')

    lines += ['## Provenance (per source_file inside the master)', '',
              '| source_file | rows | censored | duplicate rows dropped | targets | '
              'types | compounds contributed |',
              '|---|---|---|---|---|---|---|']
    for row in provenance.itertuples(index=False):
        lines.append(f'| {row.source_file} | {row.rows:,} | {row.censored:,} | '
                     f'{row.duplicates:,} | {row.targets} | {row.activity_types} | '
                     f'{row.compounds:,} |')
    lines.append('')
    total_dupes = int(provenance['duplicates'].sum()) if len(provenance) else 0
    if total_dupes:
        lines += [f'{total_dupes:,} rows were the same measurement reported by more than '
                  f'one source (PubChem mirrors ChEMBL and BindingDB). Each is counted '
                  f'once, so n_measurements reflects distinct measurements rather than '
                  f'distinct files. Rows, censored and compounds above are counted before '
                  f'that collapse.', '']

    problems = [r for r in reports
                if r.status in ('MISSING FILE', 'PARSE ERROR', 'UNKNOWN FORMAT')]
    silent = [r for r in reports if r.status == 'ok' and r.compounds_contributed == 0
              and not r.source.expect_zero_contribution]
    empty_labels = [row.source_file for row in provenance.itertuples(index=False)
                    if row.compounds == 0]
    if problems or silent or empty_labels:
        lines += ['## Attention', '']
        for r in problems:
            lines.append(f'- **{r.name}: {r.status}** -- {r.status_detail}')
        for r in silent:
            lines.append(f'- **{r.name} contributed 0 compounds** despite reading '
                         f'{r.rows_read:,} rows. Check its registry entry before assuming '
                         f'this is intended.')
        for label in empty_labels:
            lines.append(f'- **source_file `{label}` contributed 0 compounds** to any '
                         f'output. Every one of its rows was censored, off-target, or '
                         f'dropped; see the drop reasons above.')
        lines.append('')

    lines += ['## Redundant / superseded sources', '']
    if not findings:
        lines.append('No source is a near-subset of another. Nothing to review.')
    else:
        for f in findings:
            flag = 'REDUNDANT' if f['values_agree'] else 'OVERLAPPING (values differ)'
            lines.append(
                f'- **{flag}: `{f["subset_source"]}` inside `{f["superset_source"]}` '
                f'({f["target"]} {f["activity_type"]})** -- {f["overlap_compounds"]:,} of '
                f'{f["subset_compounds"]:,} compounds ({f["containment"]:.1%}) also appear '
                f'in `{f["superset_source"]}` ({f["superset_compounds"]:,} compounds); '
                f'median |delta value| {f["median_abs_diff"]:.3f}.')
            if f['values_agree']:
                lines.append(f'    - The identical measurements are already counted once '
                             f'(see the duplicate column above), so n_measurements is not '
                             f'inflated by this. What is left is a judgment call: '
                             f'`{f["subset_source"]}` adds little beyond '
                             f'`{f["superset_source"]}`. Nothing was removed automatically.')
    lines.append('')

    lines += ['## Target-specific source exclusions', '']
    if not exclusions:
        lines.append('No target-specific source exclusion applied '
                     '(`TARGET_SOURCE_EXCLUSIONS` in sources.py is empty or '
                     'matched nothing in this master file).')
    else:
        lines.append('These rows are present in the master and are counted in the '
                     'provenance table above; they were held out of the listed '
                     'target-and-type training set only. Every other target still '
                     'sees them.')
        lines.append('')
        for ex in exclusions:
            src = ', '.join(f'`{name}`' for name in ex['sources'])
            lines.append(
                f"- **{ex['target']} {ex['activity_type']}: excluded {src} per "
                f"`TARGET_SOURCE_EXCLUSIONS`** -- {ex['rows_removed']:,} rows removed, "
                f"{ex['compounds_removed']:,} compounds dropped from the output "
                f"entirely (the rest keep a measurement from another source). "
                f"See the TARGET_SOURCE_EXCLUSIONS comment in sources.py for the "
                f"scaffold-CV evidence behind this entry.")
    lines.append('')

    lines += ['## Outputs', '',
              '| file | compounds | mean n_measurements | baseline | change |',
              '|---|---|---|---|---|']
    drift_by_name = {n[0]: n for n in drift_notes}
    for filename, _, _, _ in OUTPUTS:
        df = results[filename]
        mean_n = df['n_measurements'].mean() if len(df) else float('nan')
        note = drift_by_name.get(filename)
        old = f'{note[1]:,}' if note and note[1] is not None else '--'
        change = note[3] if note else 'baseline written'
        lines.append(f'| {filename} | {len(df):,} | {mean_n:.2f} | {old} | {change} |')
    lines.append('')

    drifted = [n for n in drift_notes if n[1] and abs(n[2] - n[1]) / n[1] > tolerance]
    if drifted:
        lines += [f'**Drift beyond {tolerance:.0%} from the stored baseline:**', '']
        for name, old, new, change in drifted:
            lines.append(f'- {name}: {old:,} -> {new:,} ({change})')
        lines.append('')
    return '\n'.join(lines)


def print_console_summary(reports, provenance, findings, results, exclusions=()):
    print('\n=== inputs ===')
    for r in reports:
        status = '' if r.status == 'ok' else f'  [{r.status}]'
        print(f'  {r.name:30s} read {r.rows_read:>7,}  kept {r.rows_kept:>7,}  '
              f'censored {r.censored_rows:>6,}  compounds {r.compounds_contributed:>7,}{status}')
        if r.status != 'ok':
            print(f'      {r.status_detail}')
        elif r.compounds_contributed == 0 and not r.source.expect_zero_contribution:
            print('      WARNING: contributed 0 compounds')
    if len(provenance):
        print('\n=== provenance (source_file inside the master) ===')
        for row in provenance.itertuples(index=False):
            flag = '   <-- contributes nothing' if row.compounds == 0 else ''
            print(f'  {row.source_file:36s} rows {row.rows:>7,}  censored {row.censored:>6,}  '
                  f'dup {row.duplicates:>6,}  compounds {row.compounds:>7,}{flag}')
    if findings:
        print('\n=== redundancy check ===')
        for f in findings:
            print(f'  {f["subset_source"]} is {f["containment"]:.1%} contained in '
                  f'{f["superset_source"]} ({f["target"]} {f["activity_type"]}), '
                  f'median |diff| {f["median_abs_diff"]:.3f}'
                  f'{"  <-- review" if f["values_agree"] else ""}')
    else:
        print('\n=== redundancy check === no near-subset sources found')
    if exclusions:
        print('\n=== target-specific source exclusions ===')
        for ex in exclusions:
            print(f"  {ex['target']} {ex['activity_type']}: excluded "
                  f"{', '.join(ex['sources'])}  ({ex['rows_removed']:,} rows, "
                  f"{ex['compounds_removed']:,} compounds)")
    if results:
        print('\n=== outputs ===')
        for filename, _, _, _ in OUTPUTS:
            print(f'  {filename:32s} {len(results[filename]):>7,} compounds')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--master', default=MASTER_FILE, help='the one input file a run needs')
    ap.add_argument('--out-dir', default=DATA_DIR)
    ap.add_argument('--report', default=REPORT_FILE)
    ap.add_argument('--baseline', default=BASELINE_FILE)
    ap.add_argument('--rebuild-master', action='store_true',
                    help='re-derive the master file from raw vendor downloads first')
    ap.add_argument('--raw-dir', default=RAW_DIR,
                    help='where the raw vendor downloads live (only for --rebuild-master)')
    ap.add_argument('--update-baseline', action='store_true',
                    help='overwrite the stored drift baseline with this run')
    ap.add_argument('--dry-run', action='store_true', help='write nothing')
    ap.add_argument('--keep-duplicates', action='store_true',
                    help='do not collapse a measurement reported by several sources')
    ap.add_argument('--overlap-threshold', type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    ap.add_argument('--value-tolerance', type=float, default=DEFAULT_VALUE_TOLERANCE)
    args = ap.parse_args(argv)

    if args.rebuild_master:
        _, raw_reports = rebuild_master(args.raw_dir, args.master, write=not args.dry_run)
        print_console_summary(raw_reports, pd.DataFrame(), [], None)

    source = replace(MASTER_SOURCE, path=args.master)
    frame, reports = ingest([source])
    # Redundancy and provenance are reported on everything that was read; the duplicate
    # collapse below is what the outputs are actually built from.
    findings = find_redundant_sources(frame, args.overlap_threshold, args.value_tolerance)
    deduped, duplicate_counts = ((frame, Counter()) if args.keep_duplicates
                                 else deduplicate_measurements(frame))
    provenance = provenance_breakdown(frame, duplicate_counts)
    results, exclusions = build_outputs(deduped, args.out_dir, write=not args.dry_run)
    counts, baseline, drift_notes = check_drift(
        results, baseline_path=args.baseline, update=args.update_baseline)

    print_console_summary(reports, provenance, findings, results, exclusions)

    report_text = format_report(reports, provenance, findings, results, drift_notes,
                                exclusions=exclusions)
    if not args.dry_run:
        with open(args.report, 'w', encoding='utf-8') as fh:
            fh.write(report_text + '\n')
        print(f'\nReport written to {args.report}')

    return frame, reports, findings, results


if __name__ == '__main__':
    main()
