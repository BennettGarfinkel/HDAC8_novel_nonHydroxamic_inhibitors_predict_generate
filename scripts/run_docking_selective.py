"""
Docking-selectivity run: reuses the already-trained model bundles from
models/run_state.pkl (HDAC8 IC50 + docking, HDAC1/HDAC6 IC50, HDAC1/HDAC6
docking) -- no retraining -- and runs the two-stage per-lineage island GA with
the docking-selectivity hard gate + Pareto objective in ga.py:

  - HDAC8 predicted docking score must be <= -7 (strong on-target binding),
    and not below the -13 applicability-domain floor.
  - HDAC1 and HDAC6 predicted docking scores must each be > -7 (weak
    off-target binding).

This is a clean one-sided directional threshold per isoform. Anything outside
those bounds is rejected outright (Tier-1 style) as a qualifying "hit", not
soft-penalized -- though it does not drop a candidate from the evolving
population (see passes_docking_selectivity_gate in ga.py). Among survivors,
docking_selectivity = min(dock_hdac1, dock_hdac6) - dock_hdac8 is an additional
Pareto objective (maximized), alongside the existing pIC50 (maximize) / HDAC8
docking (minimize) / liability (minimize) / potency-selectivity (maximize)
objectives.

Everything else (Tier-1 gates, ZBG-locked mutation/crossover, lineage protection,
AD gating, precedent checks, clustering, quota-balanced export) is shared with
the underlying ga.py/pipeline.py search machinery -- this script only wires
those pieces together per ZBG lineage and handles export.
"""
import os, pickle, random, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina

from pipeline import (descriptors_from_mol, compute_liability_flags, passes_tier1_gates,
                       morgan_fp_bitvect, max_training_similarity, AD_HARD_THRESHOLD,
                       compute_zbg_precedent_counts, ZBG_PRECEDENT_MIN, ic50_tier,
                       rough_2d_l_shape_proxy, ZBG_TAGS, SYNTHETIC_SEEDS)
from ga import (run_ga_pareto, ZBG_PERIPHERY_ELIGIBLE, HDAC8_DOCK_MAX, OFFTARGET_DOCK_MIN,
                 OFFTARGET_IC50_MIN_NM, MIN_SELECTIVITY_LOG)

CLUSTER_CUTOFF = 0.4
MAX_IC50_NM = 500
MAX_DOCKING_SCORE = -7.0        # legacy "hit" bound, kept for the printed hit-count only;
                                 # the real screen is now the docking-selectivity gate below
GENERATIONS_PER_STAGE = 55
POP_SIZE = 100
MAX_HITS_PER_ISLAND = 60        # lowered from 150 so 2-3 strong lineages can't dominate
                                 # the pool and crowd out weaker ones at export time
QUOTA_PER_CELL = 3
QUOTA_REPS = 4
ALL_LINEAGES = [name for name, _ in ZBG_TAGS]

# =========================================================================
# EXECUTION MODE SWITCHES -- all default False. Set the ones you want to
# run to True before executing; enabling all three runs them back to back in
# one process. Each mode is independent: its own seeded RNG state, its own
# output subfolder under candidates/ and figures/, and its own slot inside
# the single shared run-state pickle. Plain module constants, not a new
# config-file format -- consistent with this repo's existing convention
# (QUOTA_PER_CELL etc. above).
# =========================================================================
RUN_ALL = True             # default: run all three modes back to back, in the
                            # order listed in MODES. Set False to pick a subset
                            # with the three switches below.

# Only consulted when RUN_ALL is False.
RUN_BASELINE = False       # normal run, no ZBG-periphery changes (identical
                            # to pre-refactor behavior)
RUN_SEPARATE_ZBG = False   # periphery mutation ONLY, in isolation --
                            # crossover disabled. Only runs
                            # ZBG_PERIPHERY_ELIGIBLE lineages; others skipped.
RUN_INTEGRATED = False     # normal run PLUS periphery mutation blended into
                            # the same population via normal crossover

RANDOM_SEED = 42           # reset before EACH enabled mode's run. Does not
                            # cover PYTHONHASHSEED or sklearn's internal
                            # threading nondeterminism -- known gap, same
                            # one already noted for GLOBAL_SEED elsewhere.
PERIPHERY_MUTATION_PROB = 0.05   # used by RUN_INTEGRATED only

# =========================================================================
# SHARED RUN-STATE PICKLE
# One file for all three modes, keyed by mode name, updated in place after
# each run rather than one file per mode:
#
#   {'baseline':     {'hits_df', 'balanced', 'reps', 'final_screen', 'saved_at'},
#    'separate_zbg': {...},
#    'integrated':   {...}}
#
# Model bundles are deliberately absent. No mode retrains anything, so all six
# are identical across modes and already live in run_state.pkl, which is where
# make_figures.py reads them from. Storing them here per mode would add ~5.4 MB
# apiece for no new information.
#
# The notebook imports these two helpers so the read and write formats cannot
# drift apart between the script and the notebook.
# =========================================================================
RUN_STATE_PKL = '../models/docking_selective_run_state.pkl'
MODES = ['baseline', 'separate_zbg', 'integrated']
RESULT_KEYS = ('hits_df', 'balanced', 'reps', 'final_screen')


def _read_run_state(path=None):
    """Load the mode-keyed run state, migrating the pre-refactor layout if found.
    Returns {} when the file does not exist yet."""
    path = path or RUN_STATE_PKL   # resolved at call time, not at import time
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if 'hits_df' in state:
        # Pre-refactor layout: the four dataframes at the top level alongside
        # the six model bundles. That run predates the mode switches and used
        # what is now baseline behavior, so it lands in the baseline slot
        # instead of being dropped. The bundles are discarded on the next save.
        print(f"Migrating pre-refactor {os.path.basename(path)} into the 'baseline' slot.")
        legacy = {k: state[k] for k in RESULT_KEYS if k in state}
        legacy['saved_at'] = os.path.getmtime(path)
        return {'baseline': legacy}
    return state


def save_mode_run_state(mode, hits_df, balanced, reps, final_screen, path=None):
    """Update this mode's slot in the shared pickle, leaving the other modes as
    they are. Written to a temp file and moved into place, so a crash partway
    through cannot destroy results from modes that already finished."""
    path = path or RUN_STATE_PKL
    state = _read_run_state(path)
    state[mode] = {'hits_df': hits_df, 'balanced': balanced, 'reps': reps,
                   'final_screen': final_screen, 'saved_at': time.time()}
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmp, path)
    return state


def load_mode_run_state(mode=None, path=None):
    """Return (mode, entry) for one mode. mode=None picks whichever mode was
    saved most recently, so the notebook can follow the script without edits."""
    path = path or RUN_STATE_PKL
    state = _read_run_state(path)
    if not state:
        raise FileNotFoundError(
            f"{path} does not exist or holds no mode results yet. Run "
            f"run_docking_selective.py with one of RUN_BASELINE / RUN_SEPARATE_ZBG / "
            f"RUN_INTEGRATED set to True first.")
    if mode is None:
        mode = max(state, key=lambda m: state[m].get('saved_at', 0))
    elif mode not in state:
        raise KeyError(
            f"No results stored for mode {mode!r}. Stored modes: {sorted(state)}.")
    return mode, state[mode]


def full_tier1_ok(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    ok, _ = passes_tier1_gates(m, descriptors_from_mol(m), compute_liability_flags(m))
    return ok


def zbg_of(smi):
    return compute_liability_flags(Chem.MolFromSmiles(smi))['zbg_tag']


def butina_cluster(fps, cutoff=CLUSTER_CUTOFF):
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    return Butina.ClusterData(dists, n, cutoff, isDistData=True)


def quota_by_cell(df, quota, group_cols, sort_col='pIC50_pred'):
    parts = [grp.sort_values(sort_col, ascending=False).head(quota)
             for _, grp in df.groupby(group_cols)]
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts).sort_values(list(group_cols) + [sort_col],
                                         ascending=[True] * len(group_cols) + [False]).reset_index(drop=True)


def run_island(lineage, non_hydrox_df, ic50_bundle, dock_bundle, hdac1_bundle, hdac6_bundle,
               hdac1_dock_bundle, hdac6_dock_bundle,
               crossover_prob=0.5, periphery_mutation_prob=0.0):
    real = [(s, p) for s, p in zip(non_hydrox_df['canonical_smiles'], non_hydrox_df['pIC50'])
            if zbg_of(s) == lineage and full_tier1_ok(s)]
    real_df = pd.DataFrame(real, columns=['smi', 'pIC50']) if real else pd.DataFrame(columns=['smi', 'pIC50'])
    seeds = real_df.sort_values('pIC50', ascending=False)['smi'].head(10).tolist()
    if lineage in SYNTHETIC_SEEDS:
        seeds = seeds + [SYNTHETIC_SEEDS[lineage]]
    print(f"\n--- Island: {lineage} ({len(real_df)} real seeds available, {len(seeds)} used) ---")
    if not seeds:
        print(f"  No seeds available for {lineage}, skipping.")
        return {}

    t0 = time.time()
    _, evaluated, hits = run_ga_pareto(
        seeds, ic50_bundle, dock_bundle, max_ic50_nM=MAX_IC50_NM, max_docking_score=MAX_DOCKING_SCORE,
        generations=GENERATIONS_PER_STAGE, pop_size=POP_SIZE, protect_lineages=False,
        hdac1_bundle=hdac1_bundle, hdac6_bundle=hdac6_bundle,
        hdac1_dock_bundle=hdac1_dock_bundle, hdac6_dock_bundle=hdac6_dock_bundle,
        crossover_prob=crossover_prob, periphery_mutation_prob=periphery_mutation_prob)
    print(f"  stage 1: {len(hits)} hits ({time.time()-t0:.0f}s)")

    if evaluated:
        eval_df = pd.DataFrame(evaluated)
        stage2_seeds = eval_df.sort_values('pic50', ascending=False)['smi'].head(15).tolist()
        t0 = time.time()
        _, _, hits2 = run_ga_pareto(
            stage2_seeds, ic50_bundle, dock_bundle, max_ic50_nM=MAX_IC50_NM, max_docking_score=MAX_DOCKING_SCORE,
            generations=GENERATIONS_PER_STAGE, pop_size=POP_SIZE, protect_lineages=False,
            hdac1_bundle=hdac1_bundle, hdac6_bundle=hdac6_bundle,
            hdac1_dock_bundle=hdac1_dock_bundle, hdac6_dock_bundle=hdac6_dock_bundle,
            crossover_prob=crossover_prob, periphery_mutation_prob=periphery_mutation_prob)
        hits.update(hits2)
        print(f"  stage 2 (breeding): {len(hits)} combined hits ({time.time()-t0:.0f}s)")

    true_hits = {smi: res for smi, res in hits.items() if res['zbg_tag'] == lineage}
    print(f"  {len(true_hits)} genuinely still match {lineage} after drift filter")
    if len(true_hits) > MAX_HITS_PER_ISLAND:
        true_hits = dict(sorted(true_hits.items(), key=lambda kv: -kv[1]['pic50'])[:MAX_HITS_PER_ISLAND])
    return true_hits


def assemble_and_export(all_hits, ic50_df, non_hydrox, ic50_bundle, dock_bundle,
                         output_dir, mode, t_start):
    """Shared by all three execution modes: AD-gating, precedent-checking,
    clustering, quota-balanced export, and a lightweight run-state pickle.
    Writes every CSV deliverable under output_dir instead of the
    previously-hardcoded '../candidates' path.

    Only the HDAC8 IC50 and docking bundles are needed here, for the
    applicability-domain similarity columns. The HDAC1/HDAC6 bundles were
    parameters until the run-state pickle stopped storing model bundles;
    nothing in this function reads them now.

    Returns (hits_df, balanced, reps, final_screen), or a 4-tuple of Nones when
    nothing survived the gates and no export happened -- callers must check
    before running downstream steps that read the CSVs back (make_figures)."""
    print("\n=== Island summary ===")
    for lineage, hits in all_hits.items():
        print(f"  {lineage}: {len(hits)} hits")

    print("\n=== Assemble, AD-gate, precedent-check ===")
    rows = []
    for lineage, hits in all_hits.items():
        for smi, res in hits.items():
            ic50_nm = 10 ** (9 - res['pic50'])
            rows.append({
                'SMILES': smi, 'pIC50_pred': res['pic50'], 'IC50_nM_est': ic50_nm,
                'IC50_tier': ic50_tier(ic50_nm),
                'docking_pred': res['dock'], 'zbg_tag': res['zbg_tag'],
                'pains_brenk_flagged': res['pains_brenk_flagged'],
                'soft_metabolic_liability': res['soft_metabolic_liability'],
                'sa_score': res['sa_score'],
                'MW': res.get('mw'), 'cLogP': res.get('clogp'), 'HBD': res.get('hbd'),
                'HBA': res.get('hba'), 'TPSA': res.get('tpsa'), 'RotB': res.get('rotb'),
                'l_shape_2d_proxy': rough_2d_l_shape_proxy(Chem.MolFromSmiles(smi)),
                'pIC50_hdac1_pred': res.get('pic50_hdac1'), 'pIC50_hdac6_pred': res.get('pic50_hdac6'),
                'selectivity_vs_hdac1': (res['pic50'] - res['pic50_hdac1']) if res.get('pic50_hdac1') is not None else None,
                'selectivity_vs_hdac6': (res['pic50'] - res['pic50_hdac6']) if res.get('pic50_hdac6') is not None else None,
                'docking_hdac1_pred': res.get('dock_hdac1'), 'docking_hdac6_pred': res.get('dock_hdac6'),
                'docking_selectivity': res.get('docking_selectivity'),
                'passes_ic50_selectivity': res.get('passes_ic50_selectivity'),
                # Tracked per-candidate in ga.py (sticky across mutation/crossover)
                # but was being dropped here, since this row dict is an explicit
                # field list rather than a passthrough of res.
                'periphery_touched': res.get('periphery_touched', False),
            })
    # Checked BEFORE building the DataFrame: pd.DataFrame([]) has no columns, so
    # drop_duplicates(subset='SMILES') and sort_values('pIC50_pred') below both
    # raise KeyError on an empty pool rather than producing a zero-row frame.
    # A mode where every lineage returns zero hits is a real possibility (most
    # likely Separate ZBG, which runs 5 lineages with crossover disabled), and
    # crashing here would take any later enabled mode down with it.
    if not rows:
        print("No candidates survived the docking-selectivity gate. Exiting without export.")
        return None, None, None, None

    hits_df = pd.DataFrame(rows).drop_duplicates(subset='SMILES').sort_values('pIC50_pred', ascending=False).reset_index(drop=True)
    print(f"Full pool (all already docking-selectivity-gated): {len(hits_df)}")
    print(hits_df['zbg_tag'].value_counts())

    hits_df['ic50_model_max_similarity'] = hits_df['SMILES'].apply(
        lambda s: max_training_similarity(Chem.MolFromSmiles(s), ic50_bundle['train_fps'], nbits=ic50_bundle['nbits']))
    hits_df['dock_model_max_similarity'] = hits_df['SMILES'].apply(
        lambda s: max_training_similarity(Chem.MolFromSmiles(s), dock_bundle['train_fps'], nbits=dock_bundle['nbits']))
    hits_df['ic50_model_low_confidence'] = hits_df['ic50_model_max_similarity'] < 0.35
    hits_df['dock_model_low_confidence'] = hits_df['dock_model_max_similarity'] < 0.35
    before_ad = len(hits_df)
    hits_df = hits_df[hits_df['ic50_model_max_similarity'] >= AD_HARD_THRESHOLD].reset_index(drop=True)
    print(f"AD hard-reject: removed {before_ad - len(hits_df)}")

    # Second place the pool can empty out. Without this the run continues and
    # writes four header-only CSVs plus a pickle, which reads as a successful
    # run until you open the files. Bail the same way as the no-hits case.
    if len(hits_df) == 0:
        print("Every candidate was rejected by the applicability-domain filter. "
              "Exiting without export.")
        return None, None, None, None

    zbg_precedent_counts = compute_zbg_precedent_counts(ic50_df['canonical_smiles'].tolist())
    hits_df['zbg_precedent_count'] = hits_df['zbg_tag'].map(zbg_precedent_counts).fillna(0).astype(int)
    hits_df['zbg_low_precedent'] = hits_df['zbg_precedent_count'] < ZBG_PRECEDENT_MIN

    real_rows = [(zbg_of(s), p) for s, p in zip(non_hydrox['canonical_smiles'], non_hydrox['pIC50']) if full_tier1_ok(s)]
    ceilings = pd.DataFrame(real_rows, columns=['zbg', 'pIC50']).groupby('zbg')['pIC50'].max()
    hits_df['zbg_real_pic50_ceiling'] = hits_df['zbg_tag'].map(ceilings)
    hits_df['exceeds_real_precedent'] = hits_df['pIC50_pred'] > hits_df['zbg_real_pic50_ceiling']
    print(f"Exceeds real precedent: {hits_df['exceeds_real_precedent'].sum()} / {len(hits_df)}")

    print("\n=== Clustering + balanced export ===")
    hit_mols = [Chem.MolFromSmiles(s) for s in hits_df['SMILES']]
    hit_fps = [morgan_fp_bitvect(m) for m in hit_mols]
    if len(hit_fps) >= 2:
        clusters = butina_cluster(hit_fps)
        cluster_id = [None] * len(hits_df)
        for cid, idxs in enumerate(clusters):
            for idx in idxs:
                cluster_id[idx] = cid
        hits_df['cluster_id'] = cluster_id
    else:
        hits_df['cluster_id'] = 0
    hits_df['cluster_size'] = hits_df['cluster_id'].map(hits_df['cluster_id'].value_counts())
    print(f"{hits_df['cluster_id'].nunique()} structural clusters")

    os.makedirs(output_dir, exist_ok=True)
    hits_df.to_csv(f'{output_dir}/ga_candidates_RAW.csv', index=False)

    balanced = quota_by_cell(hits_df, QUOTA_PER_CELL, ['zbg_tag', 'IC50_tier'])
    balanced.to_csv(f'{output_dir}/ga_candidates.csv', index=False)
    print(f"\nPrimary deliverable (balanced by scaffold AND tier): {len(balanced)} rows")
    print(pd.crosstab(balanced['zbg_tag'], balanced['IC50_tier']))

    best_per_cluster = hits_df.loc[hits_df.groupby('cluster_id')['pIC50_pred'].idxmax()]
    reps = quota_by_cell(best_per_cluster, QUOTA_REPS, ['zbg_tag'])
    export_cols = ['SMILES', 'zbg_tag', 'cluster_id', 'cluster_size', 'pIC50_pred', 'IC50_nM_est',
                   'IC50_tier', 'docking_pred', 'sa_score', 'l_shape_2d_proxy',
                   'MW', 'cLogP', 'HBD', 'HBA', 'TPSA', 'RotB',
                   'pains_brenk_flagged', 'soft_metabolic_liability',
                   'zbg_precedent_count', 'zbg_low_precedent', 'exceeds_real_precedent',
                   'ic50_model_low_confidence', 'dock_model_low_confidence',
                   'pIC50_hdac1_pred', 'pIC50_hdac6_pred', 'selectivity_vs_hdac1', 'selectivity_vs_hdac6',
                   'docking_hdac1_pred', 'docking_hdac6_pred', 'docking_selectivity',
                   'periphery_touched']
    reps[export_cols].to_csv(f'{output_dir}/top_diverse_cluster_leads.csv', index=False)
    print(f"\nCluster reps: {len(reps)} rows")
    print(reps['zbg_tag'].value_counts())

    # Strict "everything satisfied" screen: every hit here already passed the
    # docking + IC50 selectivity gates (that's how it got into hits_df), so this
    # adds the remaining drug-likeness constraints explicitly: no PAINS/Brenk,
    # SA <= 3.5, Lipinski/Veber (via MW/cLogP/HBD/HBA/TPSA/RotB), and a potency
    # floor. Written so the column filter is transparent and auditable.
    fs = hits_df.copy()
    fs = fs[~fs['pains_brenk_flagged']]
    fs = fs[fs['sa_score'] <= 3.5]
    fs = fs[fs['MW'] <= 500]
    fs = fs[fs['cLogP'] <= 5]
    fs = fs[fs['HBD'] <= 5]
    fs = fs[fs['HBA'] <= 10]
    fs = fs[fs['TPSA'] <= 140]
    fs = fs[fs['RotB'] <= 10]
    fs = fs[fs['docking_pred'] <= -7.0]
    if 'docking_hdac1_pred' in fs.columns and fs['docking_hdac1_pred'].notna().any():
        fs = fs[(fs['docking_hdac1_pred'] > -7.0) & (fs['docking_hdac6_pred'] > -7.0)]
    fs = fs[fs['IC50_nM_est'] <= 500]
    final_screen = fs.reset_index(drop=True)
    # IC50 selectivity is reported on this clean list, not used to empty it. The
    # enumerated constraint set (no PAINS/Brenk, Lipinski/Veber, SA<=3.5, both
    # docking gates, potency) defines "clean"; passes_ic50_selectivity is kept
    # as a column so the fully-IC50-selective subset is one filter away. Making
    # it a hard filter here zeroed the list (only ~11/291 pass IC50 selectivity,
    # and none of those also clear SA<=3.5 + no-PAINS), which is not useful.
    final_screen.to_csv(f'{output_dir}/final_screened_candidates.csv', index=False)
    print(f"\nFinal strict all-constraints screen: {len(final_screen)} rows")

    # Updates this mode's slot in the shared pickle; other modes are left as
    # they are. See the SHARED RUN-STATE PICKLE block at the top of this file
    # for the layout and for why model bundles are not stored here.
    state = save_mode_run_state(mode, hits_df, balanced, reps, final_screen)
    print(f"Updated {os.path.basename(RUN_STATE_PKL)} slot '{mode}' "
          f"(dataframes only). Modes now stored: {sorted(state)}")
    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    return hits_df, balanced, reps, final_screen


def modes_to_run():
    """RUN_ALL wins outright; otherwise the individual switches select a subset.
    Returned in MODES order regardless of which switches were set."""
    if RUN_ALL:
        return list(MODES)
    enabled = {'baseline': RUN_BASELINE, 'separate_zbg': RUN_SEPARATE_ZBG,
               'integrated': RUN_INTEGRATED}
    return [m for m in MODES if enabled[m]]


def mode_plan(mode):
    """Which lineages a mode searches and the GA dial settings it searches with.
    Returns (lineages, crossover_prob, periphery_prob_for_lineage)."""
    if mode == 'separate_zbg':
        # Periphery mutation in isolation: crossover off, and only the ZBGs with
        # real measured HDAC8 data, since periphery mutation on a chemotype with
        # no measured precedent has nothing to anchor to.
        eligible = [lin for lin in ALL_LINEAGES if lin in ZBG_PERIPHERY_ELIGIBLE]
        skipped = [lin for lin in ALL_LINEAGES if lin not in ZBG_PERIPHERY_ELIGIBLE]
        print(f"Running periphery-only exploration for: {eligible}")
        print(f"Skipped (not ZBG-periphery-eligible): {skipped}")
        return eligible, 0.0, lambda lin: 1.0
    if mode == 'integrated':
        return ALL_LINEAGES, 0.5, (
            lambda lin: PERIPHERY_MUTATION_PROB if lin in ZBG_PERIPHERY_ELIGIBLE else 0.0)
    return ALL_LINEAGES, 0.5, lambda lin: 0.0


def main():
    print(f"Docking gate: HDAC8 dock <= {HDAC8_DOCK_MAX}, HDAC1/HDAC6 dock > {OFFTARGET_DOCK_MIN}")
    print(f"IC50 gate: off-target IC50 >= {OFFTARGET_IC50_MIN_NM} nM, selectivity margin >= {MIN_SELECTIVITY_LOG} log")

    selected = modes_to_run()
    if not selected:
        print("No mode enabled -- either set RUN_ALL = True, or set one or more of "
              "RUN_BASELINE / RUN_SEPARATE_ZBG / RUN_INTEGRATED to True at the top "
              "of this file.")
        return
    print(f"Modes to run ({'RUN_ALL' if RUN_ALL else 'individual switches'}): {selected}")

    with open('../models/run_state.pkl', 'rb') as f:
        st = pickle.load(f)
    ic50_bundle, dock_bundle = st['ic50_bundle'], st['dock_bundle']
    hdac1_bundle, hdac6_bundle = st['hdac1_bundle'], st['hdac6_bundle']
    hdac1_dock_bundle, hdac6_dock_bundle = st['hdac1_dock_bundle'], st['hdac6_dock_bundle']
    ic50_df = st['ic50_df']
    non_hydrox = ic50_df[~ic50_df['is_hydroxamate']]

    from make_figures import make_all_figures

    for mode in selected:
        print("\n" + "=" * 70 + f"\n=== MODE: {mode.replace('_', ' ').upper()} ===\n" + "=" * 70)
        # Reseeded per mode, so each one is reproducible on its own regardless of
        # which other modes ran first or in what order.
        random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
        t_start = time.time()

        lineages, crossover_prob, periphery_prob = mode_plan(mode)
        all_hits = {lin: run_island(lin, non_hydrox, ic50_bundle, dock_bundle,
                                     hdac1_bundle, hdac6_bundle, hdac1_dock_bundle, hdac6_dock_bundle,
                                     crossover_prob=crossover_prob,
                                     periphery_mutation_prob=periphery_prob(lin))
                    for lin in lineages}

        cand_dir, fig_dir = f'../candidates/{mode}', f'../figures/{mode}'
        exported, _, _, _ = assemble_and_export(
            all_hits, ic50_df, non_hydrox, ic50_bundle, dock_bundle,
            output_dir=cand_dir, mode=mode, t_start=t_start)
        if exported is None:
            # Nothing was written, so make_all_figures would fail reading the CSVs
            # back. Skip it and carry on to the next mode rather than aborting.
            print(f"{mode}: nothing exported, skipping figures.")
        else:
            make_all_figures(cand_dir=cand_dir, models_dir='../models', fig_dir=fig_dir)


if __name__ == '__main__':
    main()
