"""
hHDAC8 non-hydroxamate pipeline -- fixed version.
Changes vs. hHDAC8_predict_generate.ipynb documented in accompanying narrative.
"""
import os, sys, random, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, FilterCatalog, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# ---------------------------------------------------------------------------
# SMARTS definitions
# ---------------------------------------------------------------------------
HYDROXAMATE_PATTERN = Chem.MolFromSmarts('[CX3](=[OX1])[NX3][OX2H1]')

METABOLIC_LIABILITIES_HARD = [
    Chem.MolFromSmarts('[CX3](=[OX1])O[CX3](=[OX1])'),   # anhydride
    Chem.MolFromSmarts('N=C=O'),                          # isocyanate
    Chem.MolFromSmarts('N=C=S'),                          # isothiocyanate
    Chem.MolFromSmarts('[O,S]-C#N'),                      # cyanate/thiocyanate
    Chem.MolFromSmarts('C(=O)C(=O)N'),                    # alpha-keto amide
]
METABOLIC_LIABILITIES_SOFT = [
    Chem.MolFromSmarts('[CX3](=[OX1])[NX3][CX3](=[OX1])'),  # acyclic imide/diacylamide
    Chem.MolFromSmarts('[CX3](=[OX1])S'),                    # thioester
    Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX4]'),           # aliphatic ester
]
assert all(p is not None for p in METABOLIC_LIABILITIES_HARD + METABOLIC_LIABILITIES_SOFT)

# NEW: N-halogen (chloramine/bromamine/iodamine) liability. Found live during this
# session -- the fixed GA generated an N-Cl chloramine on an acylurea nitrogen as a
# Tier-1 survivor, because mutate() can attach a halogen substituent to any atom with
# an open valence, including ring/chain nitrogens, and nothing in the original
# liability set screened N-halogen bonds specifically. N-Cl/N-Br bonds are reactive,
# oxidizing, and not something you'd want in a drug candidate. Hard-rejected here.
N_HALOGEN_LIABILITY_PATTERN = Chem.MolFromSmarts('[NX3]-[Cl,Br,I]')
assert N_HALOGEN_LIABILITY_PATTERN is not None

pains_brenk_params = FilterCatalog.FilterCatalogParams()
pains_brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
pains_brenk_catalog = FilterCatalog.FilterCatalog(pains_brenk_params)

# SAFETY-AUDITED WHITELIST (revised this session). Removed: Hydrazide (N-N oxidative
# cleavage / hydrazine-release precedent, e.g. isoniazid), Thiol (free thiol oxidizes to
# disulfide, reactive toward off-target proteins via thiol-disulfide exchange),
# Benzisothiazolone (S-N ring-opens to a reactive electrophile; related chemistry is a
# known contact sensitizer), 8-Hydroxyquinoline (promiscuous, non-Zn-selective metal
# chelation; redox-active metal complexes; clioquinol-class neurotoxicity precedent).
# Added: Carboxylate and Trifluoromethyl ketone -- both explicitly named in the
# project's original ZBG scope but never implemented; both are metabolically inert or
# reversible with a much better safety track record than what was removed.
# 3-HPT is KEPT but flagged: pyrithione/thiopyridone-adjacent chemistry is a known
# nonspecific zinc ionophore at higher exposure -- retained because it's an explicit
# open-SAR area in the project, not because the concern doesn't apply.
ZBG_TAGS = [
    ('Salicylamide',            Chem.MolFromSmarts('[OX2H1]c1ccccc1[CX3](=O)[NX3]')),
    ('Ortho-aminoanilide',      Chem.MolFromSmarts('[NX3;H2]c1ccccc1[NX3][CX3]=O')),
    # Acylurea: UNCONFIRMED as a genuine HDAC zinc-binding mechanism -- no direct
    # literature validation found after a full sweep this session. It IS this
    # project's strongest real-Glide-docking chemotype (aryl-acylurea/DKP hits
    # scored -10.8 to -11.8 from the FragBreed library, dominates gated hits
    # across multiple runs), but an earlier session found the Glide metal-
    # coordination term (r_i_glide_metal) flat at -2.3 kcal/mol across top
    # compounds -- suggesting the strong scores may reflect cap/linker shape fit
    # rather than confirmed zinc chelation. TODO before treating any Acylurea
    # hit as validated: manually inspect zinc-ligand contact distances/geometry
    # on a top pose (not just total score) to check for real bidentate contact.
    # Kept in the whitelist pending that check -- same "kept but flagged" status
    # as 3-HPT below, not equivalent confidence to the other 6 entries.
    ('Acylurea',                Chem.MolFromSmarts('[#7][CX3](=O)[NX3][CX3](=O)')),
    # NOTE: corrected this session -- the original 'Sc1ncccc1O' pattern matches the
    # free-thiol tautomer of 3-hydroxypyridine-2-thione, which RDKit sanitizes to an
    # actual S-H (free thiol), triggering the free-thiol hard gate below. The real
    # ZBG chemotype is the thione tautomer (ring C=S, N-H), matched here instead.
    ('3-HPT',                   Chem.MolFromSmarts('[n;H0;D3]1c(=S)c([OX2H1])ccc1')),
    ('Carboxylate',             Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')),
    ('Trifluoromethyl-ketone',  Chem.MolFromSmarts('[CX3](=O)C(F)(F)F')),
    # Cyclic-thione: generalizes the 3-HPT pattern to ANY ring thioamide/thione
    # (ring N adjacent to a ring C=S). This is a PROTECTED thione -- sulfur locked
    # in a C=S tautomer, not a free S-H -- confirmed as a real HDAC8 ZBG class
    # (imidazole-thione, PMC6009916). Free thiols remain hard-excluded below.
    ('Cyclic-thione',           Chem.MolFromSmarts('[#7;R][#6;R](=[SX1])')),
    # alpha-amino-amide is the Whitehead (2011)/Greenwood (2020) bidentate ZBG
    # (free NH2 + adjacent amide C=O chelate Zn), confirmed by crystal structure
    # and MM-GBSA in Bandaru et al. Nat Sci Rep 2026. Ring-N-restricted here to
    # match the acyl-piperazine architecture actually used in the literature and
    # reduce false-positive matches to unrelated peptide-backbone fragments.
    # Originally paired with Triazoloquinoline/Triazolopyridine (removed -- those
    # SMARTS matched bare fused-ring atoms with no chelating heteroatom, confirmed
    # cap groups not ZBGs) and briefly with Azetidinone (removed this session --
    # no viable non-hydroxamate replacement found after testing 5 candidates).
    # 72 HDAC8 IC50 rows in HDAC_Docking_Inhibition.csv, median 840 nM.
    ('alpha-amino-amide',       Chem.MolFromSmarts('[NX3;H2][CX4;H1]([#6])[CX3](=O)[NX3;R]')),
]
assert all(p is not None for _, p in ZBG_TAGS)

# Single source of truth for lineages with too few real tier1-passing training seeds
# to run the GA off real data alone (run_island/run_ga_pareto cap real seeds at 10
# per lineage). Defined once here and imported by both run_docking_selective.py and
# the docking-selective notebook, instead of being hardcoded in both places.
# Real tier1-passing seed counts (see scripts/run_docking_selective.py run history):
# Salicylamide=2 -- thin. Lineages NOT here (e.g. alpha-amino-amide=27,
# Trifluoromethyl-ketone=12, Cyclic-thione=47, Carboxylate=68, Acylurea=8,
# Ortho-aminoanilide=217) already clear or approach the 10-seed cap on real data
# alone. 3-HPT's synthetic seed was removed -- it was added back when 3-HPT's
# SMARTS pattern was stricter and matched ZERO real training compounds; a later
# SMARTS correction (see ZBG_TAGS comment above) broadened the pattern and it now
# has 26 real tier1-passing seeds, well past the cap. Azetidinone's synthetic
# seed was removed along with the ZBG itself (no viable replacement found).
SYNTHETIC_SEEDS = {
    'Salicylamide': 'Oc1ccccc1C(=O)NCCN1CCN(Cc2c[nH]c3ccccc23)CC1',
}

# NEW: explicit substituted N-N hydrazide/hydrazine liability (Task 1, item 4). This is
# broader than the carbonyl-adjacent HYDRAZIDE_LIABILITY_PATTERN below -- it catches any
# N-N single bond where neither N is part of an amidine/imine (=N-), substituted or not.
# BRENK's own hydrazine alert only fires on a free terminal -NH-NH2, so it goes silent
# the instant the GA substitutes that nitrogen, even though the N-N bond (oxidative
# cleavage risk) is still there.
NN_LIABILITY_PATTERN = Chem.MolFromSmarts('[NX3;!$(N=*)]-[NX3;!$(N=*)]')
assert NN_LIABILITY_PATTERN is not None
HYDRAZIDE_LIABILITY_PATTERN = Chem.MolFromSmarts('[CX3](=O)[NX3][NX3]')
assert HYDRAZIDE_LIABILITY_PATTERN is not None

# Free thiol -- removed from the ZBG whitelist this session (oxidizes to disulfide,
# reactive toward off-target proteins). Checked independently of ZBG tagging, see
# passes_tier1_gates().
FREE_THIOL_LIABILITY_PATTERN = Chem.MolFromSmarts('[#6][SX2H1]')
assert FREE_THIOL_LIABILITY_PATTERN is not None

SA_SCORE_SOFT = 3.5   # penalized above this
SA_SCORE_HARD = 4.5   # hard rejected above this
AD_FLAG_THRESHOLD = 0.35   # below this: flagged low-confidence
AD_HARD_THRESHOLD = 0.20   # below this: hard-rejected (Task 3)

IC50_NBITS = 1024
DOCK_NBITS = 1024


def morgan_fp(mol, nbits=256, radius=2):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.array(gen.GetFingerprint(mol))

def morgan_fp_bitvect(mol, nbits=256, radius=2):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetFingerprint(mol)

def descriptors_from_mol(mol):
    return np.array([
        Descriptors.MolWt(mol), Crippen.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol)
    ])

def is_hydroxamate(mol):
    return mol.HasSubstructMatch(HYDROXAMATE_PATTERN)

def tag_zbg(mol):
    # NOTE: positional constraints on a ZBG (e.g. Acetamide, which only counts at a
    # chain end) are encoded in the SMARTS itself rather than checked here -- see the
    # Acetamide comment in ZBG_TAGS. Keeping this loop a plain first-match scan means
    # ga.py's atom-locking helpers (mutate, zbg_periphery_atoms, frag_both), which
    # call GetSubstructMatches on the raw patterns, stay consistent with tagging
    # instead of needing a parallel copy of the positional rule.
    for name, patt in ZBG_TAGS:
        if mol.HasSubstructMatch(patt):
            return name
    return 'Unknown'

def passes_lipinski_veber(desc):
    mw, logp, hbd, hba, tpsa, rotb = desc
    return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and rotb <= 10 and tpsa <= 140)


# ---------------------------------------------------------------------------
# Task 5 / Task 1: single source of truth for every liability + Tier-1 hard gate
# ---------------------------------------------------------------------------
def compute_liability_flags(mol):
    """Centralized liability + ZBG computation used everywhere: predict(), GA
    scoring, and final candidate export. Nothing here gates by itself -- gating
    logic lives in passes_tier1_gates() below, kept separate so predict() can
    still report on arbitrary compounds without silently dropping them."""
    return {
        'sa_score': sascorer.calculateScore(mol),
        'pains_brenk_flagged': pains_brenk_catalog.HasMatch(mol),
        'hard_metabolic_liability': (any(mol.HasSubstructMatch(p) for p in METABOLIC_LIABILITIES_HARD)
                                      or mol.HasSubstructMatch(N_HALOGEN_LIABILITY_PATTERN)),
        'n_halogen_liability': mol.HasSubstructMatch(N_HALOGEN_LIABILITY_PATTERN),
        'n_soft_metabolic_liabilities': sum(mol.HasSubstructMatch(p) for p in METABOLIC_LIABILITIES_SOFT),
        'hydrazide_liability': mol.HasSubstructMatch(HYDRAZIDE_LIABILITY_PATTERN),
        'nn_liability': mol.HasSubstructMatch(NN_LIABILITY_PATTERN),
        'free_thiol_liability': mol.HasSubstructMatch(FREE_THIOL_LIABILITY_PATTERN),
        'zbg_tag': tag_zbg(mol),
    }


def passes_tier1_gates(mol, desc, flags):
    """Tier-1 INVARIANT hard gates (Task 1). Any failure here means the molecule
    is dropped outright, before any scalar fitness math happens -- these are not
    soft-penalized. This replaces the earlier design where SA/ZBG/liability were
    partly folded into a stacked scalar penalty, which is exactly the mechanism
    that let the GA cheat its way into hydrazide-only convergence (see narrative).
    Returns (passed: bool, reason: str or None)."""
    if is_hydroxamate(mol):
        return False, 'hydroxamate'
    if not passes_lipinski_veber(desc):
        return False, 'lipinski_veber'
    if flags['sa_score'] > SA_SCORE_HARD:
        return False, 'sa_score'
    if flags['hard_metabolic_liability']:
        return False, 'hard_metabolic_liability'
    if flags['zbg_tag'] == 'Unknown':
        return False, 'no_zbg'
    # Belt-and-suspenders: hydrazide/N-N and free thiol are removed from the ZBG
    # whitelist, but tag_zbg() only reports the FIRST matching pattern -- a molecule
    # could in principle carry BOTH a whitelisted ZBG (e.g. Acylurea) AND a hydrazide
    # or free thiol elsewhere in the structure and still pass the ZBG gate above.
    # These checks catch that regardless of what ZBG tag the molecule got.
    if flags['hydrazide_liability'] or flags['nn_liability']:
        return False, 'hydrazide_or_nn'
    if flags['free_thiol_liability']:
        return False, 'free_thiol'
    return True, None


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_model_from_df(df, value_col, smi_col='canonical_smiles', nbits=None,
                         informative_min_count=15, label='model'):
    """Generic scaffold-grouped-CV regressor trainer usable for ANY target/value
    column (HDAC8/1/6 IC50 or docking). Returns a bundle dict: {model,
    informative_mask, train_fps, nbits, cv_r2}. Uses HistGradientBoostingRegressor
    (the model selected across the original per-target sweeps) and Murcko-scaffold
    GroupKFold so the reported R2 is an honest generalization estimate, not a
    memorization score.

    value_col: e.g. 'pIC50' for IC50 sets, 'r_i_docking_score' for docking sets.
    nbits: fingerprint size; defaults to IC50_NBITS for pIC50, DOCK_NBITS otherwise.
    """
    if nbits is None:
        nbits = IC50_NBITS if value_col == 'pIC50' else DOCK_NBITS

    df = df.dropna(subset=[smi_col, value_col]).copy()
    mols = [Chem.MolFromSmiles(s) for s in df[smi_col]]
    keep = [i for i, m in enumerate(mols) if m is not None]
    df = df.iloc[keep].reset_index(drop=True)
    mols = [mols[i] for i in keep]

    def get_scaffold(smi):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi))
        except Exception:
            return None
    df['scaffold'] = df[smi_col].apply(get_scaffold)

    desc = np.array([descriptors_from_mol(m) for m in mols])
    fps = np.array([morgan_fp(m, nbits=nbits) for m in mols])
    informative_mask = fps.sum(axis=0) > informative_min_count
    X = np.hstack([desc, fps[:, informative_mask]])
    y = df[value_col].values

    cv_r2 = None
    valid = df['scaffold'].notna()
    if valid.sum() > 10:
        gkf = GroupKFold(n_splits=5)
        scores = cross_val_score(
            HistGradientBoostingRegressor(max_depth=None, learning_rate=0.1, random_state=42),
            X[valid.values], y[valid.values], cv=gkf,
            groups=df.loc[valid, 'scaffold'].values, scoring='r2', n_jobs=-1)
        cv_r2 = float(scores.mean())
        print(f"  [{label}] scaffold-CV R2 = {scores.mean():.3f} (+/- {scores.std():.3f}), n = {len(df)}")

    model = HistGradientBoostingRegressor(max_depth=None, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    train_fps = [morgan_fp_bitvect(m, nbits=nbits) for m in mols]
    return {'model': model, 'informative_mask': informative_mask, 'train_fps': train_fps,
            'nbits': nbits, 'cv_r2': cv_r2}


def compute_zbg_precedent_counts(ic50_bundle_train_smiles):
    """Counts how many TRAINING compounds actually carry each ZBG tag. Used to flag
    'model has zero/near-zero real precedent for this exact ZBG' independent of
    whole-molecule AD similarity, which can look confident purely because the cap/
    linker is familiar even when the chelating fragment itself was never seen (this
    is exactly what happened with the 3-HPT and Salicylamide synthetic seeds --
    checked directly this session: both score 0.4-0.55 whole-molecule similarity,
    'confident' by the AD_FLAG_THRESHOLD=0.35 cutoff, despite one ZBG having ZERO
    real examples anywhere in the training set and the other having exactly 1)."""
    counts = {}
    for smi in ic50_bundle_train_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        tag = tag_zbg(m)
        counts[tag] = counts.get(tag, 0) + 1
    return counts


ZBG_PRECEDENT_MIN = 5  # below this many real training examples, treat as "not
                        # meaningfully precedented" regardless of whole-molecule AD score

def ic50_tier(ic50_nm):
    """Tiers within the 500 nM triage bound."""
    if ic50_nm < 150:
        return 'Tier 1 (<150 nM)'
    elif ic50_nm < 300:
        return 'Tier 2 (150-300 nM)'
    else:
        return 'Tier 3 (300-500 nM)'


def rough_2d_l_shape_proxy(mol):
    """APPROXIMATE 2D stand-in ONLY -- not a substitute for the real L-shape check.
    The actual L-shape hypothesis (Task 2) is a 3D vector-angle test between the
    ZBG-linker axis and the cap sub-cavity projection, 80-130 degrees, and requires a
    real 3D conformer plus knowledge of the binding-site geometry (Maestro/Glide,
    unavailable in this sandbox). This function instead computes, from a 2D depiction,
    the ratio of the molecule's principal-axis span to its perpendicular span using
    RDKit's 2D coordinate generation -- a crude 'is this molecule elongated/bent
    rather than round or linear' signal. A ratio far from 1.0 suggests SOME
    directional extent, consistent with (but nowhere near proof of) an L-shaped
    cap-linker-ZBG architecture. Treat this purely as a cheap pre-filter to deprior-
    itize obviously compact/spherical candidates, not as an L-shape confirmation."""
    from rdkit.Chem import rdDepictor
    try:
        mol2d = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol2d)
        conf = mol2d.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                            for i in range(mol2d.GetNumAtoms())])
        coords -= coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[1] < 1e-6:
            return None
        return float(np.sqrt(eigvals[0] / eigvals[1]))
    except Exception:
        return None


def max_training_similarity(mol, train_fps, nbits=256):
    fp = morgan_fp_bitvect(mol, nbits=nbits)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return max(sims) if sims else 0.0


def predict(smiles, ic50_bundle, dock_bundle, hdac1_bundle=None, hdac6_bundle=None,
            hdac1_dock_bundle=None, hdac6_dock_bundle=None):
    """Single-SMILES lookup. hdac1_bundle/hdac6_bundle are optional real off-target
    IC50 model bundles -- when provided, adds HDAC1/HDAC6 predicted pIC50 and
    pIC50(HDAC8)-pIC50(isoform) selectivity to the output. hdac1_dock_bundle/
    hdac6_dock_bundle are the analogous optional off-target DOCKING model bundles
    (trained the same way as dock_bundle, just on HDAC1/HDAC6 docking data) --
    when provided, adds predicted HDAC1/HDAC6 docking scores. Every optional field
    is explicitly None when its bundle isn't supplied (never fabricated)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'SMILES': smiles, 'valid': False}
    desc = descriptors_from_mol(mol)
    fp_ic50 = morgan_fp(mol, nbits=ic50_bundle.get('nbits', IC50_NBITS))
    fp_dock = morgan_fp(mol, nbits=dock_bundle.get('nbits', DOCK_NBITS))
    X_ic50 = np.hstack([desc, fp_ic50[ic50_bundle['informative_mask']]]).reshape(1, -1)
    pic50 = ic50_bundle['model'].predict(X_ic50)[0]
    ic50_sim = max_training_similarity(mol, ic50_bundle['train_fps'], nbits=ic50_bundle.get('nbits', IC50_NBITS))
    X_dock = np.hstack([desc, fp_dock[dock_bundle['informative_mask']]]).reshape(1, -1)
    dock = dock_bundle['model'].predict(X_dock)[0]
    dock_sim = max_training_similarity(mol, dock_bundle['train_fps'], nbits=dock_bundle.get('nbits', DOCK_NBITS))
    flags = compute_liability_flags(mol)
    passed, reason = passes_tier1_gates(mol, desc, flags)

    pic50_hdac1 = pic50_hdac6 = selectivity_vs_hdac1 = selectivity_vs_hdac6 = None
    if hdac1_bundle is not None:
        fp1 = morgan_fp(mol, nbits=hdac1_bundle.get('nbits', IC50_NBITS))
        X1 = np.hstack([desc, fp1[hdac1_bundle['informative_mask']]]).reshape(1, -1)
        pic50_hdac1 = hdac1_bundle['model'].predict(X1)[0]
        selectivity_vs_hdac1 = pic50 - pic50_hdac1
    if hdac6_bundle is not None:
        fp6 = morgan_fp(mol, nbits=hdac6_bundle.get('nbits', IC50_NBITS))
        X6 = np.hstack([desc, fp6[hdac6_bundle['informative_mask']]]).reshape(1, -1)
        pic50_hdac6 = hdac6_bundle['model'].predict(X6)[0]
        selectivity_vs_hdac6 = pic50 - pic50_hdac6

    dock_hdac1 = dock_hdac6 = None
    if hdac1_dock_bundle is not None:
        fp_d1 = morgan_fp(mol, nbits=hdac1_dock_bundle.get('nbits', DOCK_NBITS))
        X_d1 = np.hstack([desc, fp_d1[hdac1_dock_bundle['informative_mask']]]).reshape(1, -1)
        dock_hdac1 = hdac1_dock_bundle['model'].predict(X_d1)[0]
    if hdac6_dock_bundle is not None:
        fp_d6 = morgan_fp(mol, nbits=hdac6_dock_bundle.get('nbits', DOCK_NBITS))
        X_d6 = np.hstack([desc, fp_d6[hdac6_dock_bundle['informative_mask']]]).reshape(1, -1)
        dock_hdac6 = hdac6_dock_bundle['model'].predict(X_d6)[0]

    mw, clogp, hbd, hba, tpsa, rotb = desc
    return {
        'SMILES': smiles, 'valid': True, 'pIC50_pred': pic50, 'IC50_nM_est': 10 ** (9 - pic50),
        'docking_pred': dock, 'is_hydroxamate': is_hydroxamate(mol),
        'MW': mw, 'cLogP': clogp, 'HBD': int(hbd), 'HBA': int(hba), 'TPSA': tpsa, 'RotB': int(rotb),
        'passes_lipinski_veber': passes_lipinski_veber(desc), 'sa_score': flags['sa_score'],
        'pains_brenk_flagged': flags['pains_brenk_flagged'],
        'hard_metabolic_liability': flags['hard_metabolic_liability'],
        'soft_metabolic_liability': flags['n_soft_metabolic_liabilities'] > 0,
        'hydrazide_liability': flags['hydrazide_liability'], 'nn_liability': flags['nn_liability'],
        'zbg_tag': flags['zbg_tag'], 'passes_tier1_gates': passed, 'tier1_fail_reason': reason,
        'ic50_model_max_similarity': ic50_sim, 'ic50_model_low_confidence': ic50_sim < AD_FLAG_THRESHOLD,
        'ic50_model_ad_reject': ic50_sim < AD_HARD_THRESHOLD,
        'dock_model_max_similarity': dock_sim, 'dock_model_low_confidence': dock_sim < AD_FLAG_THRESHOLD,
        'dock_model_ad_reject': dock_sim < AD_HARD_THRESHOLD,
        'pIC50_hdac1_pred': pic50_hdac1, 'IC50_hdac1_nM_est': (10 ** (9 - pic50_hdac1)) if pic50_hdac1 is not None else None,
        'pIC50_hdac6_pred': pic50_hdac6, 'IC50_hdac6_nM_est': (10 ** (9 - pic50_hdac6)) if pic50_hdac6 is not None else None,
        'selectivity_vs_hdac1': selectivity_vs_hdac1, 'selectivity_vs_hdac6': selectivity_vs_hdac6,
        'docking_hdac1_pred': dock_hdac1, 'docking_hdac6_pred': dock_hdac6,
        'docking_selectivity': (min(dock_hdac1, dock_hdac6) - dock) if (dock_hdac1 is not None and dock_hdac6 is not None) else None,
    }


def predict_batch(smiles_list, ic50_bundle, dock_bundle, hdac1_bundle=None, hdac6_bundle=None,
                   hdac1_dock_bundle=None, hdac6_dock_bundle=None):
    """Convenience wrapper -- run predict() over a list of SMILES, including HDAC1/HDAC6
    off-target IC50 and docking predictions when those bundles are supplied."""
    return [predict(s, ic50_bundle, dock_bundle, hdac1_bundle=hdac1_bundle, hdac6_bundle=hdac6_bundle,
                     hdac1_dock_bundle=hdac1_dock_bundle, hdac6_dock_bundle=hdac6_dock_bundle)
            for s in smiles_list]
