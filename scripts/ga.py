"""
Tier-2 Pareto (NSGA-II style) non-dominated sorting + crowding distance. Tier-1
hard gates (pipeline.py) run first and unconditionally reject; among survivors,
selection is by Pareto rank on up to five objectives: (1) predicted pIC50
(maximize), (2) docking score (minimize / more negative better), (3) a
liability score combining SA score, PAINS/BRENK, soft metabolic flags,
hydrazide/N-N liability (minimize), and, when off-target model bundles are
supplied, (4) IC50 selectivity and (5) docking selectivity (both maximize).
Ties within a Pareto rank are broken by crowding distance computed directly
from spacing along each objective (see crowding_distance()), not by molecular
similarity.
"""
import random
import numpy as np
from rdkit import Chem
from pipeline import (compute_liability_flags, passes_tier1_gates, descriptors_from_mol,
                       morgan_fp, is_hydroxamate, passes_lipinski_veber,
                       IC50_NBITS, DOCK_NBITS, ZBG_TAGS)

SUBSTITUENTS_EXTENDED = [
    '[H]', 'C', 'CC', 'CCC', 'CC(C)C', 'C1CC1',
    'OC', 'OCC', 'O',
    'F', 'Cl', 'Br', 'C(F)(F)F',
    'C(=O)N', 'C(=O)C', 'C#N',
    'c1ccccc1', 'c1ccncc1', 'c1ccoc1', 'c1cscn1',
    'N1CCOCC1', 'N1CCNCC1',
    'S(=O)(=O)C', 'C(C)(C)C',
]

ZBG_PERIPHERY_ELIGIBLE = [
    'alpha-amino-amide', 'Cyclic-thione', '3-HPT', 'Ortho-aminoanilide', 'Carboxylate',
]
# Restricted to ZBGs with real measured HDAC8 data. Verified counts (non-hydroxamate
# rows in hdac8_ic50_clean_MERGED.csv, tag_zbg() applied): Ortho-aminoanilide 384,
# Carboxylate 136, Cyclic-thione 50, alpha-amino-amide 29, 3-HPT 27. Excludes
# Acylurea (core chelation mechanism itself unconfirmed -- see pipeline.py
# flag comment), Salicylamide and Trifluoromethyl-ketone (periphery here is
# mostly ring substitution already covered by normal mutate()).


def mutate(smi):
    """Mutates a random substituent-bearing atom. ZBG atoms are locked (excluded from
    candidate positions) so mutation can't silently destroy the chelating group and
    turn e.g. a Carboxylate seed into something with no ZBG at all -- this was a real
    gap: the original candidate-atom list only checked for a free H, and a carboxylic
    acid O-H satisfies that, meaning the ZBG itself was a legal mutation site."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    zbg_atom_idxs = set()
    for _, patt in ZBG_TAGS:
        for match in mol.GetSubstructMatches(patt):
            zbg_atom_idxs.update(match)
    mol = Chem.RWMol(mol)
    candidates = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetTotalNumHs() > 0 and a.GetIdx() not in zbg_atom_idxs]
    if not candidates:
        return None
    idx = random.choice(candidates)
    sub_smi = random.choice(SUBSTITUENTS_EXTENDED)
    if sub_smi == '[H]':
        return Chem.MolToSmiles(mol)
    sub_mol = Chem.MolFromSmiles(sub_smi)
    if sub_mol is None:
        return None
    combo = Chem.RWMol(Chem.CombineMols(mol, sub_mol))
    offset = mol.GetNumAtoms()
    try:
        combo.AddBond(idx, offset, Chem.BondType.SINGLE)
        new_mol = combo.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception:
        return None


def zbg_periphery_atoms(mol):
    """Returns atom indices bonded to a ZBG-matched atom but NOT themselves
    part of any ZBG_TAGS match -- the immediate anchor positions where cap/
    linker/substituent chemistry attaches to the warhead. Deliberately
    atoms-adjacent-to-the-match rather than a fixed bond-radius, so it stays
    anchored to whatever the ZBG SMARTS actually captures. NOTE: for a
    multi-atom ZBG SMARTS (e.g. alpha-amino-amide, which matches N-C-C-C(=O)-N
    including a generic '[#6]' branch atom), that branch atom is itself part
    of the match and therefore excluded from periphery -- periphery starts
    one bond further out than the branch atom, not at it. Don't assume the
    periphery lands on the atom immediately named in a SMARTS branch; verify
    per-pattern before relying on it."""
    zbg_atom_idxs = set()
    for _, patt in ZBG_TAGS:
        for match in mol.GetSubstructMatches(patt):
            zbg_atom_idxs.update(match)
    periphery = set()
    for idx in zbg_atom_idxs:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() not in zbg_atom_idxs:
                periphery.add(nbr.GetIdx())
    return periphery


def mutate_zbg_periphery(smi):
    """Like mutate(), but restricted to zbg_periphery_atoms(smi) instead of
    every free-H atom in the molecule. Returns None if no eligible atom
    exists -- caller falls back to mutate() in that case."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    periphery = zbg_periphery_atoms(mol)
    mol = Chem.RWMol(mol)
    candidates = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetTotalNumHs() > 0 and a.GetIdx() in periphery]
    if not candidates:
        return None
    idx = random.choice(candidates)
    sub_smi = random.choice(SUBSTITUENTS_EXTENDED)
    if sub_smi == '[H]':
        return Chem.MolToSmiles(mol)
    sub_mol = Chem.MolFromSmiles(sub_smi)
    if sub_mol is None:
        return None
    combo = Chem.RWMol(Chem.CombineMols(mol, sub_mol))
    offset = mol.GetNumAtoms()
    try:
        combo.AddBond(idx, offset, Chem.BondType.SINGLE)
        new_mol = combo.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception:
        return None


def frag_both(m):
    zbg_atom_idxs = set()
    for _, patt in ZBG_TAGS:
        for match in m.GetSubstructMatches(patt):
            zbg_atom_idxs.update(match)
    bonds = [b.GetIdx() for b in m.GetBonds()
             if not b.IsInRing()
             and b.GetBeginAtomIdx() not in zbg_atom_idxs
             and b.GetEndAtomIdx() not in zbg_atom_idxs]
    if not bonds:
        return None, None
    bidx = random.choice(bonds)
    frag_mol = Chem.FragmentOnBonds(m, [bidx], addDummies=True)
    pieces = sorted(Chem.GetMolFrags(frag_mol, asMols=True), key=lambda p: p.GetNumAtoms())
    if len(pieces) < 2:
        return None, None
    return pieces[0], pieces[-1]


def crossover(smi1, smi2):
    """Fixed crossover (verified bug, see memory): large 'core' fragment from one
    parent + small 'arm' fragment from the other, never large+large."""
    try:
        m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return None
        small1, large1 = frag_both(m1)
        small2, large2 = frag_both(m2)
        if small1 is None or small2 is None:
            return None
        core, arm = random.choice([(large1, small2), (large2, small1)])
        combo = Chem.CombineMols(core, arm)
        rw = Chem.RWMol(combo)
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == '*']
        if len(dummies) < 2:
            return None
        neighbors = [rw.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx() for d in dummies[:2]]
        rw.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
        for d in sorted(dummies[:2], reverse=True):
            rw.RemoveAtom(d)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception:
        return None


def liability_score(flags):
    """Single scalar liability axis for Pareto objective #3. Lower is better.
    This is NOT stacked with pIC50/docking into one fitness number -- it's its
    own Pareto dimension, so a molecule can't buy potency by paying down its
    liability score against a different currency."""
    # NOTE: hydrazide/N-N and free thiol are no longer scored here -- they're Tier-1
    # hard gates now (passes_tier1_gates), so anything reaching this function has
    # already cleared them. Scoring them again here would be double-counting.
    score = 0.0
    score += max(0.0, flags['sa_score'] - 3.5)          # soft SA penalty above 3.5
    score += 1.0 if flags['pains_brenk_flagged'] else 0.0
    score += 1.0 * flags['n_soft_metabolic_liabilities']
    return score


# ---------------------------------------------------------------------------
# Docking-score selectivity gate (revised): HDAC8 must dock strongly (<= -7),
# HDAC1 and HDAC6 must dock weakly (> -7). This is a clean directional
# threshold on each isoform, NOT the old +/-1.5 two-sided band -- the band was
# overcomplicating things and could reject a genuinely strong HDAC8 pose just
# for being "too good". The only two-sided element kept is a sanity floor
# (DOCK_AD_FLOOR): a predicted score below it is treated as an applicability-
# domain artifact (the regression extrapolating past anything it ever saw in
# training) rather than a real binding pose, and excluded.
#
# Off-target IC50 selectivity is enforced separately (see
# passes_ic50_selectivity_gate): HDAC8 should be more POTENT than HDAC1/HDAC6.
# Docking measures pose/shape complementarity; IC50 measures actual inhibition.
# Requiring both closes the loophole where a molecule docks selectively but is
# predicted to inhibit the off-targets just as hard.
HDAC8_DOCK_MAX = -7.0            # HDAC8 predicted docking must be <= this (strong)
OFFTARGET_DOCK_MIN = -7.0        # HDAC1/HDAC6 predicted docking must be > this (weak)
DOCK_AD_FLOOR = -13.0            # scores below this are treated as AD artifacts

# Off-target IC50 selectivity thresholds (new). Rationale in the header comment
# above. Numbers picked from the training distributions: HDAC1 median IC50
# ~260 nM, HDAC6 median ~85 nM, HDAC8 median ~920 nM -- so "off-target IC50 must
# be weaker than 1000 nM" plus "HDAC8 at least ~5x (0.7 log) more potent than
# each off-target" is a meaningful but not impossibly strict selectivity window.
OFFTARGET_IC50_MIN_NM = 1000.0        # HDAC1/HDAC6 predicted IC50 must be >= this (weak)
MIN_SELECTIVITY_LOG = 0.7             # HDAC8 pIC50 must beat each off-target by >= this


def passes_docking_selectivity_gate(dock_hdac8, dock_hdac1, dock_hdac6):
    """Hard gate (reject outright, no partial credit) -- mirrors the Tier-1
    gate style in pipeline.py. Requires:
      - HDAC8 predicted docking <= -7 (strong on-target binding), and not
        below the AD floor of -13 (implausible extrapolation),
      - HDAC1 and HDAC6 predicted docking each > -7 (weak off-target binding).
    If either off-target score is None (bundle not supplied), the off-target
    half of the gate is skipped so this stays backward compatible with callers
    that don't have off-target docking models."""
    if not (DOCK_AD_FLOOR <= dock_hdac8 <= HDAC8_DOCK_MAX):
        return False
    if dock_hdac1 is None or dock_hdac6 is None:
        return True
    if not (dock_hdac1 > OFFTARGET_DOCK_MIN):
        return False
    if not (dock_hdac6 > OFFTARGET_DOCK_MIN):
        return False
    return True


def passes_ic50_selectivity_gate(pic50_hdac8, pic50_hdac1, pic50_hdac6):
    """Hard gate on predicted IC50 selectivity: HDAC8 must be meaningfully more
    potent than both off-targets AND both off-targets must be genuinely weak.
    Requires each off-target predicted IC50 >= OFFTARGET_IC50_MIN_NM and each
    (pIC50_HDAC8 - pIC50_offtarget) >= MIN_SELECTIVITY_LOG. If either off-target
    pIC50 is None, the gate is skipped (backward compatible)."""
    if pic50_hdac1 is None or pic50_hdac6 is None:
        return True
    ic50_hdac1_nm = 10 ** (9 - pic50_hdac1)
    ic50_hdac6_nm = 10 ** (9 - pic50_hdac6)
    if ic50_hdac1_nm < OFFTARGET_IC50_MIN_NM or ic50_hdac6_nm < OFFTARGET_IC50_MIN_NM:
        return False
    if (pic50_hdac8 - pic50_hdac1) < MIN_SELECTIVITY_LOG:
        return False
    if (pic50_hdac8 - pic50_hdac6) < MIN_SELECTIVITY_LOG:
        return False
    return True


def evaluate_candidate(mol, ic50_bundle, dock_bundle, hdac1_bundle=None, hdac6_bundle=None,
                        hdac1_dock_bundle=None, hdac6_dock_bundle=None):
    """Returns None if it fails a Tier-1 hard gate (hydroxamate, Lipinski/Veber,
    SA score, hard metabolic liability, unrecognized ZBG, hydrazide/N-N, free
    thiol -- see passes_tier1_gates). Otherwise a dict with the Pareto
    objectives plus reporting fields. If hdac1_bundle/hdac6_bundle are
    provided (real off-target IC50 models), a 'selectivity' objective =
    min(pIC50_hdac8 - pIC50_hdac1, pIC50_hdac8 - pIC50_hdac6) is computed and
    maximized -- this directly prioritizes HDAC8 selectivity during the
    search itself, rather than only reporting it after the fact. If
    hdac1_dock_bundle/hdac6_dock_bundle are ALSO provided (real off-target
    DOCKING models), a second, independent objective 'docking_selectivity' =
    min(dock_hdac1, dock_hdac6) - dock_hdac8 is computed and maximized
    (bigger gap between weak off-target docking and strong HDAC8 docking =
    better), and a 'passes_dock_gate' boolean flags whether all three docking
    scores fall inside their tolerance bands. passes_dock_gate is NOT a hard
    rejection here (see the comment at its computation for why) -- it's
    consumed downstream by run_ga_pareto to decide what counts as a
    qualifying hit, while the docking_selectivity Pareto objective is what
    actually drives the population toward the band across generations."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    desc = descriptors_from_mol(mol)
    flags = compute_liability_flags(mol)
    passed, reason = passes_tier1_gates(mol, desc, flags)
    if not passed:
        return None

    fp_ic50 = morgan_fp(mol, nbits=ic50_bundle.get('nbits', IC50_NBITS))
    X_ic50 = np.hstack([desc, fp_ic50[ic50_bundle['informative_mask']]]).reshape(1, -1)
    pic50_pred = ic50_bundle['model'].predict(X_ic50)[0]

    fp_dock = morgan_fp(mol, nbits=dock_bundle.get('nbits', DOCK_NBITS))
    X_dock = np.hstack([desc, fp_dock[dock_bundle['informative_mask']]]).reshape(1, -1)
    dock_pred = dock_bundle['model'].predict(X_dock)[0]

    liab = liability_score(flags)

    selectivity = None
    pic50_hdac1 = pic50_hdac6 = None
    if hdac1_bundle is not None and hdac6_bundle is not None:
        fp1 = morgan_fp(mol, nbits=hdac1_bundle.get('nbits', IC50_NBITS))
        X1 = np.hstack([desc, fp1[hdac1_bundle['informative_mask']]]).reshape(1, -1)
        pic50_hdac1 = hdac1_bundle['model'].predict(X1)[0]
        fp6 = morgan_fp(mol, nbits=hdac6_bundle.get('nbits', IC50_NBITS))
        X6 = np.hstack([desc, fp6[hdac6_bundle['informative_mask']]]).reshape(1, -1)
        pic50_hdac6 = hdac6_bundle['model'].predict(X6)[0]
        selectivity = min(pic50_pred - pic50_hdac1, pic50_pred - pic50_hdac6)

    dock_hdac1 = dock_hdac6 = None
    passes_dock_gate = True
    if hdac1_dock_bundle is not None and hdac6_dock_bundle is not None:
        fp_d1 = morgan_fp(mol, nbits=hdac1_dock_bundle.get('nbits', DOCK_NBITS))
        X_d1 = np.hstack([desc, fp_d1[hdac1_dock_bundle['informative_mask']]]).reshape(1, -1)
        dock_hdac1 = hdac1_dock_bundle['model'].predict(X_d1)[0]
        fp_d6 = morgan_fp(mol, nbits=hdac6_dock_bundle.get('nbits', DOCK_NBITS))
        X_d6 = np.hstack([desc, fp_d6[hdac6_dock_bundle['informative_mask']]]).reshape(1, -1)
        dock_hdac6 = hdac6_dock_bundle['model'].predict(X_d6)[0]
        # NOTE: this used to be a hard `return None` here, mirroring the Tier-1
        # gate style. That was a real bug once exposed by a strict band: if
        # gen-1's seed population has NOTHING inside the band yet (common --
        # real seeds are picked for potency, not docking selectivity), then
        # `evaluated` comes back empty, run_ga_pareto's "if not evaluated:
        # continue" fires, and the population is NEVER regenerated -- it's
        # frozen at the original seeds forever, generation after generation,
        # producing a permanent 0-hit lineage regardless of budget. Confirmed
        # directly: 6 of 9 lineages went to 0 hits this way. Fixed by scoring
        # this as a flag instead of a rejection -- the population keeps
        # evolving via Tier-1 survival + the docking_selectivity Pareto
        # objective (which rewards exactly the direction that reaches the
        # band: weaker off-target, stronger on-target), and the gate is
        # enforced only downstream, when deciding what counts as a qualifying
        # "hit" (see run_ga_pareto).
        passes_dock_gate = passes_docking_selectivity_gate(dock_pred, dock_hdac1, dock_hdac6)
    # If off-target docking bundles aren't supplied, passes_dock_gate stays True
    # (skipped, not enforced) -- legacy call sites are unaffected either way.

    # IC50 selectivity is REPORTED and PREFERRED but not a hard hit gate. Making
    # it a hard gate on top of the docking gate was tried and proved far too
    # strict: HDAC8's own training potency is modest (median IC50 ~920 nM), so
    # requiring HDAC8 to be >=0.7 log more potent than off-targets that are
    # themselves forced weaker than 1000 nM collapses the feasible region to
    # ~1 hit per lineage. Instead, the docking-selectivity gate alone defines a
    # qualifying hit (that's the item-1 ask), the existing 'selectivity' Pareto
    # objective (min pIC50 margin vs off-targets, maximized) pushes the search
    # toward IC50-selective molecules, and passes_ic50_gate is recorded as a
    # flag so the strict final screen and any downstream filtering can require
    # it. This keeps hit yield healthy while still tracking real IC50 selectivity.
    passes_ic50_gate = passes_ic50_selectivity_gate(pic50_pred, pic50_hdac1, pic50_hdac6)

    docking_selectivity = None
    if dock_hdac1 is not None and dock_hdac6 is not None:
        docking_selectivity = min(dock_hdac1, dock_hdac6) - dock_pred

    # Physicochemical descriptors (desc = [MolWt, MolLogP, HBD, HBA, TPSA, RotB]),
    # surfaced so they land in the output CSVs for downstream filtering/reporting.
    mw, clogp, hbd, hba, tpsa, rotb = desc
    return {
        'smi': Chem.MolToSmiles(mol), 'pic50': pic50_pred, 'dock': dock_pred, 'liab': liab,
        'selectivity': selectivity, 'pic50_hdac1': pic50_hdac1, 'pic50_hdac6': pic50_hdac6,
        'dock_hdac1': dock_hdac1, 'dock_hdac6': dock_hdac6, 'docking_selectivity': docking_selectivity,
        'passes_dock_gate': passes_dock_gate, 'passes_ic50_selectivity': passes_ic50_gate,
        'mw': mw, 'clogp': clogp, 'hbd': int(hbd), 'hba': int(hba), 'tpsa': tpsa, 'rotb': int(rotb),
        'zbg_tag': flags['zbg_tag'], 'pains_brenk_flagged': flags['pains_brenk_flagged'],
        'soft_metabolic_liability': flags['n_soft_metabolic_liabilities'] > 0,
        'hydrazide_liability': flags['hydrazide_liability'], 'sa_score': flags['sa_score'],
    }


def dominates(a, b):
    """a dominates b if a is at least as good on all objectives and strictly better
    on at least one. pic50: higher better; dock: lower better; liab: lower better;
    selectivity (if present on both): higher better; docking_selectivity (if present
    on both): higher better."""
    use_selectivity = a.get('selectivity') is not None and b.get('selectivity') is not None
    use_dock_selectivity = a.get('docking_selectivity') is not None and b.get('docking_selectivity') is not None
    better_or_eq = (a['pic50'] >= b['pic50']) and (a['dock'] <= b['dock']) and (a['liab'] <= b['liab'])
    strictly_better = (a['pic50'] > b['pic50']) or (a['dock'] < b['dock']) or (a['liab'] < b['liab'])
    if use_selectivity:
        better_or_eq = better_or_eq and (a['selectivity'] >= b['selectivity'])
        strictly_better = strictly_better or (a['selectivity'] > b['selectivity'])
    if use_dock_selectivity:
        better_or_eq = better_or_eq and (a['docking_selectivity'] >= b['docking_selectivity'])
        strictly_better = strictly_better or (a['docking_selectivity'] > b['docking_selectivity'])
    return better_or_eq and strictly_better


def fast_non_dominated_sort(pop):
    """Standard NSGA-II fast non-dominated sort. Returns list of fronts (each a
    list of indices into pop), front 0 = best (Pareto-optimal within this pop)."""
    n = len(pop)
    S = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(pop, front):
    """Crowding distance within one front, across objectives (pic50, dock, liab,
    and selectivity when real off-target models are available)."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float('inf')
        return dist
    objectives = [('pic50', False), ('dock', True), ('liab', True)]
    if all(pop[i].get('selectivity') is not None for i in front):
        objectives.append(('selectivity', False))
    if all(pop[i].get('docking_selectivity') is not None for i in front):
        objectives.append(('docking_selectivity', False))
    for key, better_low in objectives:
        vals = sorted(front, key=lambda i: pop[i][key])
        vmin, vmax = pop[vals[0]][key], pop[vals[-1]][key]
        dist[vals[0]] = dist[vals[-1]] = float('inf')
        if vmax == vmin:
            continue
        for k in range(1, len(vals) - 1):
            dist[vals[k]] += (pop[vals[k + 1]][key] - pop[vals[k - 1]][key]) / (vmax - vmin)
    return dist


def run_ga_pareto(seed_smiles, ic50_bundle, dock_bundle, max_ic50_nM, max_docking_score,
                   generations=60, pop_size=100, elite_frac=0.3, protect_lineages=True,
                   hdac1_bundle=None, hdac6_bundle=None,
                   hdac1_dock_bundle=None, hdac6_dock_bundle=None,
                   crossover_prob=0.5, periphery_mutation_prob=0.0):
    """NSGA-II style GA. Returns (history, all_scored_final_gen, hits_meeting_both_bounds).

    hdac1_bundle/hdac6_bundle: real off-target IC50 model bundles. When provided,
    'selectivity' becomes a Pareto objective (see evaluate_candidate), so the search
    itself prioritizes HDAC8-selective candidates rather than only reporting
    selectivity after the fact.

    hdac1_dock_bundle/hdac6_dock_bundle: real off-target DOCKING model bundles (new
    this session). When BOTH are provided, every evaluated candidate gets a
    'docking_selectivity' Pareto objective (maximized: bigger gap between weak
    off-target docking and strong HDAC8 docking = better) that drives the search
    toward directional docking selectivity across generations: HDAC8 predicted
    docking <= -7 (strong, and not below the -13 applicability-domain floor)
    while HDAC1 and HDAC6 predicted docking are each > -7 (weak) -- see
    passes_docking_selectivity_gate, a one-sided threshold on each isoform, not
    a symmetric +/- band around a target value. A candidate only counts as a
    returned "hit" once it both meets the pIC50/docking bounds AND actually
    passes this directional docking-selectivity gate -- but unlike a Tier-1
    gate, failing it does NOT drop a candidate from the evolving population,
    so a lineage whose real seeds start outside the gate can still search its
    way in rather than being frozen at generation 1.

    protect_lineages: if True, tracks which ZBG chemotype each candidate's *originating
    seed* belongs to, and reserves a minimum quota of elite slots per lineage that has
    living members -- so a single fast-converging seed (e.g. Carboxylate) can't crowd
    out the others before they've had a chance to develop competitive descendants. This
    was added after a run where 5 single-ZBG synthetic seeds were used to fix a
    diversity gap, and one lineage wiped out the other 4 by generation ~15 purely from
    elitism dynamics, not because its chemistry was actually better.

    crossover_prob: probability a new individual comes from crossover rather
    than mutation (default 0.5, matching the prior hardcoded value --
    passing the default reproduces the exact prior code path).

    periphery_mutation_prob: probability a MUTATION event (not crossover)
    uses mutate_zbg_periphery() instead of mutate(). Default 0.0 (off).
    Falls back to mutate() if the candidate's ZBG isn't in
    ZBG_PERIPHERY_ELIGIBLE or periphery mutation returns None. Tracked via
    periphery_touched (sticky across mutation/crossover, same propagation
    pattern as `lineage`), surfaced in the result dict."""
    pic50_target = 9 - np.log10(max_ic50_nM)

    clean_seeds = []
    for s in seed_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None and not is_hydroxamate(m) and passes_lipinski_veber(descriptors_from_mol(m)):
            clean_seeds.append(s)
    if not clean_seeds:
        raise ValueError("No provided seeds pass Tier-1 gates.")

    # lineage[smi] = ZBG tag of the seed this candidate (or its ancestor) traces back
    # to. Propagated through mutation (1 parent) and crossover (inherits from the
    # 'core' parent, since that fragment carries the ZBG in this design).
    lineage = {}
    for s in clean_seeds:
        m = Chem.MolFromSmiles(s)
        flags = compute_liability_flags(m)
        lineage[s] = flags['zbg_tag']

    # Sticky flag: True once this candidate (or an ancestor) was produced by a
    # ZBG-periphery mutation. Propagated the same way as `lineage`.
    periphery_touched = {s: False for s in clean_seeds}

    population_smi = list(clean_seeds)
    while len(population_smi) < pop_size:
        population_smi.append(random.choice(clean_seeds))

    history = []
    hits = {}
    n_elite = max(4, int(pop_size * elite_frac))
    evaluated = []
    unique_lineages = sorted(set(lineage.values()))
    n_lineages = len(unique_lineages)

    for gen in range(generations):
        evaluated = []
        for smi in population_smi:
            res = evaluate_candidate(Chem.MolFromSmiles(smi), ic50_bundle, dock_bundle,
                                      hdac1_bundle=hdac1_bundle, hdac6_bundle=hdac6_bundle,
                                      hdac1_dock_bundle=hdac1_dock_bundle, hdac6_dock_bundle=hdac6_dock_bundle)
            if res is not None:
                # FIX: previously used the inherited historical label (lineage.get(smi)),
                # which can silently diverge from the molecule's ACTUAL current ZBG once
                # mutation/crossover changes which pattern it matches (found this
                # directly: an Ortho-aminoanilide island's tracked label said
                # 'Ortho-aminoanilide' every single generation, but 100% of its final
                # hits were actually Carboxylate on fresh evaluation -- the free
                # aniline NH2 got mutated away while a carboxylic acid formed
                # elsewhere, and nothing ever re-checked the label against reality).
                # Floor protection now keys off res['zbg_tag'] (recomputed fresh this
                # generation), so it protects actual chemotypes, not stale names.
                res['lineage'] = res['zbg_tag']
                res['periphery_touched'] = periphery_touched.get(smi, False)
                evaluated.append(res)
                meets_bounds = (res['pic50'] >= pic50_target) and (res['dock'] <= max_docking_score) \
                               and res.get('passes_dock_gate', True)
                if meets_bounds:
                    hits[res['smi']] = res

        if not evaluated:
            print(f"Gen {gen+1}: no candidates survived Tier-1 gates")
            history.append(history[-1] if history else 0.0)
            continue

        fronts = fast_non_dominated_sort(evaluated)

        best_pic50 = max(e['pic50'] for e in evaluated)
        history.append(best_pic50)

        if protect_lineages and n_lineages > 1:
            # Reserve floor(n_elite / n_lineages) slots per lineage (that still has
            # living members), filled by that lineage's own best-ranked individuals
            # (lowest front index, then highest crowding distance). Remaining slots
            # filled globally by front-rank + crowding distance as before.
            per_lineage_floor = max(1, n_elite // (2 * n_lineages))
            # global rank/crowding-distance lookup, computed once
            rank_of = {}
            cd_of = {}
            for fi, front in enumerate(fronts):
                cd = crowding_distance(evaluated, front)
                for i in front:
                    rank_of[i] = fi
                    cd_of[i] = cd[i]

            elite_idx = []
            by_lineage = {ln: [] for ln in unique_lineages}
            for i, e in enumerate(evaluated):
                by_lineage.setdefault(e['lineage'], []).append(i)
            for ln, idxs in by_lineage.items():
                idxs_sorted = sorted(idxs, key=lambda i: (rank_of[i], -cd_of[i]))
                elite_idx.extend(idxs_sorted[:per_lineage_floor])

            remaining_slots = n_elite - len(set(elite_idx))
            if remaining_slots > 0:
                already = set(elite_idx)
                pool = [i for i in range(len(evaluated)) if i not in already]
                pool_sorted = sorted(pool, key=lambda i: (rank_of[i], -cd_of[i]))
                elite_idx = list(dict.fromkeys(elite_idx + pool_sorted[:remaining_slots]))
        else:
            elite_idx = []
            for front in fronts:
                if len(elite_idx) + len(front) <= n_elite:
                    elite_idx.extend(front)
                else:
                    cd = crowding_distance(evaluated, front)
                    remaining = n_elite - len(elite_idx)
                    front_sorted = sorted(front, key=lambda i: -cd[i])
                    elite_idx.extend(front_sorted[:remaining])
                    break

        elites = [evaluated[i]['smi'] for i in elite_idx]
        elite_lineages = [evaluated[i]['lineage'] for i in elite_idx]

        new_pop = list(dict.fromkeys(elites))
        attempts = 0
        while len(new_pop) < pop_size and attempts < pop_size * 8:
            attempts += 1
            if random.random() < crossover_prob and len(elites) >= 2:
                p1_idx, p2_idx = random.sample(range(len(elites)), 2)
                p1, p2 = elites[p1_idx], elites[p2_idx]
                child = crossover(p1, p2)
                child_lineage = elite_lineages[p1_idx]  # inherits from 'core'-eligible parent 1 slot
                touched = (periphery_touched.get(p1, False) or periphery_touched.get(p2, False)) if child else False
            else:
                p_idx = random.randrange(len(elites))
                parent_smi = elites[p_idx]
                # Order matters: `periphery_mutation_prob > 0` comes FIRST so the
                # random.random() draw is short-circuited away entirely when the
                # feature is off. That is what makes the default settings consume
                # an identical RNG stream to the pre-refactor code. Do not reorder.
                use_periphery = (periphery_mutation_prob > 0
                                 and random.random() < periphery_mutation_prob
                                 and elite_lineages[p_idx] in ZBG_PERIPHERY_ELIGIBLE)
                child = mutate_zbg_periphery(parent_smi) if use_periphery else None
                touched_this_step = child is not None
                if child is None:
                    child = mutate(parent_smi)
                child_lineage = elite_lineages[p_idx]
                touched = (touched_this_step or periphery_touched.get(parent_smi, False)) if child else False
            if child:
                new_pop.append(child)
                lineage[child] = child_lineage
                periphery_touched[child] = touched
        population_smi = new_pop[:pop_size]

        n_front0 = len(fronts[0])
        n_hits_gen = sum(1 for e in evaluated
                          if e['pic50'] >= pic50_target and e['dock'] <= max_docking_score)
        lineage_counts = pd_value_counts(elite_lineages)
        print(f"Gen {gen+1:2d}: survived Tier-1={len(evaluated):3d}  front-0={n_front0:3d}  "
              f"best pIC50={best_pic50:.3f}  meeting-bounds-this-gen={n_hits_gen}  "
              f"elite-lineages={lineage_counts}")

    return history, evaluated, hits


def pd_value_counts(items):
    counts = {}
    for i in items:
        counts[i] = counts.get(i, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
