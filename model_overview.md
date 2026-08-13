# How the Model Runs

A walkthrough of what actually happens end to end, from raw data to generated
candidates. Four stages: clean the data, train six models on it, use those
models to score molecules, then use them again to breed new ones.

## Step 0: Data prep

- Raw data gets pulled from the master CSV (`HDAC_Docking_Inhibition.csv`, ~73k rows).
- Data is cleaned and every SMILES string is put into one standard text form
  (canonicalized), so the same molecule written two different ways doesn't
  get counted twice.
- If a molecule has multiple measurements, they're averaged together.
- IC50 gets converted to pIC50 (log scale).
- Invalid molecules for this project get removed, including anything built on
  a hydroxamic acid: that ZBG is off the table entirely.
  hydrazine: very toxic as well
- A Murcko scaffold is computed for every molecule. This is basically the
  skeleton of the molecule — strip off the side chains and substituents and
  keep just the ring system underneath. It's used later both to keep training
  honest (see Step 1) and to sort real molecules into the right GA lineage.
- The cleaned data gets split into separate files, one per model to train.
  Each subset,  HDAC8 IC50, HDAC8 docking, HDAC1 IC50, HDAC1 docking, HDAC6
  IC50, HDAC6 docking, is its own independent model. Six total.

## Step 1: Training

- Each SMILES string gets parsed into an actual molecular object by RDKit.
- That object gets turned into a Morgan fingerprint, also by RDKit. This maps
  the molecule into something a computer can actually reason about — basically
  a long checklist of "is this substructure present, yes or no," repeated
  across a fixed number of bits (256 for the IC50 models, 1024 for docking).
- The model gets handed all of these checklists, with their real potency or
  docking values attached, and has to find patterns — knowing going in that a
  low IC50 or a low (more negative) docking score means "good," and vice versa.
- The model itself is gradient boosted trees — `HistGradientBoostingRegressor`
  from sklearn. In plain terms: it makes a bunch of simple guesses first (e.g.
  "if this one box is checked, guess potent"), looks at what it got wrong and
  by how much, then builds another small tree that corrects for exactly that
  error. Repeat hundreds to thousands of times, stacking corrections on top of
  corrections.
- RDKit also computes a handful of descriptors — rotatable bonds, Lipinski
  violations, and the like — used to judge drug-likeness. These get appended
  to the fingerprint as extra input columns, not treated separately.
- R² gets measured honestly, not just fit on the training data: compounds are
  split into 5 groups by Murcko scaffold, the model trains on 4 of the groups
  and predicts on the 1 held-out group, and R² is computed on just that group.
  Repeat, rotating which group is held out, then average the 5 R² scores
  together. This is done for each of the six models separately, so a model
  never gets credit for having memorized a scaffold it was trained on.

## Step 2: Make a prediction

When the model is handed a molecule, it does the following:

- Maps it into a fingerprint, the same way as training.
- Runs it through the stacked trees, each one altering the guess a bit, and
  adds up all those adjustments to get the final predicted number.
- Checks how much the compound actually resembles things it's seen before
  (how much fingerprint overlap there is) and flags whether it should be
  trusted. This is done with Tanimoto similarity against the training set:
  flagged as low-confidence below 0.35 similarity, hard-removed below 0.20 
- Checks the liability gates (SA score, PAINS/Brenk, metabolic red flags,
  whether it even carries a whitelisted ZBG) to see what gets removed outright
  versus just penalized.

## Step 3: Generate new compounds

- The model mutates a random atom that has a free hydrogen open for a
  substituent. It never touches the ZBG itself. That's locked, so a mutation
  can't accidentally delete the one part of the molecule that actually binds
  zinc.
- It also crosses compounds over to produce new ones: each parent gets split
  into a large piece and a small piece, and children are built from one
  parent's large piece plus the other parent's small piece.
- Selection is Pareto optimization via NSGA-II, not a single blended score.
  Five objectives, scored independently so nothing can buy its way past a
  liability by just being unusually potent: pIC50 (maximized), HDAC8 docking
  score (minimized), liability score (minimized), off-target docking score
  (maximized — weak off-target binding is good), off-target pIC50 (minimized
  — weak off-target potency is good).
- This runs for 55 generations, then reseeds with the top 15 candidates for
  another 55 generations. That's done separately for each ZBG scaffold: 8
  independent islands, one per whitelisted ZBG.
- 1 of those 8 scaffolds (Salicylamide) doesn't have enough real training
  examples (2 real tier1-passing seeds), so it runs largely off a synthetic
  seed molecule instead of real ones. Every other scaffold, including 3-HPT
  (26 real seeds) and alpha-amino-amide (27), clears the GA's 10-real-seed cap
  on its own and doesn't get a synthetic assist.
- The ZBG is re-checked fresh every generation, not just inherited from the
  seed. A mutation/crossover can change a molecule into a different
  chemotype than the one it started as, and this catches that instead of
  mislabeling it.
- Once all 8 islands finish, everything gets pooled together, deliverables
  get assembled (applicability-domain gating, ZBG-precedent checks,
  clustering, balanced export), and the run produces the candidate CSVs and
  the figures.
- Compounds get split into tiers based on predicted IC50, and there's a
  60-molecule cap per island on the final pool — otherwise one or two strong
  scaffolds would flood the results and crowd out everything else.
