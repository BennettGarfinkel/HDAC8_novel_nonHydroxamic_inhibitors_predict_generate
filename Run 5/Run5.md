## Run 5

First run of the ZBG-periphery-mutation refactor (`ZBG-Mutation` branch). This is the
first run to produce all three execution modes (baseline / separate_zbg / integrated)
side by side from a single script invocation, using the existing trained models
(no retrain — `models/run_state.pkl` reused as-is).

### Execution note (not a pipeline change — how this run was actually produced)
This run was executed in a sandboxed environment where background processes are
killed at each tool-call boundary and a single call has a hard wall-clock limit well
under the ~25 min/mode this GA needs. To get a full run out under those constraints,
each island's hits were checkpointed to disk immediately after that island finished,
and the script was re-invoked repeatedly, reloading already-completed islands from
checkpoint and computing only the remaining ones. Net effect: every island ran the
real GA for the full configured generations/pop size (55 gens x 2 stages x pop 100,
same as the pre-existing config) — but because computation was split across multiple
process restarts, this run's RNG trace is **not** bit-identical to what a single
uninterrupted process with `RANDOM_SEED=42` would produce. Each mode's results are a
legitimate, fully-computed GA output; they're just not exactly reproducible against a
hypothetical one-shot run with the same seed. Worth knowing if a future run is diffed
against this one and doesn't match exactly.

### Results by mode

**baseline** (all 8 islands, crossover 0.5, no periphery mutation — pre-refactor behavior)
- 262 raw hits across 7 lineages (Salicylamide: 0 hits this run)
- 28-row balanced deliverable, 22 cluster reps
- **Final strict screen: 11 rows, all alpha-amino-amide.** Notably *not*
  Carboxylate-dominated this run (see prior runs' known Carboxylate-dominance issue) —
  worth another run to see if that holds or was this seed's outcome.

**separate_zbg** (5 periphery-eligible islands only, crossover off, periphery mutation forced on)
- 111 raw hits pooled from 5 lineages
- 9-row balanced deliverable, 6 cluster reps
- **Final strict screen: 0 rows.** Periphery mutation in isolation (no crossover) didn't
  produce anything clearing every hard constraint simultaneously this run. Real result,
  not an error — the mode ran to completion and exported normally.

**integrated** (all 8 islands, crossover 0.5, periphery mutation at 5% for eligible lineages)
- 353 raw hits across 7 lineages (Trifluoromethyl-ketone thin at 2)
- 38-row balanced deliverable, 26 cluster reps
- **Final strict screen: 37 rows — 27 Carboxylate, 9 alpha-amino-amide, 1
  Trifluoromethyl-ketone.** This run reproduces the known Carboxylate-dominance pattern
  documented in earlier runs (5/6 final candidates being Carboxylate warheads despite
  Carboxylate being the weakest chemotype on real Glide docking, per Run 1-3 findings).
  Worth cross-referencing against real docking again before treating any of these 27 as
  validated.

### Cross-mode takeaway
baseline and integrated diverge sharply in the strict screen (alpha-amino-amide-only vs.
Carboxylate-dominated), despite integrated being baseline-plus-a-small-periphery-mutation-
perturbation. Given the RNG-trace caveat above, some of that divergence could be seed-path
sensitivity rather than a real effect of periphery mutation — can't fully separate the two
without a clean uninterrupted re-run for comparison.

### Repo issues found and fixed/flagged during this session
- `README.md` states the shipped `models/*.pkl` were pickled under scikit-learn 1.8.0.
  Loading them under 1.8.0 throws `InconsistentVersionWarning`; loading under 1.9.0
  throws no warning at all. The models were actually pickled under 1.9.0 — README's
  version pin is wrong. Used 1.9.0 for this run.
