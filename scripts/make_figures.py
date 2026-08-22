"""
Figure generation for the HDAC8 docking-selective pipeline.

Produces nine figures (fig1..fig9) from the docking-selective run outputs plus the
trained model bundles. Designed to be called from the notebook (import and run
make_all_figures(...)) or standalone (python3 make_figures.py from the notebooks/
or run dir, with ../models and ../candidates present).

Every figure is drawn from data actually produced by the pipeline -- nothing is
mocked. Where a figure needs the raw hit pool it reads ga_candidates_RAW.csv; where
it needs model metadata it reads the bundles out of the run-state pickle.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY = '#1b2a4a'
TEAL = '#2a9d8f'
CORAL = '#e76f51'
GOLD = '#e9c46a'
SLATE = '#8d99ae'
ZBG_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))


def _load(cand_dir, models_dir):
    raw = pd.read_csv(os.path.join(cand_dir, 'ga_candidates_RAW.csv'))
    with open(os.path.join(models_dir, 'run_state.pkl'), 'rb') as f:
        st = pickle.load(f)
    return raw, st


def fig1_model_performance(st, out):
    """Scaffold-grouped CV R^2 for all six trained models."""
    names = ['HDAC8\nIC50', 'HDAC8\ndock', 'HDAC1\nIC50', 'HDAC6\nIC50', 'HDAC1\ndock', 'HDAC6\ndock']
    keys = ['ic50_bundle', 'dock_bundle', 'hdac1_bundle', 'hdac6_bundle', 'hdac1_dock_bundle', 'hdac6_dock_bundle']
    r2 = [st[k]['cv_r2'] for k in keys]
    n = [len(st[k]['train_fps']) for k in keys]
    colors = [TEAL, TEAL, SLATE, SLATE, CORAL, CORAL]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, r2, color=colors, edgecolor=NAVY, linewidth=0.8)
    for b, v, c in zip(bars, r2, n):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'R²={v:.2f}\nn={c:,}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Scaffold-grouped cross-validated R²')
    ax.set_title('Model performance across all six trained regressors')
    ax.set_ylim(0, max(r2) * 1.25)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig2_training_pic50(st, out):
    """Training pIC50 distributions for the three IC50 targets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ic50_df = st['ic50_df']
    ax.hist(ic50_df['pIC50'], bins=40, alpha=0.6, color=TEAL, label=f"HDAC8 (n={len(ic50_df):,})", density=True)
    ax.axvline(ic50_df['pIC50'].median(), color=TEAL, ls='--', lw=1)
    ax.set_xlabel('pIC50 (higher = more potent)')
    ax.set_ylabel('Density')
    ax.set_title('HDAC8 training-set potency distribution')
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig3_docking_selectivity_landscape(raw, out):
    """HDAC8 docking vs. weaker off-target docking, colored by ZBG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    raw = raw.copy()
    raw['weak_off'] = raw[['docking_hdac1_pred', 'docking_hdac6_pred']].max(axis=1)
    tags = sorted(raw['zbg_tag'].unique())
    for tag, c in zip(tags, ZBG_PALETTE):
        s = raw[raw['zbg_tag'] == tag]
        ax.scatter(s['docking_pred'], s['weak_off'], s=20, alpha=0.55, color=c, label=f'{tag} ({len(s)})')
    ax.axvline(-7, color='k', ls='--', lw=1)
    ax.axhline(-7, color='k', ls='--', lw=1)
    ax.axvspan(ax.get_xlim()[0], -7, ymin=0, ymax=1, color=TEAL, alpha=0.04)
    ax.set_xlabel('Predicted HDAC8 docking (target ≤ -7, strong)')
    ax.set_ylabel('Weaker of HDAC1/HDAC6 predicted docking (target > -7, weak)')
    ax.set_title('Docking-selectivity landscape\n(target region: lower-right of the crosshairs)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig4_zbg_distribution(raw, out):
    """Hit count per ZBG lineage."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = raw['zbg_tag'].value_counts().sort_values()
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(counts)))
    ax.barh(counts.index, counts.values, color=colors, edgecolor=NAVY, linewidth=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=9)
    ax.set_xlabel('Docking + IC50 selectivity-gated hits')
    ax.set_title('Hit distribution across ZBG chemotypes')
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig5_docking_by_target(raw, out):
    """Boxplot of predicted docking across the three isoforms."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [raw['docking_pred'], raw['docking_hdac1_pred'], raw['docking_hdac6_pred']]
    bp = ax.boxplot(data, tick_labels=['HDAC8\n(on-target)', 'HDAC1\n(off-target)', 'HDAC6\n(off-target)'],
                    patch_artist=True, medianprops=dict(color=NAVY))
    for patch, color in zip(bp['boxes'], [TEAL, CORAL, GOLD]):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    ax.axhline(-7, color='k', ls='--', lw=1, label='Selectivity threshold (-7)')
    ax.set_ylabel('Predicted docking score')
    ax.set_title('Predicted docking by isoform\n(HDAC8 strong, HDAC1/HDAC6 weak by design)')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig6_potency_vs_selectivity(raw, out):
    """HDAC8 pIC50 vs. minimum IC50-selectivity margin, colored by docking selectivity."""
    fig, ax = plt.subplots(figsize=(8, 6))
    raw = raw.copy()
    raw['min_sel'] = raw[['selectivity_vs_hdac1', 'selectivity_vs_hdac6']].min(axis=1)
    sc = ax.scatter(raw['pIC50_pred'], raw['min_sel'], c=raw['docking_selectivity'],
                    cmap='viridis', s=24, alpha=0.7, edgecolor='none')
    ax.axhline(0.7, color=CORAL, ls='--', lw=1, label='Min IC50 selectivity (0.7 log)')
    ax.set_xlabel('Predicted HDAC8 pIC50')
    ax.set_ylabel('Min IC50 selectivity margin vs. HDAC1/HDAC6 (log units)')
    ax.set_title('Potency vs. IC50-selectivity, colored by docking selectivity')
    cb = plt.colorbar(sc); cb.set_label('Docking selectivity (min off-target − HDAC8)')
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig7_property_space(raw, out):
    """cLogP vs. MW with Lipinski box; point size by RotB."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if 'MW' not in raw.columns:
        ax.text(0.5, 0.5, 'MW/cLogP columns not present in this run', ha='center')
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); return
    sizes = 20 + (raw['RotB'].fillna(0) * 8)
    tags = sorted(raw['zbg_tag'].unique())
    for tag, c in zip(tags, ZBG_PALETTE):
        s = raw[raw['zbg_tag'] == tag]
        ax.scatter(s['MW'], s['cLogP'], s=sizes[s.index], alpha=0.55, color=c, label=tag, edgecolor='none')
    ax.axvline(500, color='k', ls='--', lw=1)
    ax.axhline(5, color='k', ls='--', lw=1)
    ax.set_xlabel('Molecular weight (Da)')
    ax.set_ylabel('cLogP')
    ax.set_title('Physicochemical property space\n(dashed = Lipinski limits; point size ∝ rotatable bonds)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig8_tier_by_scaffold(raw, out):
    """Stacked potency-tier composition per ZBG."""
    fig, ax = plt.subplots(figsize=(9, 5))
    tiers = ['Tier 1 (<150 nM)', 'Tier 2 (150-300 nM)', 'Tier 3 (300-500 nM)']
    ct = pd.crosstab(raw['zbg_tag'], raw['IC50_tier']).reindex(columns=tiers, fill_value=0)
    colorsT = [TEAL, GOLD, CORAL]
    bottom = None
    for tier, color in zip(tiers, colorsT):
        v = ct[tier].values
        ax.barh(ct.index[::-1], v[::-1], left=(bottom[::-1] if bottom is not None else None),
                color=color, label=tier, edgecolor=NAVY, linewidth=0.5)
        bottom = v if bottom is None else bottom + v
    ax.set_xlabel('Hits')
    ax.set_title('Potency-tier composition by ZBG scaffold')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig9_sa_vs_potency(raw, out):
    """Synthetic accessibility vs. potency, flagging the clean drug-like corner."""
    fig, ax = plt.subplots(figsize=(8, 6))
    clean = raw[(~raw['pains_brenk_flagged']) & (raw['sa_score'] <= 3.5)]
    dirty = raw.drop(clean.index)
    ax.scatter(dirty['pIC50_pred'], dirty['sa_score'], s=20, alpha=0.4, color=SLATE, label='Flagged / SA>3.5')
    ax.scatter(clean['pIC50_pred'], clean['sa_score'], s=28, alpha=0.8, color=TEAL,
               edgecolor=NAVY, linewidth=0.5, label='Clean & SA≤3.5')
    ax.axhline(3.5, color=CORAL, ls='--', lw=1)
    ax.set_xlabel('Predicted HDAC8 pIC50')
    ax.set_ylabel('Synthetic accessibility score (lower = easier)')
    ax.set_title('Synthetic accessibility vs. potency\n(teal = passes PAINS/Brenk and SA screen)')
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def make_all_figures(cand_dir='../candidates', models_dir='../models', fig_dir='../figures'):
    os.makedirs(fig_dir, exist_ok=True)
    raw, st = _load(cand_dir, models_dir)
    fig1_model_performance(st, os.path.join(fig_dir, 'fig1_model_performance.png'))
    fig2_training_pic50(st, os.path.join(fig_dir, 'fig2_training_pic50_dist.png'))
    fig3_docking_selectivity_landscape(raw, os.path.join(fig_dir, 'fig3_docking_selectivity_landscape.png'))
    fig4_zbg_distribution(raw, os.path.join(fig_dir, 'fig4_zbg_distribution.png'))
    fig5_docking_by_target(raw, os.path.join(fig_dir, 'fig5_docking_by_target.png'))
    fig6_potency_vs_selectivity(raw, os.path.join(fig_dir, 'fig6_potency_vs_selectivity.png'))
    fig7_property_space(raw, os.path.join(fig_dir, 'fig7_property_space.png'))
    fig8_tier_by_scaffold(raw, os.path.join(fig_dir, 'fig8_tier_by_scaffold.png'))
    fig9_sa_vs_potency(raw, os.path.join(fig_dir, 'fig9_sa_vs_potency.png'))
    return [f'fig{i}' for i in range(1, 10)]


# ---------------------------------------------------------------------------
# Internal diagnostics -- NOT part of the poster/candidate figure set
# ---------------------------------------------------------------------------
# Everything below this line is development-only validation. It is deliberately kept
# out of make_all_figures(), writes to its own figures/diagnostic/ folder, and must not
# be used to justify any change to the production models, the GA, or candidate output.


def make_ic50_parity_diagnostic(data_dir='../data', out_path='../figures/diagnostic/ic50_parity.png'):
    """Out-of-fold predicted vs. actual pIC50 for the three IC50 models.

    A visual complement to the scaffold-CV R2 numbers train_all_models.py already
    prints: it shows WHERE the error sits rather than just how much there is. Expect
    tight clustering near the diagonal in the mid-potency range and wider scatter at
    both extremes, which is the known assay-noise ceiling, not a defect.

    Slower than the fig1..fig9 set because it refits each model five times via
    cross_val_predict instead of reusing the saved run_state.pkl bundles, so it is not
    called from make_all_figures(). Run it explicitly:
        python3 make_figures.py --diagnostic

    Features come from pipeline.build_training_features() and the boosting settings from
    pipeline.get_hgb_params('pIC50'), so this stays in step with the real training path
    instead of carrying its own copy of either.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import r2_score
    from pipeline import build_training_features, get_hgb_params

    targets = [
        ('HDAC8', 'hdac8_ic50_clean_MERGED.csv', TEAL),
        ('HDAC1', 'hdac1_ic50_clean.csv', SLATE),
        ('HDAC6', 'hdac6_ic50_clean.csv', CORAL),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, fname, color) in zip(axes, targets):
        df = pd.read_csv(os.path.join(data_dir, fname))
        X, y, groups = build_training_features(df, value_col='pIC50')
        y_pred = cross_val_predict(
            HistGradientBoostingRegressor(**get_hgb_params('pIC50')),
            X, y, cv=GroupKFold(n_splits=5), groups=groups, n_jobs=-1)
        # Pooled over every out-of-fold point (1 - SS_res/SS_tot), not an average of
        # per-fold scores, so it will sit near but not exactly on the cross_val_score
        # figure train_all_models.py reports.
        r2 = r2_score(y, y_pred)

        ax.scatter(y, y_pred, s=12, alpha=0.35, color=color, edgecolor='none')
        lo = min(y.min(), y_pred.min()) - 0.25
        hi = max(y.max(), y_pred.max()) + 0.25
        ax.plot([lo, hi], [lo, hi], ls='--', lw=1, color=NAVY)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{name} IC50\nR² = {r2:.3f}  (n = {len(y):,})')
        ax.set_xlabel('Actual pIC50')
        ax.set_ylabel('Predicted pIC50 (out-of-fold)')
        ax.grid(alpha=0.25)

    fig.suptitle('Out-of-fold parity, scaffold-grouped 5-fold CV (internal diagnostic)',
                 fontsize=12, y=0.99)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # rect keeps the suptitle clear of the panel titles; bbox_inches='tight' keeps the
    # equal-aspect panels from clipping their own axis labels.
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    return out_path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic', action='store_true',
                    help='build the internal IC50 parity diagnostic instead of the '
                         'fig1..fig9 set (slow: refits the IC50 models via 5-fold CV)')
    args = ap.parse_args()
    if args.diagnostic:
        print('Wrote:', make_ic50_parity_diagnostic())
    else:
        figs = make_all_figures()
        print('Wrote:', figs)
