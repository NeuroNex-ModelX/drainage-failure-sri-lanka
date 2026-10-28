# %% [markdown]
# # 06 · Model A (expected flooding from nature) and the Drainage Failure Score
# **Owner:** Nadeesha · **Needs:** 05 · **Makes:** `outputs/model_A_metrics.csv`, `outputs/dfs_ds.csv`,
# `outputs/obs_vs_exp_yearly.csv`, `outputs/dfs_stability.csv`, `outputs/known_areas_check.csv`
#
# Proposal sections 6 (Steps 2–3) and 9 (evaluation of Model A, stability, known problem areas).

# %%
# --- SETUP ---
import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print('xgboost not installed - XGBoost skipped')

t = pd.read_csv(os.path.join(PROC, 'analysis_table.csv'))
LAST = int(t.loc[t['label_known'] == 1, 'year'].max())
lab = t[t['label_known'] == 1].copy()

RAIN = ['rain_total', 'rain_max_day', 'days_over_50', 'days_over_100', 'rain_prev_month']
LAND = ['elev_mean', 'elev_min', 'pct_below_10m', 'is_coastal']
SEASON = ['month_sin', 'month_cos']
NATURAL = RAIN + LAND + SEASON
lab[LAND] = lab[LAND].fillna(lab[LAND].median())

# Train/test split by year (proposal: train 1995-2018, test 2019-2023; moves back if records end earlier)
TEST_END = min(LAST, 2023)
TRAIN_END = 2018 if TEST_END >= 2021 else TEST_END - 5
train = lab[lab['year'] <= TRAIN_END]
test = lab[(lab['year'] > TRAIN_END) & (lab['year'] <= TEST_END)]
print(f'Train {train.year.min()}-{TRAIN_END}: {len(train):,} rows ({train.flood.mean():.2%} flooded)')
print(f'Test  {TRAIN_END + 1}-{TEST_END}: {len(test):,} rows ({test.flood.mean():.2%} flooded)')

# %% [markdown]
# ### Model A candidates and the two baselines
# No class weighting, so the predicted probabilities stay usable as *expected numbers of floods*.

# %%
def models():
    m = {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'Random forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                                n_jobs=-1, random_state=0),
    }
    if XGBClassifier is not None:
        m['XGBoost'] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, random_state=0)
    return m


def score(y, p):
    return {'ROC_AUC': roc_auc_score(y, p), 'PR_AUC': average_precision_score(y, p),
            'Brier': brier_score_loss(y, np.clip(p, 0, 1))}


rows, fitted = [], {}
# Baseline 1: rainfall only (logistic regression on the rain features)
b1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[RAIN], train['flood'])
rows.append({'model': 'Baseline: rainfall only', **score(test['flood'], b1.predict_proba(test[RAIN])[:, 1])})
# Baseline 2: flooded in the same month of the previous year
prev = lab[['pcode', 'year', 'month', 'flood']].copy(); prev['year'] += 1
pers = test.merge(prev.rename(columns={'flood': 'prev'}), on=['pcode', 'year', 'month'], how='left')
rows.append({'model': 'Baseline: same month last year', **score(pers['flood'], pers['prev'].fillna(0))})

for name, m in models().items():
    m.fit(train[NATURAL], train['flood'])
    fitted[name] = m
    rows.append({'model': name, **score(test['flood'], m.predict_proba(test[NATURAL])[:, 1])})

metrics = pd.DataFrame(rows).round(3)
metrics['base_rate'] = round(test['flood'].mean(), 4)
metrics.to_csv(os.path.join(OUT, 'model_A_metrics.csv'), index=False)
metrics

# %%
cands = metrics[~metrics['model'].str.startswith('Baseline')]
BEST = cands.sort_values('PR_AUC', ascending=False)['model'].iloc[0]
best_base = metrics[metrics['model'].str.startswith('Baseline')]['PR_AUC'].max()
beats = cands.set_index('model').loc[BEST, 'PR_AUC'] > best_base
print(f'Best Model A: {BEST}  | beats both baselines on PR-AUC: {beats}')
save_meta('model_A', f'{BEST}; test {TRAIN_END + 1}-{TEST_END}')
if not beats:
    print('WARNING: Model A does not beat the baselines -> proposal fallback: report flood-risk ranking honestly.')

# Optional sensitivity check from the proposal: urban divisions only
hf_path = os.path.join(PROC, 'human_factors_ds.csv')
if os.path.exists(hf_path):
    hf = pd.read_csv(hf_path)
    if 'urban_share' in hf and hf['urban_share'].notna().any():
        urban = set(hf.loc[hf['urban_share'] >= 0.5, 'pcode'])
        tu = test[test['pcode'].isin(urban)]
        if tu['flood'].nunique() == 2:
            pu = fitted[BEST].predict_proba(tu[NATURAL])[:, 1]
            print('Urban divisions only:', {k: round(v, 3) for k, v in score(tu['flood'], pu).items()})

# %%
# Which natural factors matter most? (for the report)
m = fitted[BEST]
est = m[-1] if hasattr(m, 'steps') else m
if hasattr(est, 'feature_importances_'):
    imp = pd.Series(est.feature_importances_, index=NATURAL)
else:
    imp = pd.Series(np.abs(est.coef_[0]), index=NATURAL)
imp.sort_values().plot(kind='barh', title=f'Model A ({BEST}) – natural factor importance', figsize=(6, 4))

# %% [markdown]
# ### Drainage Failure Score
# Expected floods for **every** labelled month (up to 2023) come from *out-of-fold* predictions:
# the model never predicts a year it was trained on. Then per division:
#
# **DFS = (actual floods − expected floods) / expected floods**
#
# Divisions with too little evidence (expected < 1 flood, or < 10 heavy-rain months) are marked
# *low confidence* and not ranked.

# %%
SCORE_END = min(LAST, 2023)
base = lab[lab['year'] <= SCORE_END].reset_index(drop=True)
oof = np.zeros(len(base))
for tr, te in GroupKFold(n_splits=5).split(base, groups=base['year']):
    mdl = models()[BEST]
    if BEST != 'Logistic regression':
        mdl = CalibratedClassifierCV(mdl, method='sigmoid', cv=3)
    mdl.fit(base.loc[tr, NATURAL], base.loc[tr, 'flood'])
    oof[te] = mdl.predict_proba(base.loc[te, NATURAL])[:, 1]
base['expected'] = oof
print(f"Check: actual floods {base['flood'].sum():,.0f} vs expected {base['expected'].sum():,.0f} (should be close)")
print('Out-of-fold PR-AUC:', round(average_precision_score(base['flood'], oof), 3))


MIN_EXPECTED, MIN_HEAVY_MONTHS = 1.0, 10   # 'low confidence' thresholds (lower them if too many divisions drop out)


def dfs_table(df):
    g = df.groupby('pcode').agg(actual=('flood', 'sum'), expected=('expected', 'sum'),
                                heavy_rain_months=('days_over_50', lambda x: int((x > 0).sum())))
    g['dfs'] = (g['actual'] - g['expected']) / g['expected']
    g['low_confidence'] = (g['expected'] < MIN_EXPECTED) | (g['heavy_rain_months'] < MIN_HEAVY_MONTHS)
    return g


dfs = dfs_table(base)
conf = dfs[~dfs['low_confidence']]
dfs['rank'] = conf['dfs'].rank(ascending=False, method='min')
dfs['percentile'] = conf['dfs'].rank(pct=True) * 100
dfs = dfs.reset_index().merge(load_ds_list()[['pcode', 'ds_name', 'district_name']], on='pcode')
dfs.to_csv(os.path.join(OUT, 'dfs_ds.csv'), index=False)
save_meta('dfs_period', f'{base.year.min()}-{SCORE_END}')
print(f"{(~dfs['low_confidence']).sum()} divisions ranked, {dfs['low_confidence'].sum()} low-confidence")
dfs.sort_values('dfs', ascending=False).head(15)[['ds_name', 'district_name', 'actual', 'expected', 'dfs', 'low_confidence']]

# %%
# Yearly actual vs expected per division (used by the dashboard's division profile)
yearly = base.groupby(['pcode', 'year']).agg(actual=('flood', 'sum'), expected=('expected', 'sum')).reset_index()
yearly.to_csv(os.path.join(OUT, 'obs_vs_exp_yearly.csv'), index=False)

# %% [markdown]
# ### Stability: do rankings change a lot when one year is removed?

# %%
ranked = conf.index
stab = []
for y in sorted(base['year'].unique()):
    d2 = dfs_table(base[base['year'] != y]).reindex(ranked)
    rho = spearmanr(conf['dfs'], d2['dfs'], nan_policy='omit').correlation
    top_full = set(conf['dfs'].nlargest(20).index); top_y = set(d2['dfs'].nlargest(20).index)
    stab.append({'year_removed': y, 'spearman': rho, 'top20_overlap': len(top_full & top_y) / 20})
stab = pd.DataFrame(stab)
stab.to_csv(os.path.join(OUT, 'dfs_stability.csv'), index=False)
print(stab[['spearman', 'top20_overlap']].describe().round(3))
print('Worst case (removing', int(stab.loc[stab.spearman.idxmin(), 'year_removed']), '): rho =',
      round(stab.spearman.min(), 3))

# %% [markdown]
# ### Agreement with known problem areas (sanity check, not used to build the score)

# %%
KNOWN = ['Kolonnawa', 'Kotte', 'Kaduwela', 'Wattala', 'Ja-Ela']   # proposal section 1 / ref [7]
rows = []
for k in KNOWN:
    hit = dfs[dfs['ds_name'].map(norm_name).str.contains(norm_name(k))]
    for _, r in hit.iterrows():
        rows.append({'known_area': k, 'ds_name': r['ds_name'], 'dfs': round(r['dfs'], 2),
                     'percentile': None if pd.isna(r['percentile']) else round(r['percentile'], 1),
                     'low_confidence': r['low_confidence']})
known = pd.DataFrame(rows)
known.to_csv(os.path.join(OUT, 'known_areas_check.csv'), index=False)
known
