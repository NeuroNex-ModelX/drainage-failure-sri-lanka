# %% [markdown]
# # 07 · Model B: why does drainage fail? (Random Forest + SHAP + failure profiles)
# **Owner:** Nadeesha · **Needs:** 04, 06 · **Makes:** `outputs/causes_ds.csv`, `outputs/model_B_metrics.csv`,
# `outputs/shap_values.csv`, `outputs/shap_stability.csv`
#
# Proposal section 6 Step 4 and section 9 (Model B evaluation).
# Results are **factors linked to** failure, not proof of cause (proposal, risks table).

# %%
# --- SETUP ---
import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'shap'])
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    import shap
except ImportError:
    shap = None
    print('shap not installed - using a simpler contribution estimate')

dfs = pd.read_csv(os.path.join(OUT, 'dfs_ds.csv'))
hf = pd.read_csv(os.path.join(PROC, 'human_factors_ds.csv'))
d = dfs[~dfs['low_confidence']].merge(hf.drop(columns=['ds_name', 'district_name']), on='pcode', how='left')

CANDIDATES = ['est_waste_collection', 'urban_share', 'pop_density', 'pop_growth_2012_2024',
              'hh_piped_water_share', 'hh_flush_toilet_share', 'hh_clean_cooking_share',
              'tree_loss_pct_2001_2024', 'tree_loss_pct_2015_2024', 'drain_density', 'stream_density']
HUMAN = [c for c in CANDIDATES if c in d and d[c].notna().mean() >= 0.7]
dropped = [c for c in CANDIDATES if c in d and c not in HUMAN]
print('Human factors used:', HUMAN)
print('Dropped (missing for >30% of divisions):', dropped)
# urban_share and est_waste_collection are the same information - keep only the waste one
if {'urban_share', 'est_waste_collection'} <= set(HUMAN):
    HUMAN.remove('urban_share')

X = d[HUMAN].fillna(d[HUMAN].median())
# DFS is skewed (a few very large values) -> clip extremes so they don't dominate the fit
y = d['dfs'].clip(d['dfs'].quantile(0.02), d['dfs'].quantile(0.98))
groups = d['district_name']
print(len(d), 'divisions in Model B')

# %% [markdown]
# ### Cross-validated R², folds grouped by district
# Neighbouring divisions are similar, so whole districts are held out together.
# Repeated 10 times with different district groupings.

# %%
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)
r2_rf, r2_dummy = [], []
dists = groups.unique()
for rep in range(10):
    rng = np.random.default_rng(rep)
    shuffled = dict(zip(dists, rng.permutation(len(dists))))
    g = groups.map(shuffled)
    cv = GroupKFold(n_splits=5)
    r2_rf += list(cross_val_score(rf, X, y, cv=cv, groups=g, scoring='r2'))
    r2_dummy += list(cross_val_score(DummyRegressor(), X, y, cv=cv, groups=g, scoring='r2'))
mB = pd.DataFrame({'model': ['Random forest (human factors)', 'Always predict the average'],
                   'cv_R2_mean': [np.mean(r2_rf), np.mean(r2_dummy)],
                   'cv_R2_sd': [np.std(r2_rf), np.std(r2_dummy)]}).round(3)
mB.to_csv(os.path.join(OUT, 'model_B_metrics.csv'), index=False)
mB

# %% [markdown]
# ### SHAP: how much each factor pushes each division's score up or down

# %%
rf.fit(X, y)


def contributions(model, X):
    if shap is not None:
        return pd.DataFrame(shap.TreeExplainer(model).shap_values(X), columns=X.columns, index=X.index)
    # fallback (only if shap cannot be installed): effect of moving each feature to its median
    base_pred = model.predict(X)
    out = {}
    for c in X.columns:
        Xm = X.copy(); Xm[c] = X[c].median()
        out[c] = base_pred - model.predict(Xm)
    return pd.DataFrame(out, index=X.index)


sv = contributions(rf, X)
sv.insert(0, 'pcode', d['pcode'].values)
sv.to_csv(os.path.join(OUT, 'shap_values.csv'), index=False)

if shap is not None:
    shap.summary_plot(sv[HUMAN].values, X, show=True)   # screenshot for the report
global_imp = sv[HUMAN].abs().mean().sort_values(ascending=False)
print('Mean |SHAP| (overall importance):\n', global_imp.round(3))

# %% [markdown]
# ### Stability: do the top SHAP factors stay the same when we resample?

# %%
top3_counts = pd.Series(0, index=HUMAN, dtype=float)
B = 50
for b in range(B):
    idx = np.random.default_rng(100 + b).choice(len(X), len(X), replace=True)
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=b, n_jobs=-1)
    m.fit(X.iloc[idx], y.iloc[idx])
    s = contributions(m, X).abs().mean()
    top3_counts[s.nlargest(3).index] += 1
stab = (top3_counts / B).sort_values(ascending=False).rename('share_of_resamples_in_top3').reset_index()
stab.columns = ['factor', 'share_in_top3']
stab.to_csv(os.path.join(OUT, 'shap_stability.csv'), index=False)
stab

# %% [markdown]
# ### Main cause per division and failure profiles (K-means on SHAP values)

# %%
NICE = {  # plain-language names for the dashboard
    'est_waste_collection': 'waste collection (est.)', 'urban_share': 'urbanisation',
    'pop_density': 'population density', 'pop_growth_2012_2024': 'population growth',
    'hh_piped_water_share': 'piped-water coverage', 'hh_flush_toilet_share': 'flush-toilet coverage',
    'hh_clean_cooking_share': 'clean-cooking coverage', 'tree_loss_pct_2001_2024': 'tree-cover loss',
    'tree_loss_pct_2015_2024': 'recent tree-cover loss', 'drain_density': 'drain/canal density',
    'stream_density': 'stream density',
}
med = X.median()


def label(factor, value):
    """'High population density', 'Low drain/canal density', ... (direction = above/below median)."""
    if factor == 'none':
        return 'No clear human factor'
    return ('High ' if value > med[factor] else 'Low ') + NICE.get(factor, factor)


pos = sv[HUMAN].clip(lower=0)            # only factors that push the score UP
main = pos.idxmax(axis=1).where(pos.max(axis=1) > 0, 'none')
second = pos.apply(lambda r: r.nlargest(2).index[-1] if (r > 0).sum() > 1 else 'none', axis=1)
main_lbl = [label(f, X.iloc[i][f] if f != 'none' else 0) for i, f in enumerate(main)]
second_lbl = ['' if f == 'none' else label(f, X.iloc[i][f]) for i, f in enumerate(second)]

K = 4 if len(sv) >= 40 else 2
km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(StandardScaler().fit_transform(sv[HUMAN]))
prof = pd.Series(km.labels_, index=sv.index)
names = {}
for k in range(K):
    members = prof == k
    mean_sv = sv.loc[members, HUMAN].mean()
    top = mean_sv.idxmax()
    if mean_sv.max() <= 0 or y[members.values].mean() <= y.median():
        names[k] = 'Lower failure (floods close to expected)'
    else:
        names[k] = label(top, X.loc[members.values, top].median()) + '-driven'
orig = dict(names)
counts, seen = pd.Series(list(orig.values())).value_counts(), {}
for k, n in orig.items():   # two clusters can get the same name -> number them
    if counts[n] > 1:
        seen[n] = seen.get(n, 0) + 1
        names[k] = f'{n} ({seen[n]})'
causes = pd.DataFrame({
    'pcode': d['pcode'].values,
    'main_cause_factor': main.values,
    'main_cause': main_lbl,
    'second_cause': second_lbl,
    'failure_profile': prof.map(names).values,
})
# A 'cause of failure' only makes sense where the division floods MORE than expected
not_failing = (d['dfs'] <= 0).values
causes.loc[not_failing, ['main_cause', 'second_cause']] = ['Not failing (floods at or below expected)', '']
causes.to_csv(os.path.join(OUT, 'causes_ds.csv'), index=False)
print(causes['failure_profile'].value_counts())
causes.merge(dfs[['pcode', 'ds_name', 'dfs']], on='pcode').sort_values('dfs', ascending=False).head(15)
