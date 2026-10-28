# %% [markdown]
# # 09 · Test on new events: 2024 floods and Cyclone Ditwah (2025)
# **Owner:** All · **Needs:** 02, 04, 06 · **Makes:** `outputs/eval_recent_events.csv`
#
# Proposal section 9: scores are computed from data up to 2023. Do divisions with a higher
# Drainage Failure Score have **more affected families per 1,000 residents** in the 2024 floods and
# Cyclone Ditwah, **after accounting for how much rain they received**?
#
# **Inputs you type in (templates are created on first run):**
# * `data/manual/recent_events_ds.csv` – affected families by **DS division** (DMC situation reports)
# * `data/manual/recent_events_district.csv` – affected families by **district** (DMC 30 Nov sitrep,
#   IFRC Oct 2024 DREF, Ditwah R package tables)
# * optional: flood-extent maps in `data/raw/flood_extents/<event>/` (UNOSAT June 2024, EMSR851)

# %%
# --- SETUP ---
import os, sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *
from scipy import stats

# Rain windows for each event - CHECK these dates against the situation reports
EVENTS = {
    'Jun2024':    ('2024-05-30', '2024-06-05'),   # SW-monsoon floods (UNOSAT map of 4 June 2024)
    'Oct2024':    ('2024-10-08', '2024-10-15'),   # IFRC DREF MDRLK020
    'Ditwah2025': ('2025-11-24', '2025-12-01'),   # Cyclone Ditwah
}

ds_list = load_ds_list()
dfs = pd.read_csv(os.path.join(OUT, 'dfs_ds.csv'))
hf = pd.read_csv(os.path.join(PROC, 'human_factors_ds.csv'))
pop = hf[['pcode', 'population_2024']] if 'population_2024' in hf else pd.DataFrame(columns=['pcode', 'population_2024'])
rain = pd.read_csv(os.path.join(PROC, 'rain_daily.csv.gz'), parse_dates=['date'])

ev_rain = []
for ev, (a, b) in EVENTS.items():
    r = rain[(rain['date'] >= a) & (rain['date'] <= b)].groupby('pcode')['rain_mm'].sum().rename('event_rain_mm')
    ev_rain.append(r.reset_index().assign(event=ev))
ev_rain = pd.concat(ev_rain)
print(ev_rain.groupby('event')['event_rain_mm'].describe().round(0))

# %%
# Templates (first run only)
ds_path = os.path.join(MANUAL, 'recent_events_ds.csv')
dist_path = os.path.join(MANUAL, 'recent_events_district.csv')
if not os.path.exists(ds_path):
    pd.DataFrame(columns=['event', 'district_name', 'ds_name', 'affected_families', 'source']).to_csv(ds_path, index=False)
    print('Template created:', ds_path)
if not os.path.exists(dist_path):
    tmpl = pd.DataFrame([(e, d, '', '') for e in EVENTS for d in sorted(ds_list['district_name'].unique())],
                        columns=['event', 'district_name', 'affected_families', 'source'])
    tmpl.to_csv(dist_path, index=False)
    print('Template created:', dist_path)


# %%
def ols(y, X, names):
    """Ordinary least squares with p-values (kept simple, no extra libraries)."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({'coef': beta, 'p_value': p}, index=['const'] + names), r2


def evaluate(df, level, outcome, label):
    """outcome ~ standardised event rain + standardised DFS."""
    df = df.dropna(subset=[outcome, 'event_rain_mm', 'dfs'])
    if len(df) < 8:
        return None
    z = lambda s: (s - s.mean()) / s.std()
    coefs, r2 = ols(df[outcome].values, np.column_stack([z(df['event_rain_mm']), z(df['dfs'])]),
                    ['rain_z', 'dfs_z'])
    rho = stats.spearmanr(df['dfs'], df[outcome]).correlation
    return {'event': df['event'].iloc[0], 'level': level, 'outcome': label, 'n': len(df),
            'coef_dfs_z': coefs.loc['dfs_z', 'coef'], 'p_dfs': coefs.loc['dfs_z', 'p_value'],
            'coef_rain_z': coefs.loc['rain_z', 'coef'], 'R2': r2, 'spearman_dfs_vs_outcome': rho}


results = []
ranked = dfs[~dfs['low_confidence']][['pcode', 'dfs', 'district_name']]

# --- A. DS-division level (DMC situation reports) ---
evds = pd.read_csv(ds_path)
if len(evds):
    evds = match_ds(evds, 'district_name', 'ds_name', 'ds_lookup_events.csv').drop(columns=['district_name'])
    evds = evds.merge(pop, on='pcode', how='left').merge(ranked, on='pcode').merge(ev_rain, on=['pcode', 'event'])
    evds['affected_per_1000'] = 1000 * pd.to_numeric(evds['affected_families'], errors='coerce') / evds['population_2024']
    for ev, g in evds.groupby('event'):
        res = evaluate(g, 'DS division', 'affected_per_1000', 'affected families per 1,000 residents')
        if res: results.append(res)

# --- B. District level (DMC / IFRC / Ditwah package) ---
evd = pd.read_csv(dist_path)
evd['affected_families'] = pd.to_numeric(evd['affected_families'], errors='coerce')
evd = evd.dropna(subset=['affected_families'])
if len(evd):
    dmap = {norm_name(d): d for d in ds_list['district_name'].unique()}
    evd['district_name'] = evd['district_name'].map(lambda s: dmap.get(norm_name(s), s))
    by_d = ds_list[['pcode', 'district_name']].merge(pop, on='pcode', how='left')
    d_pop = by_d.groupby('district_name')['population_2024'].sum(min_count=1).rename('population_2024')
    d_dfs = ranked.groupby('district_name')['dfs'].median().rename('dfs')
    d_rain = ev_rain.merge(ds_list[['pcode', 'district_name']], on='pcode').groupby(
        ['event', 'district_name'])['event_rain_mm'].mean().reset_index()
    evd = evd.merge(d_pop, on='district_name', how='left').merge(d_dfs, on='district_name').merge(
        d_rain, on=['event', 'district_name'])
    evd['affected_per_1000'] = 1000 * evd['affected_families'] / evd['population_2024']
    for ev, g in evd.groupby('event'):
        res = evaluate(g, 'District', 'affected_per_1000', 'affected families per 1,000 residents')
        if res: results.append(res)

# %% [markdown]
# ### Optional: satellite flood extents (share of each division under water)
# Put the downloaded layers in `data/raw/flood_extents/Jun2024/` and `.../Ditwah2025/`
# (shapefile, GeoPackage or GeoJSON). Folder name = event name used above.

# %%
ext_root = os.path.join(RAW, 'flood_extents')
if os.path.isdir(ext_root) and os.listdir(ext_root):
    import geopandas as gpd
    poly = gpd.read_file(os.path.join(PROC, 'ds_divisions.gpkg'))[['pcode', 'geometry']].to_crs(SL_CRS)
    poly['area'] = poly.area
    for ev in os.listdir(ext_root):
        files = find_files(os.path.join(ext_root, ev), [r'\.(shp|gpkg|geojson)$'])
        if not files or ev not in EVENTS:
            continue
        water = pd.concat([gpd.read_file(f).to_crs(SL_CRS)[['geometry']] for f in files])
        water = water[water.geom_type.isin(['Polygon', 'MultiPolygon'])]
        inter = gpd.overlay(poly, water, how='intersection', keep_geom_type=True)
        inter['wet'] = inter.area
        frac = (inter.groupby('pcode')['wet'].sum() / poly.set_index('pcode')['area']).rename('flooded_pct') * 100
        sat = poly[['pcode']].merge(frac.reset_index(), on='pcode', how='left').fillna({'flooded_pct': 0})
        # only divisions the map actually covered: keep those with any water or inside the map's extent
        bbox = water.total_bounds
        cen = poly.geometry.centroid
        sat = sat[(cen.x.between(bbox[0], bbox[2]) & cen.y.between(bbox[1], bbox[3])).values]
        sat = sat.merge(ranked, on='pcode').merge(ev_rain[ev_rain['event'] == ev], on='pcode')
        res = evaluate(sat, 'DS division', 'flooded_pct', '% of division flooded (satellite)')
        if res: results.append(res)
        sat.to_csv(os.path.join(OUT, f'satellite_flooded_{ev}.csv'), index=False)

# %%
ev_out = pd.DataFrame(results).round(4)
ev_out.to_csv(os.path.join(OUT, 'eval_recent_events.csv'), index=False)
if ev_out.empty:
    print('No event data entered yet - fill the templates and re-run.')
else:
    print('Positive coef_dfs_z with small p_dfs = higher-scoring divisions were hit harder, beyond what rain explains.')
ev_out
