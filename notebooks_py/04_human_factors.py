# %% [markdown]
# # 04 · Human factors per DS division (census, tree-cover loss, drains)
# **Owner:** Jithmi · **Needs:** notebook 00 · **Makes:** `data/processed/human_factors_ds.csv`
#
# **Before running, put these files in place:**
# | File | Where it comes from |
# |---|---|
# | `data/manual/census_ds.csv` | Census 2024 tables (statistics.gov.lk), typed/copied into the template made by this notebook |
# | `data/manual/census_gn.csv` *(optional)* | GN-level toilet / water / cooking-fuel counts; summed to DS here |
# | `data/raw/gfw/*.csv` | Global Forest Watch ▸ Sri Lanka dashboard ▸ Download (tree cover loss, subnational) |
# | `data/raw/osm/` | Geofabrik `sri-lanka-latest-free.shp.zip` (upload the zip) |
#
# Each part is optional: if a file is missing that factor is skipped and Model B uses the rest.

# %%
# --- SETUP ---
import os, sys, zipfile
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *

ds_list = load_ds_list()
hf = ds_list[['pcode', 'ds_name', 'district_name', 'area_km2']].copy()

# %% [markdown]
# ## A. Census (population, households, urban share, services)
# The first run creates `data/manual/census_ds.csv` with one row per division and empty columns.
# Fill what you have (Google Sheets → download as CSV works well), then re-run.
#
# * `population_2024`, `households_2024` – Census 2024, by DS division
# * `population_2012` – Census 2012 (for growth), optional
# * `urban_share` – share of population in urban sector (0–1)
# * `hh_piped_water_share`, `hh_flush_toilet_share`, `hh_clean_cooking_share` – service coverage (0–1)

# %%
census_path = os.path.join(MANUAL, 'census_ds.csv')
CENSUS_COLS = ['population_2024', 'households_2024', 'population_2012', 'urban_share',
               'hh_piped_water_share', 'hh_flush_toilet_share', 'hh_clean_cooking_share']
if not os.path.exists(census_path):
    tmpl = ds_list[['pcode', 'district_name', 'ds_name']].copy()
    for c in CENSUS_COLS:
        tmpl[c] = ''
    tmpl.to_csv(census_path, index=False)
    print('Template created ->', census_path, '\nFill it in and re-run this notebook.')

census = pd.read_csv(census_path)
if census['pcode'].isna().any():          # rows typed by name only -> match them
    census = match_ds(census, 'district_name', 'ds_name', 'ds_lookup_census.csv')

# Optional GN-level counts -> sum to DS and turn into shares
gn_path = os.path.join(MANUAL, 'census_gn.csv')
if os.path.exists(gn_path):
    gn = pd.read_csv(gn_path)
    gn = match_ds(gn, 'district_name', 'ds_name', 'ds_lookup_census.csv')
    count_cols = [c for c in gn.columns if c.startswith('hh_') and not c.endswith('_share')]
    s = gn.groupby('pcode')[count_cols + ['households']].sum()
    for c in count_cols:
        s[c + '_share'] = s[c] / s['households']
    census = census.drop(columns=[c + '_share' for c in count_cols if c + '_share' in census], errors='ignore')
    census = census.merge(s[[c + '_share' for c in count_cols]].reset_index(), on='pcode', how='left')

census = census.drop(columns=['district_name', 'ds_name'], errors='ignore')
hf = hf.merge(census, on='pcode', how='left')

for c in CENSUS_COLS:
    if c in hf:
        hf[c] = pd.to_numeric(hf[c], errors='coerce')
if 'population_2024' in hf:
    hf['pop_density'] = hf['population_2024'] / hf['area_km2']
if {'population_2024', 'population_2012'} <= set(hf.columns):
    hf['pop_growth_2012_2024'] = hf['population_2024'] / hf['population_2012'] - 1
if 'urban_share' in hf:
    # Census 2024 Table 11.17: 82% of urban vs 12% of rural households get local-authority
    # waste collection -> estimated waste-collection coverage for each division
    hf['est_waste_collection'] = 0.82 * hf['urban_share'] + 0.12 * (1 - hf['urban_share'])
print(hf[[c for c in hf.columns if c not in ('ds_name', 'district_name')]].notna().mean().round(2))

# %% [markdown]
# ## B. Tree-cover loss (Global Forest Watch, by district)
# Uses the 30% canopy threshold (GFW default). Loss is given per district, so every DS division
# in a district gets the district's value.

# %%
gfw_files = find_files(os.path.join(RAW, 'gfw'), [r'\.csv$'])
gfw_files = [f for f in gfw_files if 'subnational' in os.path.basename(f).lower()] or gfw_files
tree = None
for f in gfw_files:
    t = pd.read_csv(f)
    t.columns = [c.lower() for c in t.columns]
    loss_cols = [c for c in t.columns if re.fullmatch(r'tc_loss_ha_(\d{4})', c)]
    if loss_cols and 'subnational2' in t.columns and 'extent_2000_ha' in t.columns:
        if 'threshold' in t.columns:
            t = t[t['threshold'] == 30]
        t['tree_loss_pct_2001_2024'] = 100 * t[loss_cols].sum(axis=1) / t['extent_2000_ha']
        recent = [c for c in loss_cols if int(c[-4:]) >= 2015]
        t['tree_loss_pct_2015_2024'] = 100 * t[recent].sum(axis=1) / t['extent_2000_ha']
        t['tree_cover_2000_pct'] = 100 * t['extent_2000_ha'] / t['area_ha']
        tree = t[['subnational2', 'tree_loss_pct_2001_2024', 'tree_loss_pct_2015_2024', 'tree_cover_2000_pct']]
        print('Used', os.path.basename(f), '| years', loss_cols[0][-4:], '-', loss_cols[-1][-4:])
        break
if tree is not None:
    tree['_d'] = tree['subnational2'].map(norm_name)
    dmap = {norm_name(d): d for d in ds_list['district_name'].unique()}
    tree['district_name'] = tree['_d'].map(dmap)
    if tree['district_name'].isna().any():
        print('Unmatched GFW districts:', tree.loc[tree['district_name'].isna(), 'subnational2'].tolist())
    hf = hf.merge(tree.drop(columns=['subnational2', '_d']), on='district_name', how='left')
    save_meta('tree_loss', 'Global Forest Watch / Hansen et al., tree cover loss 2001-2024, by district')
else:
    print('No usable GFW file found - tree loss skipped.')

# %% [markdown]
# ## C. Drain and canal density (OpenStreetMap waterways)
# Kilometres of mapped drains + canals per km², and of streams + rivers per km².
# (OSM mapping is better in cities - mention this as a data gap in the report.)

# %%
osm_dir = os.path.join(RAW, 'osm')
for z in find_files(osm_dir, [r'\.zip$']):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if 'waterways' in n:
                zf.extract(n, osm_dir)
ww_files = find_files(osm_dir, [r'waterways.*\.shp$'])
if ww_files:
    import geopandas as gpd
    ww = gpd.read_file(ww_files[0])[['fclass', 'geometry']].to_crs(SL_CRS)
    poly = gpd.read_file(os.path.join(PROC, 'ds_divisions.gpkg'))[['pcode', 'geometry']].to_crs(SL_CRS)
    pieces = gpd.overlay(ww, poly, how='intersection', keep_geom_type=True)
    pieces['km'] = pieces.length / 1000
    lens = pieces.pivot_table(index='pcode', columns='fclass', values='km', aggfunc='sum', fill_value=0)
    out = pd.DataFrame(index=lens.index)
    out['drain_canal_km'] = lens.reindex(columns=['drain', 'canal', 'ditch'], fill_value=0).sum(axis=1)
    out['stream_river_km'] = lens.reindex(columns=['stream', 'river'], fill_value=0).sum(axis=1)
    hf = hf.merge(out.reset_index(), on='pcode', how='left')
    hf[['drain_canal_km', 'stream_river_km']] = hf[['drain_canal_km', 'stream_river_km']].fillna(0)
    hf['drain_density'] = hf['drain_canal_km'] / hf['area_km2']
    hf['stream_density'] = hf['stream_river_km'] / hf['area_km2']
    save_meta('drains', 'OpenStreetMap waterways (Geofabrik extract), ODbL')
else:
    print('No OSM waterways file found - drain density skipped (first thing to cut, per proposal).')

# %%
hf.to_csv(os.path.join(PROC, 'human_factors_ds.csv'), index=False)
save_meta('census', 'Census of Population and Housing 2024 (DCS), DS level; waste collection estimated from urban share')
print('Saved human_factors_ds.csv with columns:', [c for c in hf.columns if c not in ('ds_name', 'district_name')])
