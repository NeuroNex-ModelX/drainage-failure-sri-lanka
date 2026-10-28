# %% [markdown]
# # 00 · Setup and DS-division boundaries
# **Owner:** Jithmi · **Makes:** `data/processed/ds_divisions.gpkg`, `ds_list.csv`, `ds_divisions_simplified.geojson`
#
# The DS-division list made here is the "spine" every other notebook joins to (by `pcode`).
#
# **Before running:**
# 1. Go to https://data.humdata.org/dataset/cod-ab-lka and download the **shapefile zip** (SHP).
# 2. Upload the zip into the Drive folder `ModelX/data/raw/boundaries/` (no need to unzip).

# %%
# --- SETUP: run this cell first in every session ---
import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *
print('Project folder:', BASE)

# %%
import zipfile, glob
import geopandas as gpd

bdir = os.path.join(RAW, 'boundaries')
os.makedirs(bdir, exist_ok=True)
for z in glob.glob(os.path.join(bdir, '*.zip')):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(bdir)
    print('Unzipped', os.path.basename(z))

shps = find_files(bdir, [r'\.shp$'])
print('\n'.join(os.path.basename(s) for s in shps))

# %% [markdown]
# HDX levels for Sri Lanka: **adm1 = province, adm2 = district, adm3 = DS division, adm4 = GN division.**

# %%
def pick(level):
    c = [s for s in shps if re.search(rf'adm{level}', os.path.basename(s), re.I)]
    c = sorted(c, key=lambda s: 'admbnda' not in os.path.basename(s).lower())  # polygons first
    if not c:
        raise FileNotFoundError(f'No adm{level} shapefile found in {bdir}')
    return c[0]

ds = gpd.read_file(pick(3)).to_crs('EPSG:4326')
print(ds.columns.tolist())
ds.head()

# %%
# Find the name / code columns (HDX uses ADM3_EN, ADM3_PCODE, ADM2_EN, ADM2_PCODE)
def col(df, pattern):
    hits = [c for c in df.columns if re.fullmatch(pattern, c, re.I)]
    if not hits:
        raise KeyError(f'No column like {pattern}: {df.columns.tolist()}')
    return hits[0]

ds = ds.rename(columns={
    col(ds, r'ADM3_PCODE'): 'pcode', col(ds, r'ADM3_EN'): 'ds_name',
    col(ds, r'ADM2_PCODE'): 'district_pcode', col(ds, r'ADM2_EN'): 'district_name',
})
ds = ds[['pcode', 'ds_name', 'district_pcode', 'district_name', 'geometry']]
# Some layers split a division into several polygons -> merge them into one row
ds = ds.dissolve(by=['pcode', 'ds_name', 'district_pcode', 'district_name'], as_index=False)
print(len(ds), 'DS divisions')   # expect about 330

# %%
# Area, a point inside each division, and whether it touches the coast
proj = ds.to_crs(SL_CRS)
ds['area_km2'] = proj.area / 1e6
pts = ds.representative_point()
ds['lat'], ds['lon'] = pts.y, pts.x

country_edge = proj.union_all().boundary if hasattr(proj, 'union_all') else proj.unary_union.boundary
ds['is_coastal'] = proj.geometry.boundary.buffer(200).intersects(country_edge).astype(int)
print('Coastal divisions:', ds['is_coastal'].sum())

# %%
gpkg = os.path.join(PROC, 'ds_divisions.gpkg')
ds.to_file(gpkg, driver='GPKG')
ds.drop(columns='geometry').to_csv(os.path.join(PROC, 'ds_list.csv'), index=False)

# Simplified copy for the dashboard map (small enough for GitHub / Streamlit)
simple = ds[['pcode', 'ds_name', 'district_name', 'geometry']].copy()
simple['geometry'] = simple.to_crs(SL_CRS).simplify(150).to_crs('EPSG:4326')
simple.to_file(os.path.join(PROC, 'ds_divisions_simplified.geojson'), driver='GeoJSON')

save_meta('boundaries', 'Survey Dept of Sri Lanka via OCHA/HDX (cod-ab-lka), CC BY-IGO')
print('Saved:', gpkg)

# %%
# Quick look (screenshot this for the report)
ax = ds.plot(column='is_coastal', cmap='Blues', edgecolor='grey', linewidth=0.2, figsize=(6, 8))
ax.set_title('DS divisions of Sri Lanka (dark = coastal)'); ax.set_axis_off()
