# %% [markdown]
# # 02 · Rainfall (CHIRPS v2.0 daily) → rainfall features per DS division × month
# **Owner:** Prameshka · **Needs:** notebook 00 · **Makes:** `rain_daily.csv.gz`, `rain_monthly_features.csv`
#
# Downloads one global yearly file at a time to Colab's own disk (fast, not your Drive),
# crops Sri Lanka, saves the small crop to Drive, deletes the big file.
# **Runtime:** roughly 1–2 min per year, so ~30–60 min for 1995–2025. Re-running skips years already done.
# Keep the Colab tab open (Runtime ▸ Run all, then leave it).

# %%
# --- SETUP ---
import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'netCDF4', 'h5netcdf'])
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *
import xarray as xr

START_YEAR, END_YEAR = 1995, 2025       # proposal: history from 1995; 2024-2025 used for testing
CHIRPS_URL = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.{y}.days_p05.nc'
crop_dir = os.path.join(PROC, 'chirps')
tmp_dir = '/content/tmp' if IN_COLAB else os.path.join(BASE, 'tmp')
os.makedirs(crop_dir, exist_ok=True); os.makedirs(tmp_dir, exist_ok=True)


def crop_lka(ds):
    lat = ds['latitude'].values
    lat_slice = slice(LAT_MIN, LAT_MAX) if lat[0] < lat[-1] else slice(LAT_MAX, LAT_MIN)
    return ds.sel(latitude=lat_slice, longitude=slice(LON_MIN, LON_MAX))


# %% [markdown]
# ### Step 1 – download and crop each year

# %%
for y in range(START_YEAR, END_YEAR + 1):
    out = os.path.join(crop_dir, f'chirps_lka_{y}.nc')
    if os.path.exists(out):
        print(y, 'already done'); continue
    tmp = os.path.join(tmp_dir, f'chirps_{y}.nc')
    print(y, 'downloading...', flush=True)
    r = subprocess.run(['wget', '-q', '-c', CHIRPS_URL.format(y=y), '-O', tmp])
    if r.returncode != 0 or os.path.getsize(tmp) < 1e6:
        print(f'   failed for {y} - check the URL in a browser, then re-run this cell'); continue
    with xr.open_dataset(tmp) as g:
        crop_lka(g)[['precip']].load().to_netcdf(out)
    os.remove(tmp)
    print('   saved', os.path.basename(out))

save_meta('rainfall', f'CHIRPS v2.0 daily 0.05 deg, {START_YEAR}-{END_YEAR} (UCSB CHC)')

# %% [markdown]
# ### Step 2 – which CHIRPS grid cells fall inside each DS division?
# Each ~5 km cell is assigned to the division containing its centre. Small divisions with no
# cell centre inside get the nearest cell.

# %%
import geopandas as gpd

ds_poly = gpd.read_file(os.path.join(PROC, 'ds_divisions.gpkg'))
sample = xr.open_dataset(os.path.join(crop_dir, f'chirps_lka_{START_YEAR}.nc'))
lats, lons = sample['latitude'].values, sample['longitude'].values
LON, LAT = np.meshgrid(lons, lats)  # shape (ny, nx); flat index = row*nx + col
cells = gpd.GeoDataFrame({'cell': np.arange(LAT.size)},
                         geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs='EPSG:4326')
inside = gpd.sjoin(cells, ds_poly[['pcode', 'geometry']], predicate='within')[['cell', 'pcode']]

missing = sorted(set(ds_poly['pcode']) - set(inside['pcode']))
extra = []
for p in missing:   # nearest cell for tiny divisions
    pt = ds_poly.loc[ds_poly['pcode'] == p].representative_point().iloc[0]
    i = int(np.argmin((LAT.ravel() - pt.y) ** 2 + (LON.ravel() - pt.x) ** 2))
    extra.append({'cell': i, 'pcode': p})
cell_map = pd.concat([inside, pd.DataFrame(extra)], ignore_index=True)
cell_map.to_csv(os.path.join(PROC, 'chirps_cell_map.csv'), index=False)
print(f'{inside.pcode.nunique()} divisions have cells inside; {len(missing)} use the nearest cell')

# %% [markdown]
# ### Step 3 – daily rainfall per DS division (average of its cells)

# %%
groups = cell_map.groupby('pcode')['cell'].apply(np.array).to_dict()
pcodes = sorted(groups)
frames = []
for y in range(START_YEAR, END_YEAR + 1):
    f = os.path.join(crop_dir, f'chirps_lka_{y}.nc')
    if not os.path.exists(f):
        print('missing year', y); continue
    with xr.open_dataset(f) as d:
        arr = d['precip'].transpose('time', 'latitude', 'longitude').values.reshape(d.sizes['time'], -1)  # (days, cells)
        arr = np.where(arr < 0, np.nan, arr)                      # -9999 = no data
        dates = pd.to_datetime(d['time'].values)
    with np.errstate(all='ignore'):
        mat = np.column_stack([np.nanmean(arr[:, groups[p]], axis=1) for p in pcodes])
    frames.append(pd.DataFrame(mat, index=dates, columns=pcodes))
rain_wide = pd.concat(frames)
rain_daily = rain_wide.stack().rename('rain_mm').reset_index()
rain_daily.columns = ['date', 'pcode', 'rain_mm']
rain_daily.to_csv(os.path.join(PROC, 'rain_daily.csv.gz'), index=False)
print(rain_daily.shape, rain_daily['date'].min().date(), '->', rain_daily['date'].max().date())

# %% [markdown]
# ### Step 4 – monthly features (one row per division × month)
# monthly total, wettest day, days > 50 mm, days > 100 mm, previous month's total.

# %%
r = rain_daily.copy()
r['year'], r['month'] = r['date'].dt.year, r['date'].dt.month
r['over_50'] = (r['rain_mm'] > 50).astype(int)
r['over_100'] = (r['rain_mm'] > 100).astype(int)
feat = (r.groupby(['pcode', 'year', 'month'])
        .agg(rain_total=('rain_mm', 'sum'), rain_max_day=('rain_mm', 'max'),
             days_over_50=('over_50', 'sum'), days_over_100=('over_100', 'sum'))
        .reset_index().sort_values(['pcode', 'year', 'month']))
feat['rain_prev_month'] = feat.groupby('pcode')['rain_total'].shift(1)
feat.to_csv(os.path.join(PROC, 'rain_monthly_features.csv'), index=False)
print(feat.shape)
feat.describe().round(1)

# %%
# Sanity check (good figure for the report): Colombo-area divisions around Cyclone Ditwah
ditwah = rain_daily[(rain_daily['date'] >= '2025-11-15') & (rain_daily['date'] <= '2025-12-10')]
top = ditwah.groupby('pcode')['rain_mm'].sum().sort_values(ascending=False).head(10)
print('Wettest divisions, 15 Nov - 10 Dec 2025 (mm):'); print(top.round(0))

# %% [markdown]
# ### Optional – cross-check with GPM IMERG for a flood event
# If you downloaded IMERG daily files (Earthdata Search) for Ditwah into `data/raw/imerg/`,
# this compares IMERG and CHIRPS event totals per division. Skip if you have no IMERG files.

# %%
imerg_files = find_files(os.path.join(RAW, 'imerg'), [r'\.nc4?$'])
if imerg_files:
    im = xr.open_mfdataset(imerg_files, combine='by_coords')['precipitation']
    im = im.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX)).sum('time').load()
    tot = []
    for _, row in ds_poly.iterrows():
        pt = row.geometry.representative_point()
        tot.append(float(im.sel(lat=pt.y, lon=pt.x, method='nearest')))
    comp = pd.DataFrame({'pcode': ds_poly['pcode'], 'imerg_mm': tot})
    span = (rain_daily['date'] >= '2025-11-15') & (rain_daily['date'] <= '2025-12-15')
    comp = comp.merge(rain_daily[span].groupby('pcode')['rain_mm'].sum().rename('chirps_mm'), on='pcode')
    print('Correlation CHIRPS vs IMERG (event totals):', round(comp[['imerg_mm', 'chirps_mm']].corr().iloc[0, 1], 2))
    comp.to_csv(os.path.join(OUT, 'imerg_vs_chirps_ditwah.csv'), index=False)
else:
    print('No IMERG files found - skipped (optional).')
