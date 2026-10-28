# %% [markdown]
# # 03 · Land height per DS division (Open-Meteo elevation API, Copernicus DEM GLO-90)
# **Owner:** Prameshka · **Needs:** notebook 00 · **Makes:** `data/processed/elevation_ds.csv`
#
# Samples points every ~1 km inside each division and asks the API for their height
# (100 points per request, ~600 calls in total, ~10–15 min). Progress is saved after every
# division, so if Colab disconnects just re-run: it continues where it stopped.

# %%
# --- SETUP ---
import os, sys, time
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MODELX_BASE'] = '/content/drive/MyDrive/ModelX'
sys.path.append(os.path.join(os.environ.get('MODELX_BASE', '..'), 'src'))
from common import *
import requests
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

STEP_DEG = 0.01      # ~1.1 km grid
API = 'https://api.open-meteo.com/v1/elevation'
ds_poly = gpd.read_file(os.path.join(PROC, 'ds_divisions.gpkg'))
partial = os.path.join(PROC, 'elevation_points_partial.csv')
done = pd.read_csv(partial) if os.path.exists(partial) else pd.DataFrame(columns=['pcode', 'lat', 'lon', 'elev'])
done_codes = set(done['pcode'])
print(f'{len(done_codes)} of {len(ds_poly)} divisions already done')


# %%
def get_heights(points, tries=5):
    """points: list of (lat, lon) -> list of heights (m). Retries politely if rate-limited."""
    params = {'latitude': ','.join(f'{a:.4f}' for a, _ in points),
              'longitude': ','.join(f'{b:.4f}' for _, b in points)}
    for k in range(tries):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()['elevation']
        time.sleep(5 * (k + 1))   # 429 = too many requests -> wait and retry
    raise RuntimeError(f'API failed: {r.status_code} {r.text[:200]}')


for _, row in ds_poly.iterrows():
    if row['pcode'] in done_codes:
        continue
    geom = row.geometry
    pg = prep(geom)
    minx, miny, maxx, maxy = geom.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, STEP_DEG), np.arange(miny, maxy, STEP_DEG))
    pts = [(y, x) for x, y in zip(xs.ravel(), ys.ravel()) if pg.contains(Point(x, y))]
    if len(pts) < 5:   # very small division: use a few points inside it
        rp = geom.representative_point()
        pts = pts + [(rp.y, rp.x)]
    heights = []
    for i in range(0, len(pts), 100):
        heights += get_heights(pts[i:i + 100])
        time.sleep(0.15)
    chunk = pd.DataFrame({'pcode': row['pcode'], 'lat': [p[0] for p in pts],
                          'lon': [p[1] for p in pts], 'elev': heights})
    chunk.to_csv(partial, mode='a', header=not os.path.exists(partial), index=False)
    print(row['pcode'], row['ds_name'], len(pts), 'points', flush=True)

# %%
pts = pd.read_csv(partial)
pts.loc[pts['elev'] < -50, 'elev'] = np.nan   # sea / no-data
elev = pts.groupby('pcode')['elev'].agg(
    elev_mean='mean', elev_min='min', elev_median='median',
    pct_below_10m=lambda h: 100 * (h < 10).mean(),
    pct_below_5m=lambda h: 100 * (h < 5).mean(),
    n_points='size').reset_index()

ds_list = load_ds_list()[['pcode', 'is_coastal']]
elev = elev.merge(ds_list, on='pcode', how='right')
elev.to_csv(os.path.join(PROC, 'elevation_ds.csv'), index=False)
save_meta('elevation', 'Open-Meteo elevation API (Copernicus DEM GLO-90), ~1 km sample grid')
print(elev.describe().round(1))
