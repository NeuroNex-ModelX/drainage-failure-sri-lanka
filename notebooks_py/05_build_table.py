# %% [markdown]
# # 05 · Build the analysis table (one row per DS division × month)
# **Owner:** Prameshka · **Needs:** 01, 02, 03 · **Makes:** `data/processed/analysis_table.csv`
#
# Proposal, Step 1: rainfall features + land-height features + coastal flag + label
# (flood / flash flood recorded that month: yes/no).

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

rain = pd.read_csv(os.path.join(PROC, 'rain_monthly_features.csv'))
elev = pd.read_csv(os.path.join(PROC, 'elevation_ds.csv'))
floods = pd.read_csv(os.path.join(PROC, 'floods_ds_month.csv'))
meta = load_meta()
LAST_FLOOD_YEAR = int(meta.get('flood_last_year', floods['year'].max()))
print('Flood records end in', LAST_FLOOD_YEAR)

# %%
t = rain.merge(elev, on='pcode', how='left')
t = t.merge(floods[['pcode', 'year', 'month', 'flood', 'n_records', 'affected']],
            on=['pcode', 'year', 'month'], how='left')
t['flood'] = t['flood'].fillna(0).astype(int)
# NB: 'no record' is stored as 0. The proposal treats it as 'unknown, probably no flood';
# small urban floods are under-recorded, which we report as a limitation.

# Season as a natural factor (monsoons): month on a circle
t['month_sin'] = np.sin(2 * np.pi * t['month'] / 12)
t['month_cos'] = np.cos(2 * np.pi * t['month'] / 12)

# Rows used for modelling = years covered by the flood records.
# Later years (e.g. 2024-2025) stay in the file with the label unknown, for the rain watch / tests.
t['label_known'] = (t['year'] <= LAST_FLOOD_YEAR).astype(int)
t = t[t['year'] >= 1995].dropna(subset=['rain_prev_month'])
t.to_csv(os.path.join(PROC, 'analysis_table.csv'), index=False)

known = t[t['label_known'] == 1]
print(f"{len(t):,} rows | {t.pcode.nunique()} divisions | years {t.year.min()}-{t.year.max()}")
print(f"Flood rate in labelled rows: {known['flood'].mean():.2%} ({known['flood'].sum():,} flooded division-months)")

# %%
# Coverage check: every division should have rainfall and elevation
missing = t[['elev_mean', 'rain_total']].isna().mean()
print('Missing share:\n', missing.round(3))
