# %% [markdown]
# # 01 · Flood records (DesInventar) → flood label per DS division × month
# **Owner:** Jithmi · **Needs:** notebook 00 · **Makes:** `data/processed/floods_ds_month.csv`
#
# **Before running:** open
# https://desinventar-api.undrr.org/DesInventar/download_base.jsp?countrycode=lka
# in your browser, save the file, and upload it to `ModelX/data/raw/desinventar/`.
# (A `.zip`, `.xml`, `.csv` or `.xlsx` all work - this notebook detects the format.)

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
import zipfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

ddir = os.path.join(RAW, 'desinventar')
os.makedirs(ddir, exist_ok=True)
for z in find_files(ddir, [r'\.zip$']):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(ddir)
files = find_files(ddir, [r'\.xml$', r'\.csv$', r'\.xlsx?$'])
print(files)


# %% [markdown]
# ### Read the file
# The DesInventar XML export holds several tables. The one we need has a column `evento`
# (event type) and `fechano` (year). This reader finds it automatically.

# %%
def read_desinventar_xml(path):
    """Return {table_name: DataFrame} for every table in a DesInventar XML export."""
    tables, stack = {}, []
    for event, elem in ET.iterparse(path, events=('start', 'end')):
        if event == 'start':
            stack.append(elem.tag)
            continue
        stack.pop()
        kids = list(elem)
        # a 'row' = element whose children are all simple fields
        if len(kids) >= 3 and all(len(k) == 0 for k in kids):
            table = stack[-1] if stack else 'root'
            tables.setdefault(table, []).append({k.tag.lower(): (k.text or '').strip() for k in kids})
            elem.clear()
    return {t: pd.DataFrame(rows) for t, rows in tables.items()}


def read_any(path):
    if path.lower().endswith('.xml'):
        return read_desinventar_xml(path)
    if path.lower().endswith('.csv'):
        return {'csv': pd.read_csv(path, low_memory=False)}
    return {'excel': pd.read_excel(path)}


tables = {}
for f in files:
    for name, t in read_any(f).items():
        t.columns = [str(c).strip().lower() for c in t.columns]
        tables[name] = t
        print(f'{name:15s} {t.shape}')

# %%
# Pick the records table: the one with an event column and a year column
EVENT_COLS = ['evento', 'event', 'event type', 'eventtype']
YEAR_COLS = ['fechano', 'year', 'fecha ano']
rec_name = next(n for n, t in tables.items()
                if any(c in t.columns for c in EVENT_COLS) and any(c in t.columns for c in YEAR_COLS))
rec = tables[rec_name].copy()
first = lambda opts: next(c for c in opts if c in rec.columns)
print('Records table:', rec_name, rec.shape)
rec.head(3)

# %% [markdown]
# ### Which column is the district and which is the DS division?
# Sri Lanka has 25 districts and about 330 DS divisions. We pick the name columns whose
# number of distinct values is closest to those numbers. **Check the printout makes sense.**

# %%
# If records only carry level codes (level0, level1...), attach names from the lev0/lev1 tables
for lvl in range(3):
    code_col, name_col = f'level{lvl}', f'name{lvl}'
    if code_col in rec.columns and name_col not in rec.columns:
        for t in tables.values():
            cods = [c for c in t.columns if re.fullmatch(rf'lev{lvl}_cod', c)]
            nams = [c for c in t.columns if re.fullmatch(rf'lev{lvl}_name(_en)?', c)]
            if cods and nams:
                rec = rec.merge(t[[cods[0], nams[0]]].rename(columns={cods[0]: code_col, nams[0]: name_col}),
                                on=code_col, how='left')
                break

name_cols = [c for c in rec.columns if re.fullmatch(r'name\d|district|ds division|ds_division|division', c)]
nuniq = {c: rec[c].nunique() for c in name_cols}
print(nuniq)
DIST_COL = min(nuniq, key=lambda c: abs(nuniq[c] - 25))
DS_COL = min(nuniq, key=lambda c: abs(nuniq[c] - 330))
print('District column ->', DIST_COL, '| DS column ->', DS_COL)

# %%
ev_col, yr_col = first(EVENT_COLS), first(YEAR_COLS)
mo_col = first(['fechames', 'month'])
print(rec[ev_col].value_counts().head(20))

# %%
flood = rec[rec[ev_col].astype(str).str.contains(r'FLOOD|INUND', case=False, na=False)].copy()
flood['year'] = pd.to_numeric(flood[yr_col], errors='coerce')
flood['month'] = pd.to_numeric(flood[mo_col], errors='coerce')
n_before = len(flood)
flood = flood[flood['year'].between(1970, 2030) & flood['month'].between(1, 12)]
print(f'{len(flood)} flood records with a valid year and month ({n_before - len(flood)} dropped: no month)')

# Loss columns (Spanish names in the XML export; English in some CSV exports)
LOSS = {'deaths': ['muertos', 'deaths'], 'affected': ['afectados', 'affected'],
        'houses_destroyed': ['vivdest', 'houses destroyed'], 'houses_damaged': ['vivafec', 'houses damaged']}
for new, opts in LOSS.items():
    c = next((o for o in opts if o in flood.columns), None)
    flood[new] = pd.to_numeric(flood[c], errors='coerce') if c else np.nan

# %% [markdown]
# ### Match DS names to official pcodes
# Unmatched names are written to `data/manual/ds_lookup_desinventar.csv`.
# Fill the `pcode` column by hand (look the division up in `data/processed/ds_list.csv`) and re-run this cell.

# %%
flood = match_ds(flood, DIST_COL, DS_COL, 'ds_lookup_desinventar.csv')
flood = flood[flood['pcode'].notna()]

agg = (flood.groupby(['pcode', 'year', 'month'])
       .agg(n_records=('year', 'size'), deaths=('deaths', 'sum'), affected=('affected', 'sum'),
            houses_destroyed=('houses_destroyed', 'sum'), houses_damaged=('houses_damaged', 'sum'))
       .reset_index())
agg['flood'] = 1
agg.to_csv(os.path.join(PROC, 'floods_ds_month.csv'), index=False)

first_year, last_year = int(agg['year'].min()), int(agg['year'].max())
save_meta('flood_records', f'DesInventar Sri Lanka, {first_year}-{last_year}')
save_meta('flood_last_year', last_year)
print(f'Saved {len(agg)} flooded division-months, {first_year}-{last_year}')

# %%
# IMPORTANT CHECK: what is the last year with records? (decides the train/test split)
per_year = agg.groupby('year')['flood'].sum()
per_year.plot(kind='bar', figsize=(12, 3), title='Flooded DS-division-months per year (DesInventar)')
plt.tight_layout(); plt.show()
print(per_year.tail(10))
