# %% [markdown]
# # 08 · Trend: is drainage getting worse?
# **Owner:** Nadeesha · **Needs:** 05 · **Makes:** `outputs/trend_ds.csv`, `outputs/trend_national.csv`
#
# Proposal Step 5: for each division, floods per heavy-rain day in **2005–2014** vs **2015–(last year of
# flood records)**. If floods now happen more often for the same amount of heavy rain, drainage is degrading.
# Significance: exact test comparing the two rates (binomial test conditional on the total number of floods).

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
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt

t = pd.read_csv(os.path.join(PROC, 'analysis_table.csv'))
t = t[t['label_known'] == 1]
LAST = int(t['year'].max())
P1, P2 = (2005, 2014), (2015, LAST)
HEAVY = 'days_over_50'       # a 'heavy-rain day' = more than 50 mm in a day
print('Periods:', P1, 'vs', P2)

# %%
def period_sums(a, b):
    s = t[t['year'].between(a, b)].groupby('pcode').agg(floods=('flood', 'sum'), heavy_days=(HEAVY, 'sum'))
    return s


s1, s2 = period_sums(*P1), period_sums(*P2)
tr = s1.join(s2, lsuffix='_p1', rsuffix='_p2', how='outer').fillna(0)
tr['rate_p1'] = 100 * tr['floods_p1'] / tr['heavy_days_p1'].replace(0, np.nan)   # floods per 100 heavy-rain days
tr['rate_p2'] = 100 * tr['floods_p2'] / tr['heavy_days_p2'].replace(0, np.nan)
tr['change'] = tr['rate_p2'] - tr['rate_p1']


def test_row(r):
    k1, k2, n1, n2 = r['floods_p1'], r['floods_p2'], r['heavy_days_p1'], r['heavy_days_p2']
    if n1 == 0 or n2 == 0 or (k1 + k2) < 3:
        return np.nan
    return binomtest(int(k2), int(k1 + k2), n2 / (n1 + n2)).pvalue


tr['p_value'] = tr.apply(test_row, axis=1)
tr['trend'] = np.select(
    [tr['p_value'].isna(), (tr['p_value'] < 0.10) & (tr['change'] > 0), (tr['p_value'] < 0.10) & (tr['change'] < 0)],
    ['Not enough data', 'Worsening', 'Improving'], default='No clear change')
tr = tr.reset_index().merge(load_ds_list()[['pcode', 'ds_name', 'district_name']], on='pcode')
tr.to_csv(os.path.join(OUT, 'trend_ds.csv'), index=False)
save_meta('trend_periods', f'{P1[0]}-{P1[1]} vs {P2[0]}-{P2[1]}')
print(tr['trend'].value_counts())

# %%
# Island-wide: did the rate go up across divisions? (paired Wilcoxon test)
ok = tr.dropna(subset=['rate_p1', 'rate_p2'])
if len(ok) > 10:
    w = wilcoxon(ok['rate_p2'], ok['rate_p1'])
    print(f"Median change: {ok['change'].median():.2f} floods per 100 heavy-rain days; Wilcoxon p = {w.pvalue:.3g}")

# National series: floods per 100 heavy-rain days, by year (figure for the report)
nat = t.groupby('year').agg(floods=('flood', 'sum'), heavy_days=(HEAVY, 'sum'))
nat['floods_per_100_heavy_days'] = 100 * nat['floods'] / nat['heavy_days']
nat.reset_index().to_csv(os.path.join(OUT, 'trend_national.csv'), index=False)
nat['floods_per_100_heavy_days'].plot(marker='o', figsize=(9, 3),
                                      title='Island-wide: flooded division-months per 100 heavy-rain days')
plt.tight_layout(); plt.show()

tr.sort_values('change', ascending=False).head(10)[['ds_name', 'district_name', 'rate_p1', 'rate_p2', 'p_value', 'trend']]
