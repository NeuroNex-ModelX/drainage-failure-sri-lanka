# Drainage Failure Map for Sri Lanka
**Team NeuroNex · ModelX 2.0 · Domain 02: Climate, environment and disaster risk**

> *Where does Sri Lanka flood more than its rain and land height explain, why, and is it getting worse?*

Every flood in Sri Lanka is logged simply as a flood, so nobody can tell floods caused by extreme rain
from floods caused by **failing drainage** (blocked drains, filled retention land, paved ground, tree loss).
This system separates the two for every **DS division (~330)**:

1. **Model A** learns how often each division *should* flood from **natural factors only** (CHIRPS rainfall,
   land height, coast, season).
2. **Drainage Failure Score (DFS) = (actual floods − expected floods) ÷ expected floods.**
   High DFS means the division floods far more than nature explains.
3. **Model B** (Random Forest + SHAP) links each division's DFS to **human factors**: estimated waste
   collection, population density and growth, service coverage, tree-cover loss and drain density.
   K-means on SHAP values groups divisions into **failure profiles**.
4. **Trend:** floods per 100 heavy-rain days, 2005–2014 vs 2015–latest.
5. **Dashboard** (Streamlit): failure map · division profile · priority list (CSV) · **live 3-day rain watch**.

**Live dashboard:** `<add Streamlit Cloud link>` · **Demo video:** `<add link>`

---

## Results (fill in from `outputs/`)
| Check (proposal §9) | Result | File |
|---|---|---|
| Model A vs baselines (test years) | PR-AUC __ vs rainfall-only __ / last-year __ | `outputs/model_A_metrics.csv` |
| Ranking stability (leave one year out) | median Spearman __ | `outputs/dfs_stability.csv` |
| Known problem areas (Kolonnawa, Kotte …) | percentiles __ | `outputs/known_areas_check.csv` |
| Model B cross-validated R² (district-grouped) | __ vs average-only __ | `outputs/model_B_metrics.csv` |
| Top SHAP factors stable in resamples | __ | `outputs/shap_stability.csv` |
| 2024 floods / Ditwah: DFS effect after rain | coef __, p __ | `outputs/eval_recent_events.csv` |
| Divisions with worsening trend | __ | `outputs/trend_ds.csv` |

---

## Data (all public, aggregated, no personal data)
| Layer | Source | Period | Licence |
|---|---|---|---|
| Flood records (label) | DesInventar Sri Lanka (UNDRR / DMC) | 1995–last year available | Public; credited |
| Recent floods (test) | DMC situation reports, IFRC DREF MDRLK020, Ditwah R package, UNOSAT, Copernicus EMSR851 | 2024–2025 | Public / Copernicus open |
| Rainfall | CHIRPS v2.0 daily 0.05° (UCSB CHC); GPM IMERG V07 cross-check | 1995–2025 | Public domain / NASA open |
| Live forecast | Open-Meteo forecast API (3 days) | refreshed on every visit | CC BY 4.0 |
| Land height | Open-Meteo elevation API (Copernicus DEM GLO-90) | static | Free, attribution |
| Population, households, urban share, services | Census of Population and Housing 2024 (DCS) | 2024 (2012 for growth) | Public |
| Tree-cover loss | Global Forest Watch / Hansen et al., by district | 2001–2024 | CC BY 4.0 |
| Drains and canals | OpenStreetMap (Geofabrik) | download date | ODbL |
| Boundaries | Survey Dept of Sri Lanka via OCHA HDX (cod-ab-lka) | 2022 | CC BY-IGO |

**Temporality:** history (1995–2023) trains the models; 2024–2025 events are held out for testing;
the rain watch is live. Each layer's date is shown in the dashboard sidebar.

**Known gaps:**
* Small urban floods are under-recorded in DesInventar.
* Waste disposal is only published nationally by sector, so collection is *estimated* from urban share
  (82% urban vs 12% rural households, Census 2024 Table 11.17).
* Tree loss is by district.
* OSM drain mapping is uneven.
* Results are factors **linked to** failure, not proof of cause.

---

## How to reproduce
**Google Colab (recommended):** see **[GUIDE.md](GUIDE.md)** for the full step-by-step.
1. Put this folder in Google Drive as `My Drive/ModelX/`.
2. Download the raw files listed in GUIDE §1 into `data/raw/`.
3. Run `notebooks/00` → `09` in order. Each notebook says what it needs and what it makes.

**Dashboard locally:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
The dashboard only needs `outputs/`, `data/processed/ds_list.csv` and
`data/processed/ds_divisions_simplified.geojson`, which are included in the repo.

## Repository layout
```
notebooks/       00–09 Colab notebooks (run in order)
notebooks_py/    same notebooks as plain .py (easy to read / diff)
src/common.py    shared helpers: paths, name matching to official pcodes, metadata
app/app.py       Streamlit dashboard (4 pages)
data/manual/     hand-entered tables (census, recent events, name lookups)
data/processed/  cleaned tables made by the notebooks
outputs/         model results read by the dashboard
docs/            report outline
tools/           py_to_ipynb.py (rebuilds notebooks/ from notebooks_py/)
```

## Team
Jithmi Thilakarathna (data collection) · Prameshka Fernando (data preparation) ·
Nadeesha Wickramasooriya (modelling) · Sayuni Ranasinghe (dashboard). Informatics Institute of Technology (IIT), Colombo.

*AI assistance (Claude) was used to draft code and documentation; the team reviewed, ran and is
responsible for all results.*
