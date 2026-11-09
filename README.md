# Drainage Failure Map for Sri Lanka
**Team NeuroNex · ModelX 2.0 · Domain 02: Climate, environment and disaster risk**

> *Where does Sri Lanka flood more than its rain and land height explain, why, and is it getting worse?*

**Live dashboard:** https://drainage-failure-sri-lanka.streamlit.app/ 

**Demo video:** https://drive.google.com/file/d/1QrSNXv9LdZNDLmNcYBG9JEh7aYLMS5HQ/view?usp=sharing

Every flood in Sri Lanka is logged simply as a flood, so nobody can tell floods caused by extreme rain
from floods caused by **failing drainage** (blocked drains, filled retention land, paved ground, tree loss).
This system separates the two for every **DS division (339)**:

1. **Model A** learns how often each division *should* flood from **natural factors only**
   (CHIRPS daily rainfall, land height, coast, season).
2. **Drainage Failure Score (DFS) = (actual floods − expected floods) ÷ expected floods.**
   A high DFS means the division floods far more often than its rain and land explain.
3. **Model B** (Random Forest + SHAP) looks for **human factors linked to** each division's DFS:
   population density, service coverage, tree-cover loss and drain/canal density.
   K-means on the SHAP values groups divisions into **failure profiles**.
4. **Trend:** floods per 100 heavy-rain days, 1995–2004 vs 2005–2014, per division, district and island.
5. **Recent-event check:** were districts with higher DFS hit harder by Cyclone Ditwah (2025)?
6. **Dashboard** (Streamlit): failure map · division profile · priority list (CSV download) ·
   **live 3-day rain watch** (heavy rain forecast × high DFS = early warning).

---

## Results
| Check | Result | File |
|---|---|---|
| Model A vs baselines (test years 2010–2014, flood rate 2.4%) | Random forest PR-AUC **0.117**, ROC-AUC 0.851, vs rainfall-only 0.092 / 0.836 and same-month-last-year 0.033 / 0.544 | `outputs/model_A_metrics.csv` |
| Ranking stability (leave one year out, 20 runs) | median Spearman **0.995** (min 0.969); median top-20 overlap 90% | `outputs/dfs_stability.csv` |
| Known problem areas | Ja-Ela 95th, Wattala 82nd, Kotte 76th, Kolonnawa 73rd, Kaduwela 60th percentile: all above the median | `outputs/known_areas_check.csv` |
| Divisions ranked | 333 of 339 (6 low-confidence); **136** flood more than expected, **30** at least twice as often | `outputs/dfs_ds.csv` |
| Model B, cross-validated R² (whole districts held out) | **−0.28** vs −0.16 for predicting the average: human factors do **not** predict DFS in unseen districts | `outputs/model_B_metrics.csv` |
| Most stable SHAP factors (50 resamples) | tree-cover loss (top 3 in 100%), drain/canal density (90%) | `outputs/shap_stability.csv` |
| Trend, island-wide | **4.3 → 8.0** flooded division-months per 100 heavy-rain days (exact binomial test, p < 0.001) | `outputs/trend_national.csv` |
| Trend by district | **17 of 25 worsening** (e.g. Colombo 5.1 → 12.9, Gampaha 11.9 → 21.6), 1 improving (Polonnaruwa), 7 no clear change | `outputs/trend_district.csv` |
| Trend by division | 38 worsening, 2 improving, 180 no clear change, 119 too few floods to test | `outputs/trend_ds.csv` |
| Cyclone Ditwah 2025 (11 districts) | higher DFS linked to more people affected per 1,000 (Spearman 0.49), but **not significant** (p = 0.33) | `outputs/eval_recent_events.csv` |

**How to read these results:** Model A, the DFS ranking and its stability are the solid core.
The scores stay almost identical whichever year is removed, and the known flood-prone Colombo/Gampaha
divisions all rank in the upper half. Model B's "linked factors" are **exploratory hints only**, because
they do not generalise to new districts. The trend and Ditwah results point the same way (more flooding
per heavy-rain day, and harder hits where DFS is high), but both carry the caveats listed below.

---

## Data (all public, aggregated, no personal data)
| Layer | Source | Period | Licence |
|---|---|---|---|
| Flood records (label) | DesInventar Sri Lanka (UNDRR / DMC) via OCHA HDX `di_report-srilanka`; records placed in DS divisions by DesInventar's own division code and GN location, using the official GN list | 1995–2014 used (6,350 of 6,899 flood records placed) | Public; credited |
| Recent-event impact (test) | DMC figures via IOM Situation Report 6 (8 Dec 2025), UN Sri Lanka Flash Update 5, Sunday Times (7 Dec 2025) | Cyclone Ditwah, Nov–Dec 2025 | Public |
| Rainfall | CHIRPS v2.0 daily 0.05° (UCSB CHC), read via Google Earth Engine; GPM IMERG V07 cross-check for Ditwah | 1995–2025 | Public domain / NASA open |
| Live forecast | Open-Meteo forecast API (3 days) | refreshed on every visit | CC BY 4.0 |
| Land height | Open-Meteo elevation API (Copernicus DEM GLO-90) | static | Free, attribution |
| Population, households, services | Census of Population and Housing 2024 (DCS): GN population table, tables A13, A14, A16 | 2024 | Public |
| Tree-cover loss | Global Forest Watch / Hansen et al., by DS division | 2001–2025 | CC BY 4.0 |
| Drains and canals | OpenStreetMap (Geofabrik extract) | download date | ODbL |
| Boundaries | Survey Department of Sri Lanka via OCHA HDX (cod-ab-lka): DS (adm3) and GN (adm4) | 2022 | CC BY-IGO |

**Temporality:** 1995–2014 trains and tests Model A (test years 2010–2014). Cyclone Ditwah 2025 is a
held-out recent check, and the rain watch is live. The dashboard sidebar shows each layer's date.

## Known limitations
* **Flood labels end in 2014.** DesInventar recording for Sri Lanka is sparse after 2014. Later floods exist
  only as separate situation reports, so recent years are used for event testing, not training.
* **Recording changes over time.** DesInventar logged far more events in 2006–2010 than in other years,
  so part of the worsening trend may reflect better recording, not only worse drainage.
* **Recent-event test is small.** Complete district figures were not publicly available for the June and
  October 2024 floods (dropped) or for all of Ditwah (11 of 25 districts used).
* **DFS also captures river flooding.** Some top-ranked divisions (e.g. Balangoda, Ratnapura) flood from
  rivers carrying upstream rain, which local rainfall does not capture.
* **Model B is weak** (negative cross-validated R²). Its factors are *linked to* DFS, not proof of cause.
  For example, "low tree-cover loss" mostly marks urban areas with little forest left to lose.
* About 8% of flood records could not be placed in a single DS division (district-wide or unrecognised
  place names).
* OpenStreetMap drain mapping is uneven across the island.

---

## How to reproduce
**Google Colab (recommended):** see **[GUIDE.md](GUIDE.md)** for the full step-by-step.
1. Put this repository in Google Drive as `My Drive/ModelX/`.
2. Download the raw files listed in GUIDE §1 into `data/raw/`.
3. Run `notebooks/00` → `09` in order. Each notebook says what it needs and what it makes.

**Dashboard locally:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
The dashboard only needs `outputs/`, `data/processed/ds_list.csv` and
`data/processed/ds_divisions_simplified.geojson`, which are all in the repo.

## Repository layout
```
notebooks/        00–09 Colab notebooks, run in order (saved with their outputs)
notebooks_py/     plain-text .py copies of the notebooks (easy to read / diff)
src/common.py     shared helpers: paths, name matching to official pcodes, metadata
app/app.py        Streamlit dashboard (4 pages)
data/manual/      hand-entered tables (recent-event figures, name lookups)
data/processed/   cleaned tables made by the notebooks
outputs/          model results read by the dashboard
docs/             report outline
tools/            py_to_ipynb.py (rebuilds notebooks/ from notebooks_py/)
```

## Team
Jithmi Thilakarathna · Prameshka Fernando  ·
Nadeesha Wickramasooriya  · Sayuni Ranasinghe .
Informatics Institute of Technology (IIT), Colombo.

*AI assistance (Claude) was used to draft code and documentation. The team reviewed and ran it, and is
responsible for all results.*
