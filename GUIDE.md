# Team guide: building the system in Google Colab (28 Oct → 1 Nov)

This is the step-by-step plan for **NeuroNex**. It follows the proposal exactly: same data, same two-stage
method (Model A → Drainage Failure Score → Model B + SHAP → trend), same dashboard pages.

---

## 0. One-time setup (tonight, ~20 min, one person then everyone)

### 0.1 Shared Google Drive folder (this is where everything lives)
1. **One person** (e.g. Prameshka) unzips this project and uploads the whole `ModelX` folder to
   **My Drive**. The path must be `My Drive/ModelX/` (exactly this name).
2. Right-click the folder → **Share** → add the other three IIT emails as **Editor**.
3. **Every other member:** open *Shared with me* → right-click `ModelX` → **Organize → Add shortcut → My Drive**.
   Now `/content/drive/MyDrive/ModelX` works for everyone in Colab.

### 0.2 Opening and sharing the notebooks
* In Drive, open `ModelX/notebooks/`, double-click a notebook → **Open with → Google Colaboratory**.
  (First time: *Open with → Connect more apps → Colaboratory*.)
* Because the folder is shared, **everyone can open and edit the same notebooks**. You can also press
  **Share** inside Colab like a Google Doc.
* Two things to know:
  * Each person runs code on **their own Colab machine**. If two people edit the same notebook at the
    same time the text syncs, but outputs can clash. **Rule: one owner per notebook** (table below).
  * Every notebook's first cell mounts Drive. Click **Allow** when asked. Run it first every session.
* Results are saved to the shared Drive folder, so the next notebook (even on a teammate's
  account) can read them.

### 0.3 Owners (from proposal section 8)
| Notebook | Owner | Needs first |
|---|---|---|
| 00 boundaries | Jithmi | HDX shapefile |
| 01 floods (DesInventar) | Jithmi | 00 |
| 02 rainfall (CHIRPS) | Prameshka | 00 |
| 03 elevation | Prameshka | 00 |
| 04 human factors | Jithmi | 00 + census/GFW/OSM files |
| 05 build table | Prameshka | 01, 02, 03 |
| 06 Model A + DFS | Nadeesha | 05 |
| 07 Model B + SHAP | Nadeesha | 04, 06 |
| 08 trend | Nadeesha | 05 |
| 09 test on 2024/Ditwah | All (typing) | 02, 04, 06 |
| `app/app.py` dashboard | Sayuni | 06, 07, 08 |

---

## 1. Downloads (do these first, in parallel, ~30 min)

Upload each file into the Drive folder shown. **Don't unzip** (the notebooks do it).

| What | Where to get it | Put it in |
|---|---|---|
| DS boundaries (shapefile zip) | data.humdata.org/dataset/cod-ab-lka → SHP download | `data/raw/boundaries/` |
| DesInventar Sri Lanka | open `desinventar-api.undrr.org/DesInventar/download_base.jsp?countrycode=lka` | `data/raw/desinventar/` |
| GFW tree-cover loss | globalforestwatch.org/dashboards/country/LKA → Download data (zip) → upload the **subnational** CSV(s) | `data/raw/gfw/` |
| OpenStreetMap | download.geofabrik.de/asia/sri-lanka.html → `sri-lanka-latest-free.shp.zip` | `data/raw/osm/` |
| Census 2024 tables | statistics.gov.lk (CPH 2024 DS-level tables) | used to fill `data/manual/census_ds.csv` |
| Flood extents (optional) | UNOSAT June 2024, Copernicus EMSR851, World Bank historical | `data/raw/flood_extents/Jun2024/`, `.../Ditwah2025/` |
| IMERG (optional) | Earthdata Search, GPM_3IMERGDL, Ditwah dates | `data/raw/imerg/` |

CHIRPS rainfall and elevation are downloaded **by the notebooks** (02, 03).

---

## 2. Run order and what to check

### Day 1 (Wed 28 night – Thu 29)
1. **00_setup_boundaries.** Expect about 330 DS divisions. *Check:* the map looks like Sri Lanka.
2. **02_rainfall_chirps.** Start it early: it takes ~30–60 min. Use *Runtime → Run all* and keep the
   tab open. If it disconnects, just run again (finished years are skipped).
3. **03_elevation.** ~10–15 min, resumes automatically if interrupted.
4. **01_floods_desinventar.**
   * *Check* the printout picks the right district / DS columns.
   * If it says "N names could not be matched": open `data/manual/ds_lookup_desinventar.csv`
     (in Google Sheets), type the right `pcode` from `data/processed/ds_list.csv`, save as CSV, re-run.
   * **Write down the last year with records.** It decides the train/test split (automatic).
5. **04_human_factors.**
   * First run creates `data/manual/census_ds.csv` (one row per division).
   * Fill `population_2024`, `households_2024`, `urban_share` (0–1) and any service columns you have.
     Missing columns are fine; Model B uses what exists.
   * Put the GFW and OSM files in place and re-run.

### Day 2 (Fri 30)
6. **05_build_table.** *Check:* flood rate of a few %, no missing rainfall or elevation.
7. **06_model_A_dfs.** The heart of the project.
   * Model A metrics table: **must beat both baselines** (PR-AUC). Screenshot it.
   * "actual vs expected" totals should be close.
   * Stability (leave-one-year-out Spearman) and known-areas table: screenshot both.
   * If too many divisions are "low confidence", lower `MIN_HEAVY_MONTHS` (e.g. to 5) and say so in the report.
8. **07_model_B_causes.** Cross-validated R² vs "always the average", SHAP summary plot, top-3 stability,
   failure profiles. Screenshot the SHAP plot.
9. **08_trend.** Worsening / improving counts, national chart.
10. **09_evaluate_recent_events.**
    * Type affected families into `data/manual/recent_events_district.csv` (25 districts × events) from
      the DMC 30 Nov sitrep, the IFRC Oct 2024 DREF, and your Ditwah package tables.
    * Add DS-level rows to `recent_events_ds.csv` wherever a sitrep gives them.
    * *Check the event dates* in the `EVENTS` dictionary match the reports.

### Dashboard (Sayuni, Fri 30)
* **Run on a laptop (PyCharm):**
  1. Download from Drive: `app/`, `outputs/`, `data/processed/ds_list.csv`,
     `data/processed/ds_divisions_simplified.geojson`. Keep the same folder layout.
  2. Run `pip install -r requirements.txt` then `streamlit run app/app.py`.
* **Deploy (public link for judges):** push the repo to GitHub, then share.streamlit.io → *New app* →
  pick the repo, main file `app/app.py` → Deploy. Put the link in the README.

### Day 3 (Sat 31): write-up
* Technical report: see `docs/REPORT_OUTLINE.md` (every number comes from a file in `outputs/`).
* Two-page summary: problem → method (one diagram) → 3 key results → what the planner does with it.
* Update the README "Results" table with your real numbers.

### Day 4 (Sun 1 Nov): video and submit
* **3-minute video on real data:** 20 s problem → 40 s data and method (architecture figure) →
  90 s live dashboard (map → click a red division → its causes and trend → priority list download →
  rain watch) → 30 s honest evaluation and limitations.
* **GitHub (public!):**
  1. Create the repo.
  2. Upload everything **except** `data/raw/`, `data/processed/chirps/` and `rain_daily.csv.gz`
     (too big; the notebooks recreate them).
  3. Download the notebooks from Drive **with their outputs**, so judges see results without running them.

---

## 3. If something goes wrong
| Problem | Fix |
|---|---|
| CHIRPS download fails for a year | Open the URL in the notebook in a browser to check; re-run (skips finished years). |
| DesInventar ends before 2023 | Nothing to change: the split moves back automatically. Say so in the report (data gap). |
| Elevation API "429" | The notebook waits and retries; if it keeps failing, wait an hour and re-run (it resumes). |
| Model A does not beat the baselines | That is a valid finding. Follow the proposal fallback: report the flood-risk ranking and the negative result honestly. |
| `geopandas` / `shap` import error in Colab | Run `!pip install geopandas shap` in a new cell, then *Runtime → Restart session*. |
| Drive path not found | Check the folder is named exactly `ModelX` and the shortcut is in **My Drive**. |
