# Technical report outline (Implementation, due Sun 1 Nov)

The brief asks for: architecture, the stack and why, **an honest evaluation against the proposal**,
and **an account of the data you finally used, its temporality and its gaps**. Judging: data work 30%,
insight 25%, engineering 25%, functionality 20%.

| # | Section | What to write | Where the material is |
|---|---|---|---|
| 1 | Problem and user (½ page) | Short recap of proposal §1–2 | proposal |
| 2 | Data actually used | Table: source, rows, years, licence, **what changed from the proposal**. Last DesInventar year, census columns you filled, % names matched, files you skipped | `outputs/metadata.json`, notebook 01/04 printouts |
| 3 | Temporality and gaps | History / held-out 2024–25 / live forecast. Under-reporting, waste proxy, district-level tree loss, OSM unevenness | proposal §5.3, GUIDE |
| 4 | Architecture | Figure 7.1 from proposal (update if changed); notebook pipeline 00–09 → outputs → Streamlit | `notebooks/`, `app/` |
| 5 | Stack and why | Proposal §7.2 table; note anything dropped (e.g. XGBoost if not best) | proposal |
| 6 | Method | Model A features, split years, best model; DFS formula and low-confidence rule; Model B features; SHAP; K-means; trend test | notebooks 06–08 |
| 7 | Results | Map screenshot, top-15 table, failure profiles, SHAP summary plot, trend counts + national chart | `outputs/*.csv`, notebook plots |
| 8 | **Evaluation vs proposal §9** | One row per promised check: *promised → result → passed?* Include failures honestly | `model_A_metrics`, `dfs_stability`, `known_areas_check`, `model_B_metrics`, `shap_stability`, `eval_recent_events` |
| 9 | Limitations and next steps | Correlation ≠ cause, data gaps, satellite mapping and hydraulic models as future work | proposal §6, §10 |
| 10 | Reproducibility | Link repo + README + dashboard link | README |

**Two-page summary:** problem (3 lines) → one architecture figure → 3 headline findings with numbers →
the map → what a planner does with it → limitations (3 bullets).
