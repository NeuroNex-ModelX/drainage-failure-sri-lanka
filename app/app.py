"""NeuroNex · Drainage Failure Dashboard (ModelX 2.0)

Run locally:   streamlit run app/app.py
Pages: Failure map · Division profile · Priority list · Rain watch
Reads the CSVs written by the notebooks in outputs/ and data/processed/.
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
PROC = os.path.join(ROOT, "data", "processed")

st.set_page_config(page_title="Drainage Failure Map – Sri Lanka", page_icon="🌧️", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read(name, folder=OUT):
    path = os.path.join(folder, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load():
    ds = read("ds_list.csv", PROC)
    dfs = read("dfs_ds.csv")
    causes = read("causes_ds.csv")
    trend = read("trend_ds.csv")
    shap = read("shap_values.csv")
    yearly = read("obs_vs_exp_yearly.csv")
    with open(os.path.join(PROC, "ds_divisions_simplified.geojson")) as f:
        geo = json.load(f)
    meta_path = os.path.join(OUT, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    df = ds.merge(dfs.drop(columns=["ds_name", "district_name"], errors="ignore"), on="pcode", how="left")
    if not causes.empty:
        df = df.merge(causes, on="pcode", how="left")
    if not trend.empty:
        df = df.merge(trend[["pcode", "rate_p1", "rate_p2", "trend"]], on="pcode", how="left")
    df["low_confidence"] = df["low_confidence"].fillna(True).astype(bool)
    df["DFS"] = df["dfs"].where(~df["low_confidence"])
    for c in ["main_cause", "second_cause", "failure_profile", "trend"]:
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("")
    return df, geo, shap, yearly, meta


try:
    df, geo, shap_df, yearly, meta = load()
except FileNotFoundError:
    st.error("Data files are missing. Run notebooks 00–08 first (see README).")
    st.stop()

NICE = {
    "est_waste_collection": "Waste collection (est.)", "urban_share": "Urban share",
    "pop_density": "Population density", "pop_growth_2012_2024": "Population growth",
    "hh_piped_water_share": "Piped-water coverage", "hh_flush_toilet_share": "Flush-toilet coverage",
    "hh_clean_cooking_share": "Clean-cooking coverage", "tree_loss_pct_2001_2024": "Tree-cover loss",
    "tree_loss_pct_2015_2024": "Recent tree-cover loss", "drain_density": "Drain/canal density",
    "stream_density": "Stream density",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌧️ Drainage Failure")
st.sidebar.caption("Where does Sri Lanka flood more than its rain and land height explain?")
page = st.sidebar.radio("Page", ["Failure map", "Division profile", "Priority list", "Rain watch"])
if "selected" not in st.session_state:
    ranked = df.dropna(subset=["DFS"]).sort_values("DFS", ascending=False)
    st.session_state.selected = ranked["pcode"].iloc[0] if len(ranked) else df["pcode"].iloc[0]

with st.sidebar.expander("Data layers (as of)"):
    LAYERS = {"flood_records": "Flood records", "dfs_period": "Score period", "rainfall": "Rainfall",
              "elevation": "Land height", "census": "Census", "tree_loss": "Tree loss", "drains": "Drains",
              "trend_periods": "Trend periods", "model_A": "Model A"}
    for k, lbl in LAYERS.items():
        if k in meta:
            st.markdown(f"**{lbl}:** {meta[k]}")
    st.caption(f"Models last run: {meta.get('_updated', 'unknown')}")
st.sidebar.caption("Results show factors *linked to* drainage failure, not proof of cause.")


def choropleth(data, color, title, scale="Reds", range_color=None, hover=None):
    kw = dict(geojson=geo, locations="pcode", featureidkey="properties.pcode", color=color,
              color_continuous_scale=scale, range_color=range_color, hover_name="ds_name",
              hover_data=hover or {}, center={"lat": 7.85, "lon": 80.7}, zoom=6.3, opacity=0.75, height=700)
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(data, map_style="carto-positron", **kw)
    else:
        fig = px.choropleth_mapbox(data, mapbox_style="carto-positron", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title=title)
    return fig


# ---------------------------------------------------------------------------
# Page 1 · Failure map
# ---------------------------------------------------------------------------
if page == "Failure map":
    st.header("Drainage Failure Score by DS division")
    st.markdown("**DFS = (actual floods − expected floods) ÷ expected floods.** Expected floods come from a "
                "model that only knows rainfall and land height. Red = floods far more than nature explains. "
                "Grey/blank = low confidence (too little evidence to rank).")
    lo, hi = df["DFS"].quantile([0.05, 0.95])
    fig = choropleth(df, "DFS", "", scale="RdYlBu_r", range_color=(lo, hi),
                     hover={"district_name": True, "DFS": ":.2f", "main_cause": True, "trend": True,
                            "pcode": False})
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
    try:
        pts = event.selection.points if event else []
        if pts:
            st.session_state.selected = pts[0].get("location", st.session_state.selected)
            st.info(f"Selected **{df.set_index('pcode').loc[st.session_state.selected, 'ds_name']}** "
                    "– open *Division profile* to see why.")
    except Exception:
        pass
    c1, c2, c3 = st.columns(3)
    c1.metric("Divisions ranked", int(df["DFS"].notna().sum()))
    c2.metric("Flooding ≥ 2× expected", int((df["DFS"] >= 1).sum()))
    c3.metric("Worsening trend", int((df["trend"] == "Worsening").sum()))

# ---------------------------------------------------------------------------
# Page 2 · Division profile
# ---------------------------------------------------------------------------
elif page == "Division profile":
    opts = df.sort_values(["district_name", "ds_name"])
    labels = {p: f"{n} ({d})" for p, n, d in zip(opts["pcode"], opts["ds_name"], opts["district_name"])}
    sel = st.selectbox("DS division", list(labels), index=list(labels).index(st.session_state.selected),
                       format_func=labels.get)
    st.session_state.selected = sel
    r = df.set_index("pcode").loc[sel]
    st.header(f"{r['ds_name']} · {r['district_name']} District")
    if r["low_confidence"]:
        st.warning("Low confidence: too few heavy-rain months or expected floods to rank this division.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drainage Failure Score", "–" if pd.isna(r["dfs"]) else f"{r['dfs']:.2f}")
    c2.metric("Rank (1 = worst)", "–" if pd.isna(r.get("rank")) else f"{int(r['rank'])} / {int(df['DFS'].notna().sum())}")
    c3.metric("Floods: actual / expected", "–" if pd.isna(r.get("actual")) else f"{int(r['actual'])} / {r['expected']:.1f}")
    c4.metric("Trend", r["trend"] or "–")
    st.markdown(f"**Main linked factor:** {r['main_cause'] or '–'}  ·  **Second:** {r['second_cause'] or '–'}  ·  "
                f"**Failure profile:** {r['failure_profile'] or '–'}")
    if pd.notna(r.get("rate_p1")) and pd.notna(r.get("rate_p2")):
        st.caption(f"Floods per 100 heavy-rain days: {r['rate_p1']:.1f} → {r['rate_p2']:.1f} "
                   f"({meta.get('trend_periods', '')})")

    left, right = st.columns(2)
    with left:
        st.subheader("Why? Factors pushing the score up or down (SHAP)")
        if not shap_df.empty and sel in set(shap_df["pcode"]):
            s = shap_df.set_index("pcode").loc[sel].rename(NICE).sort_values()
            fig = px.bar(s, orientation="h", labels={"value": "contribution to DFS", "index": ""},
                         color=s.values > 0, color_discrete_map={True: "#c0392b", False: "#2e86c1"})
            fig.update_layout(showlegend=False, height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cause analysis for this division (low confidence or not in Model B).")
    with right:
        st.subheader("Floods each year: actual vs expected")
        y = yearly[yearly["pcode"] == sel]
        if len(y):
            fig = px.line(y, x="year", y=["actual", "expected"], markers=True,
                          labels={"value": "flooded months", "variable": ""})
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 3 · Priority list
# ---------------------------------------------------------------------------
elif page == "Priority list":
    st.header("Priority list for pre-monsoon work")
    c1, c2, c3 = st.columns(3)
    dist = c1.multiselect("District", sorted(df["district_name"].unique()))
    cause = c2.multiselect("Main linked factor", sorted(x for x in df["main_cause"].unique() if x))
    prof = c3.multiselect("Failure profile", sorted(x for x in df["failure_profile"].unique() if x))
    t = df.dropna(subset=["DFS"]).sort_values("DFS", ascending=False)
    if dist: t = t[t["district_name"].isin(dist)]
    if cause: t = t[t["main_cause"].isin(cause)]
    if prof: t = t[t["failure_profile"].isin(prof)]
    show = t[["rank", "ds_name", "district_name", "DFS", "actual", "expected", "main_cause",
              "second_cause", "failure_profile", "trend"]].rename(columns={
        "rank": "Rank", "ds_name": "DS division", "district_name": "District", "actual": "Actual floods",
        "expected": "Expected floods", "main_cause": "Main linked factor", "second_cause": "Second factor",
        "failure_profile": "Profile", "trend": "Trend"})
    st.dataframe(show.round(2), use_container_width=True, hide_index=True, height=560)
    st.download_button("⬇️ Download as CSV", show.to_csv(index=False), "drainage_priority_list.csv", "text/csv")
    st.caption("Suggested action: waste-collection factors → cleaning crews and collection requests; "
               "low drain density → capital-works request; worsening trend → inspect first.")

# ---------------------------------------------------------------------------
# Page 4 · Rain watch (live Open-Meteo forecast)
# ---------------------------------------------------------------------------
else:
    st.header("Rain watch: next 3 days")
    cache_file = os.path.join(OUT, "forecast_latest.csv")

    @st.cache_data(ttl=3 * 3600, show_spinner="Fetching the forecast from Open-Meteo...")
    def fetch_forecast(lat_lon):
        rows = []
        for i in range(0, len(lat_lon), 100):
            chunk = lat_lon[i:i + 100]
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
                "latitude": ",".join(f"{a:.4f}" for _, a, _ in chunk),
                "longitude": ",".join(f"{b:.4f}" for _, _, b in chunk),
                "daily": "precipitation_sum", "forecast_days": 3, "timezone": "Asia/Colombo"})
            r.raise_for_status()
            js = r.json()
            js = js if isinstance(js, list) else [js]
            for (p, _, _), loc in zip(chunk, js):
                for d, v in zip(loc["daily"]["time"], loc["daily"]["precipitation_sum"]):
                    rows.append({"pcode": p, "date": d, "rain_mm": v})
        out = pd.DataFrame(rows)
        out["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return out

    pts = list(zip(df["pcode"], df["lat"], df["lon"]))
    try:
        fc = fetch_forecast(pts)
        try:
            fc.to_csv(cache_file, index=False)
        except OSError:
            pass
        st.success(f"Live forecast · fetched {fc['fetched_at'].iloc[0]} (Asia/Colombo)")
    except Exception as e:
        if os.path.exists(cache_file):
            fc = pd.read_csv(cache_file)
            st.warning(f"Forecast service unavailable – showing the last saved forecast "
                       f"(data as of {fc['fetched_at'].iloc[0]}).")
        else:
            st.error(f"Forecast unavailable and no saved copy: {e}")
            st.stop()

    tot = fc.groupby("pcode")["rain_mm"].agg(rain_3day="sum", max_day="max").reset_index()
    w = df.merge(tot, on="pcode", how="left")
    c1, c2, c3 = st.columns(3)
    heavy3 = c1.slider("Heavy rain: 3-day total ≥ (mm)", 25, 200, 75, 5)
    heavy1 = c2.slider("…or any single day ≥ (mm)", 20, 150, 50, 5)
    dfs_min = c3.slider("Drainage Failure Score ≥", 0.0, 3.0, 0.5, 0.1)
    alert = w[((w["rain_3day"] >= heavy3) | (w["max_day"] >= heavy1)) & (w["DFS"] >= dfs_min)]
    alert = alert.sort_values(["DFS", "rain_3day"], ascending=False)

    left, right = st.columns([3, 2])
    with left:
        fig = choropleth(w, "rain_3day", "Forecast rain, next 3 days (mm)", scale="Blues",
                         hover={"district_name": True, "rain_3day": ":.0f", "DFS": ":.2f", "pcode": False})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(f"⚠️ {len(alert)} high-score divisions facing heavy rain")
        st.dataframe(alert[["ds_name", "district_name", "rain_3day", "max_day", "DFS", "main_cause"]].rename(
            columns={"ds_name": "DS division", "district_name": "District", "rain_3day": "3-day rain (mm)",
                     "max_day": "Wettest day (mm)", "main_cause": "Main linked factor"}).round(1),
            hide_index=True, use_container_width=True, height=560)

st.sidebar.markdown("---")
st.sidebar.caption("Weather data by Open-Meteo.com (CC BY 4.0) · Rainfall history: CHIRPS (UCSB CHC) · "
                   "Floods: DesInventar (UNDRR/DMC) · Boundaries: Survey Dept / OCHA HDX · "
                   "Census: DCS · Tree loss: Global Forest Watch · Drains: © OpenStreetMap contributors")
