"""Shared helpers for the NeuroNex drainage-failure project (ModelX 2.0).

Every notebook does:
    from common import *
after adding the project's src/ folder to sys.path (see the setup cell).
"""
import os
import re
import json
import difflib
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.environ.get("MODELX_BASE", "/content/drive/MyDrive/ModelX")
RAW = os.path.join(BASE, "data", "raw")          # downloads (not committed to GitHub)
MANUAL = os.path.join(BASE, "data", "manual")    # small hand-made tables (committed)
PROC = os.path.join(BASE, "data", "processed")   # cleaned tables made by notebooks
OUT = os.path.join(BASE, "outputs")              # model results read by the dashboard

for _d in (RAW, MANUAL, PROC, OUT):
    os.makedirs(_d, exist_ok=True)

# Sri Lanka bounding box (a little wider than the island)
LAT_MIN, LAT_MAX = 5.8, 10.0
LON_MIN, LON_MAX = 79.4, 82.0

# Sri Lanka's national projected coordinate system (metres) - used for areas/lengths
SL_CRS = "EPSG:5235"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def save_meta(key, value):
    """Record a fact about a data layer (e.g. its date) in outputs/metadata.json.
    The dashboard shows these as 'data as of' notes."""
    path = os.path.join(OUT, "metadata.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta[key] = value
    meta["_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta():
    path = os.path.join(OUT, "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def find_files(folder, patterns):
    """Recursively find files whose name matches any regex in patterns."""
    hits = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if any(re.search(p, fn, re.I) for p in patterns):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


# ---------------------------------------------------------------------------
# Name matching (DS divisions are spelt differently in every source)
# ---------------------------------------------------------------------------
_DROP_WORDS = r"\b(ds|d s|division|divisional|secretariat|secretary|pradeshiya|sabha|ps|mc|uc|district)\b"


def norm_name(s):
    """Lower-case, strip punctuation, remove generic words, collapse spaces."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(_DROP_WORDS, " ", s)
    s = re.sub(r"\s+", "", s)  # remove spaces too: 'Nuwara Eliya' == 'NUWARAELIYA'
    return s


def load_ds_list():
    """The master list of DS divisions made by notebook 00 (pandas only, no GeoPandas)."""
    path = os.path.join(PROC, "ds_list.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run notebook 00 first: data/processed/ds_list.csv is missing.")
    return pd.read_csv(path)


def match_ds(df, district_col, ds_col, lookup_name, cutoff=0.82):
    """Attach the official HDX pcode to rows of df using district + DS names.

    1. Uses data/manual/<lookup_name> first (your hand-fixed matches).
    2. Then exact normalised match inside the same district.
    3. Then fuzzy match inside the same district (difflib, cutoff).
    Unmatched names are written back to the lookup file with an empty pcode so a
    team member can fill them in by hand and re-run.
    """
    ds = load_ds_list()
    ds["_d"] = ds["district_name"].map(norm_name)
    ds["_s"] = ds["ds_name"].map(norm_name)

    lookup_path = os.path.join(MANUAL, lookup_name)
    lookup = {}
    if os.path.exists(lookup_path):
        lk = pd.read_csv(lookup_path, dtype=str).fillna("")
        for _, r in lk.iterrows():
            if r.get("pcode", ""):
                lookup[(norm_name(r["source_district"]), norm_name(r["source_ds"]))] = r["pcode"]

    pairs = df[[district_col, ds_col]].drop_duplicates()
    result, unmatched = {}, []
    for _, r in pairs.iterrows():
        d, s = norm_name(r[district_col]), norm_name(r[ds_col])
        key = (d, s)
        if key in lookup:
            result[(r[district_col], r[ds_col])] = lookup[key]
            continue
        cand = ds[ds["_d"] == d]
        if cand.empty:  # district itself spelt differently -> fuzzy match district
            dm = difflib.get_close_matches(d, ds["_d"].unique().tolist(), n=1, cutoff=0.8)
            cand = ds[ds["_d"] == dm[0]] if dm else ds
        exact = cand[cand["_s"] == s]
        if len(exact) == 1:
            result[(r[district_col], r[ds_col])] = exact["pcode"].iloc[0]
            continue
        fm = difflib.get_close_matches(s, cand["_s"].tolist(), n=1, cutoff=cutoff)
        if fm:
            result[(r[district_col], r[ds_col])] = cand.loc[cand["_s"] == fm[0], "pcode"].iloc[0]
        else:
            unmatched.append({"source_district": r[district_col], "source_ds": r[ds_col], "pcode": ""})

    out = df.copy()
    out["pcode"] = [result.get((a, b)) for a, b in zip(out[district_col], out[ds_col])]

    if unmatched:
        old = pd.read_csv(lookup_path, dtype=str).fillna("") if os.path.exists(lookup_path) else \
            pd.DataFrame(columns=["source_district", "source_ds", "pcode"])
        new = pd.DataFrame(unmatched)
        both = pd.concat([old, new]).drop_duplicates(["source_district", "source_ds"], keep="first")
        both.to_csv(lookup_path, index=False)
        print(f"{len(unmatched)} names could not be matched automatically.")
        print(f"   -> open data/manual/{lookup_name}, fill the 'pcode' column (see data/processed/ds_list.csv), re-run.")
    print(f"Matched {out['pcode'].notna().mean():.1%} of rows to a DS division.")
    return out


def district_of(pcode_series):
    """Map DS pcode -> district name using ds_list.csv."""
    ds = load_ds_list().set_index("pcode")
    return pcode_series.map(ds["district_name"])
