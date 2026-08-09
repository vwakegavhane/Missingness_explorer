"""
missingness.py
==============
Data-preparation and missingness-computation layer for the interactive
missingness-visualisation project (UK Crop Microbiome Cryobank, or any
tabular dataset).

It reads an incomplete table, normalises the many ways "missing" can be
encoded, and computes the metrics the front end needs:

  1. Per-variable missing rate      (bar chart / matrix column order)
  2. Per-record missing count       (row order + distribution)
  3. Distinct row-level patterns    (which columns go missing together)
  4. Co-missingness relationships   (nullity correlation between columns)

Compact summaries are exported as JSON (for the D3 front end) and CSV
(for inspection and the write-up). Heavy per-record data stays in CSV;
only a histogram of it goes into the JSON, keeping the front-end payload
small.

Usage
-----
    # Run immediately on synthetic, crop-like data (no file needed):
    python missingness.py --synthetic --outdir out

    # Run on your real dataset:
    python missingness.py --input crop_microbiome.csv --outdir out
    python missingness.py --input crop_microbiome.xlsx --sheet 0 --outdir out

Requirements: pandas, numpy   (openpyxl for .xlsx, pyarrow for .parquet)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------

# Strings that commonly stand in for a missing value. Absence encoded as one
# of these will NOT be caught by isna() unless we convert it first, so every
# metric below depends on getting this right. Extend it for your dataset.
DEFAULT_SENTINELS = [
    "", " ", "NA", "N/A", "n/a", "na", "NaN", "nan", "NULL", "null",
    "None", "none", "-", "--", "?", "#N/A", "-999", "-9999",
]


def load_table(path: str, sheet=0, sentinels=None) -> pd.DataFrame:
    """Read a .csv / .tsv / .xlsx / .parquet file, treating sentinel strings
    as missing so they register as NaN."""
    sentinels = DEFAULT_SENTINELS if sentinels is None else sentinels
    ext = os.path.splitext(path)[1].lower()

    if ext in (".csv", ".txt"):
        df = pd.read_csv(path, na_values=sentinels, keep_default_na=True,
                         low_memory=False)
    elif ext in (".tsv",):
        df = pd.read_csv(path, sep="\t", na_values=sentinels,
                         keep_default_na=True, low_memory=False)
    elif ext in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path, sheet_name=sheet, na_values=sentinels,
                           keep_default_na=True)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return normalise_missing(df, sentinels)


def normalise_missing(df: pd.DataFrame, sentinels=None) -> pd.DataFrame:
    """Second pass: strip whitespace and convert any residual sentinel
    strings in object columns to NaN (parquet, and some Excel exports,
    skip the na_values step)."""
    sentinels = DEFAULT_SENTINELS if sentinels is None else sentinels
    sentinel_set = {s.strip().lower() for s in sentinels}
    df = df.copy()
    # Text-like columns only (object / str / string), across pandas versions.
    text_cols = [c for c in df.columns
                 if not pd.api.types.is_numeric_dtype(df[c])
                 and not pd.api.types.is_datetime64_any_dtype(df[c])
                 and not pd.api.types.is_bool_dtype(df[c])]
    for col in text_cols:
        stripped = df[col].astype(str).str.strip()
        is_missing = stripped.str.lower().isin(sentinel_set)
        df.loc[is_missing, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------

def overview(df: pd.DataFrame) -> dict:
    """Headline numbers for the whole table."""
    n, m = df.shape
    total = n * m
    n_missing = int(df.isna().to_numpy().sum())
    return {
        "n_records": int(n),
        "n_variables": int(m),
        "total_cells": int(total),
        "n_missing_cells": n_missing,
        "overall_missing_rate": (n_missing / total) if total else 0.0,
    }


def per_variable_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and rate for each column, sorted most-missing first.
    This ordering drives the column order of the missingness matrix."""
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame({
        "variable": miss.index,
        "dtype": [str(df[c].dtype) for c in miss.index],
        "n_missing": miss.to_numpy(),
        "n_present": n - miss.to_numpy(),
        "missing_rate": miss.to_numpy() / n if n else 0.0,
    })
    return out.sort_values("missing_rate", ascending=False).reset_index(drop=True)


def per_record_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and rate for each row."""
    m = df.shape[1]
    miss = df.isna().sum(axis=1)
    return pd.DataFrame({
        "record_index": df.index,
        "n_missing": miss.to_numpy(),
        "missing_rate": miss.to_numpy() / m if m else 0.0,
    })


def record_missing_distribution(per_record: pd.DataFrame) -> list[dict]:
    """Histogram of per-record missing counts — compact enough for JSON,
    unlike the full per-record table."""
    dist = per_record["n_missing"].value_counts().sort_index()
    return [{"n_missing": int(k), "count": int(v)} for k, v in dist.items()]


def enumerate_patterns(df: pd.DataFrame) -> list[dict]:
    """Find the distinct row-level missingness patterns and their frequencies.

    Each row's missingness mask is bit-packed so identical patterns collapse
    to the same key; this is fast even at 80k+ rows. Returns patterns sorted
    by frequency, each described by the set of variables it leaves missing.
    """
    mask = df.isna().to_numpy()
    n, m = mask.shape
    packed = np.packbits(mask, axis=1)                 # (n, ceil(m/8)) bytes
    keys = [row.tobytes() for row in packed]           # one hashable key per row
    counts = pd.Series(keys).value_counts()

    cols = list(df.columns)
    patterns = []
    for pid, (key, cnt) in enumerate(counts.items()):
        bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))[:m].astype(bool)
        missing_vars = [cols[i] for i in range(m) if bits[i]]
        patterns.append({
            "pattern_id": int(pid),
            "count": int(cnt),
            "proportion": float(cnt / n),
            "n_missing_vars": int(bits.sum()),
            "missing_variables": missing_vars,
        })
    return patterns


def nullity_correlation(df: pd.DataFrame):
    """Pairwise correlation of the missingness indicators (phi coefficient on
    0/1 masks). Columns that are entirely missing or entirely present have no
    variance, so their correlation is undefined — those are dropped here and
    reported separately by the caller."""
    mask = df.isna().astype(int)
    varying = mask.columns[mask.nunique() > 1]
    constant = [c for c in mask.columns if c not in set(varying)]
    corr = mask[varying].corr() if len(varying) else pd.DataFrame()
    return corr, constant


def comissingness_edges(corr: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Turn the (dense) correlation matrix into a compact edge list of the
    strongest co-missingness relationships, for the linked view."""
    edges = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and abs(v) >= threshold:
                edges.append({
                    "source": cols[i],
                    "target": cols[j],
                    "corr": round(float(v), 4),
                })
    return sorted(edges, key=lambda e: abs(e["corr"]), reverse=True)


# ---------------------------------------------------------------------------
# 3. Data-quality diagnostic  (--diagnose)
# ---------------------------------------------------------------------------
#
# Every metric above depends on absence being NaN. If missingness is instead
# hidden as a sentinel string ("NA" that slipped through) or a numeric code
# (-999 for "not measured"), isna() misses it and the metrics silently lie.
# This step scans each column and flags what a cleaning pass needs to fix,
# BEFORE you trust any numbers. It tells you what your clean.py should handle.

# Numeric codes that frequently stand in for "missing" in lab / survey data.
SUSPECT_NUMERIC_CODES = [-1, -9, -99, -999, -9999, -99999, 999, 9999, 99999, 999999]


def _numeric_share(series: pd.Series):
    """For an object column, return (fraction of non-null values that parse as
    numbers, up to 10 distinct values that do NOT parse). A high fraction with
    a few stragglers usually means a numeric column with hidden missing tokens
    sitting in those stragglers."""
    nonnull = series.dropna().astype(str).str.strip()
    if len(nonnull) == 0:
        return 0.0, []
    parsed = pd.to_numeric(nonnull, errors="coerce")
    non_parsing = sorted(set(nonnull[parsed.isna()]))
    return float(parsed.notna().mean()), non_parsing[:10]


def diagnose(df: pd.DataFrame) -> list[dict]:
    """Per-column data-quality report. Flags the issues that most often
    corrupt a missingness analysis: hidden sentinels, numeric-looking text,
    disguised numeric codes, constant and all-missing columns, and likely IDs."""
    n = len(df)
    report = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_present = n - n_missing
        nun = int(s.nunique(dropna=True))
        info = {
            "variable": col,
            "dtype": str(s.dtype),
            "n_unique": nun,
            "n_missing": n_missing,
            "missing_rate": (n_missing / n) if n else 0.0,
            "flags": [],
        }

        is_numeric = (pd.api.types.is_numeric_dtype(s)
                      and not pd.api.types.is_bool_dtype(s))

        if n_missing == n:
            info["flags"].append("ALL_MISSING")
        elif nun <= 1:
            info["flags"].append("CONSTANT")

        # Text-like column: look for numbers hidden as strings + stray tokens.
        if not is_numeric and n_present > 0:
            frac_num, non_parsing = _numeric_share(s)
            if frac_num >= 0.80 and non_parsing:
                # mostly numeric, but a handful of odd tokens — likely hidden missing
                info["flags"].append("NUMERIC_WITH_HIDDEN_TOKENS")
                info["suspect_tokens"] = non_parsing
            elif frac_num >= 0.95:
                info["flags"].append("NUMERIC_STORED_AS_TEXT")
            # Only genuine (non-numeric) text with near-unique values is ID-like;
            # a high-cardinality numeric measurement is not an identifier.
            elif frac_num < 0.5 and nun >= 0.95 * n_present:
                info["flags"].append("LIKELY_ID")

        # Numeric column: look for sentinel codes disguised as real values.
        if is_numeric:
            hits = {}
            for code in SUSPECT_NUMERIC_CODES:
                c = int((s == code).sum())     # == handles int/float match
                if c > 0:
                    hits[str(code)] = c
            if hits:
                info["flags"].append("SUSPECT_NUMERIC_CODE")
                info["suspect_codes"] = hits

        report.append(info)
    return report


def print_diagnosis(report: list[dict]):
    """Readable console summary: only the columns that need attention."""
    flagged = [r for r in report if r["flags"]]
    print("\n=== DATA-QUALITY DIAGNOSTIC ===")
    print(f"Columns scanned: {len(report)}   Flagged: {len(flagged)}\n")
    if not flagged:
        print("No issues found — missingness appears to be encoded as NaN.\n")
        return
    for r in flagged:
        print(f"[{', '.join(r['flags'])}]  {r['variable']}  "
              f"(dtype={r['dtype']}, unique={r['n_unique']}, "
              f"missing={r['missing_rate']:.1%})")
        if "suspect_tokens" in r:
            print(f"      hidden-missing candidates: {r['suspect_tokens']}")
        if "suspect_codes" in r:
            print(f"      suspicious numeric codes:  {r['suspect_codes']}")
    print("\nReview the flagged columns. Add any confirmed missing markers to")
    print("--sentinels (or your clean.py) and re-run before trusting the metrics.\n")


# ---------------------------------------------------------------------------
# 3b. Missingness mechanism  (why is a value absent?)
# ---------------------------------------------------------------------------
#
# The step that moves the tool beyond "what is missing" to "why". We relate a
# target variable's absence to an *explanatory* variable: if a column is
# missing in exactly one group of samples (e.g. target_gene absent for every
# metagenome run), the absence is structured, not random. This is the
# evidence a user needs to decide how to treat the gap.

def detect_explanatory(df: pd.DataFrame, max_card: int = 12,
                       max_missing: float = 0.20) -> list:
    """Candidate explanatory variables: well-populated, low-cardinality
    categoricals (not identifiers, not free text)."""
    cands = []
    for c in df.columns:
        miss = df[c].isna().mean()
        nun = df[c].nunique(dropna=True)
        if miss <= max_missing and 2 <= nun <= max_card:
            cands.append(c)
    return cands


def target_variables(per_var: pd.DataFrame) -> list:
    """Variables with partial missingness (0 < rate < 1) — the ones whose
    absence a mechanism might explain."""
    return [r["variable"] for _, r in per_var.iterrows()
            if 0 < r["missing_rate"] < 1]


def assign_pattern_ids(df: pd.DataFrame, patterns: list) -> pd.Series:
    """Label each record with the pattern_id it belongs to."""
    mask = df.isna().to_numpy()
    m = mask.shape[1]
    keys = [row.tobytes() for row in np.packbits(mask, axis=1)]
    colidx = {c: i for i, c in enumerate(df.columns)}

    def key_for(missing_vars):
        bits = np.zeros(m, dtype=bool)
        for v in missing_vars:
            bits[colidx[v]] = True
        return np.packbits(bits).tobytes()

    key2pid = {key_for(p["missing_variables"]): p["pattern_id"] for p in patterns}
    return pd.Series([key2pid[k] for k in keys], index=df.index)


def explanatory_breakdown(df: pd.DataFrame, patterns: list,
                          per_var: pd.DataFrame,
                          max_targets: int = 25) -> dict:
    """For each candidate explanatory variable, compute two things:
      - conditional_missing: each target's missing rate within each group
        (the mechanism evidence — a rate of 1.0 in one group and 0.0 in
        another means the group fully explains the absence);
      - pattern_composition: how each row-pattern splits across the groups.
    """
    ev_vars = detect_explanatory(df)
    targets = target_variables(per_var)[:max_targets]
    pids = assign_pattern_ids(df, patterns)

    out = {"candidates": ev_vars, "targets": targets,
           "conditional_missing": {}, "pattern_composition": {}}

    for ev in ev_vars:
        vc = df[ev].value_counts()
        values = [{"value": str(v), "count": int(c)} for v, c in vc.items()]

        cond = {}
        for tv in targets:
            rates = {}
            for val in vc.index:
                grp = df[df[ev] == val]
                rates[str(val)] = float(grp[tv].isna().mean())
            cond[tv] = rates
        out["conditional_missing"][ev] = {"values": values, "targets": cond}

        tab = pd.crosstab(pids, df[ev])
        comp = {}
        for pid in tab.index:
            comp[int(pid)] = {str(v): int(tab.loc[pid, v])
                              for v in tab.columns if int(tab.loc[pid, v]) > 0}
        out["pattern_composition"][ev] = comp

    return out


# ---------------------------------------------------------------------------
# 4. Export
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_and_export(df: pd.DataFrame, outdir: str, edge_threshold: float = 0.5,
                     title: str = None, subtitle: str = None):
    os.makedirs(outdir, exist_ok=True)

    ov = overview(df)
    per_var = per_variable_missingness(df)
    per_rec = per_record_missingness(df)
    patterns = enumerate_patterns(df)
    corr, constant_cols = nullity_correlation(df)
    edges = comissingness_edges(corr, edge_threshold) if not corr.empty else []
    explanatory = explanatory_breakdown(df, patterns, per_var)

    # Compact summary for the front end (small: no per-record rows here).
    summary = {
        "overview": ov,
        "variables": per_var.to_dict(orient="records"),
        "record_missing_distribution": record_missing_distribution(per_rec),
        "patterns": patterns,
        "comissingness_edges": edges,
        "constant_columns": constant_cols,  # all-missing or all-present
        "explanatory": explanatory,         # mechanism: missingness by group
    }
    if title:
        summary["title"] = title
    if subtitle:
        summary["subtitle"] = subtitle
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # Fuller tables as CSV for inspection / the write-up.
    per_var.to_csv(os.path.join(outdir, "per_variable.csv"), index=False)
    per_rec.to_csv(os.path.join(outdir, "per_record.csv"), index=False)
    if not corr.empty:
        corr.to_csv(os.path.join(outdir, "nullity_correlation.csv"))

    # Console report.
    print(f"Records:   {ov['n_records']:,}")
    print(f"Variables: {ov['n_variables']:,}")
    print(f"Overall missing rate: {ov['overall_missing_rate']:.1%}")
    print(f"Distinct row patterns: {len(patterns)}")
    if patterns:
        top = patterns[0]
        print(f"  most common pattern covers {top['proportion']:.1%} of records "
              f"({top['n_missing_vars']} vars missing)")
    print(f"Co-missingness edges (|phi| >= {edge_threshold}): {len(edges)}")
    print(f"Wrote summary.json + CSVs to '{outdir}/'")
    return summary


# ---------------------------------------------------------------------------
# 5. Synthetic crop-like data (for immediate testing without the real file)
# ---------------------------------------------------------------------------

def make_synthetic(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Generate a small table whose missingness mirrors the crop dataset:
    always-present IDs, near-complete columns, near-empty columns, an MNAR
    pair (total-digestion absent / extractable present), and a coherent
    sequencing band missing for a subset of rows."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()

    # Always-present identifiers / descriptors.
    df["soil_id"] = [f"S{i:05d}" for i in range(n)]
    df["crop"] = rng.choice(["wheat", "barley", "oat", "bean"], n)
    df["region"] = rng.choice(["N", "S", "E", "W"], n)

    # Near-complete numeric soil properties (~2% random missing).
    for c in ["pH", "organic_carbon", "moisture", "nitrogen", "phosphorus"]:
        vals = rng.normal(size=n)
        vals[rng.random(n) < 0.02] = np.nan
        df[c] = vals

    # Extractable trace elements: fully recorded.
    for el in ["Fe", "Zn", "Cu", "Mn"]:
        df[f"extractable_{el}"] = rng.normal(size=n)
    # Total-digestion counterparts: entirely absent (MNAR — never measured).
    for el in ["Fe", "Zn", "Cu", "Mn"]:
        df[f"total_digestion_{el}"] = np.nan

    # Sequencing block: missing together for ~40% of rows (coherent band).
    seq_missing = rng.random(n) < 0.40
    for c in ["read_count", "otu_richness", "shannon_index"]:
        vals = rng.normal(size=n)
        vals[seq_missing] = np.nan
        df[c] = vals

    # A moderately missing column, missing independently (~25%).
    vals = rng.normal(size=n)
    vals[rng.random(n) < 0.25] = np.nan
    df["bulk_density"] = vals

    return df


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute missingness metrics.")
    ap.add_argument("--input", help="Path to .csv/.tsv/.xlsx/.parquet dataset")
    ap.add_argument("--sheet", default=0, help="Sheet name/index for .xlsx")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--edge-threshold", type=float, default=0.5,
                    help="Min |phi| for a co-missingness edge")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use built-in crop-like synthetic data instead of a file")
    ap.add_argument("--n", type=int, default=2000, help="Rows for synthetic data")
    ap.add_argument("--diagnose", action="store_true",
                    help="Run a data-quality scan (hidden sentinels, disguised "
                         "codes, constant/ID columns) before computing metrics")
    ap.add_argument("--diagnose-only", action="store_true",
                    help="Run the data-quality scan and stop (no metrics)")
    ap.add_argument("--title", default=None,
                    help="Dataset name shown in the front end (dataset-agnostic)")
    ap.add_argument("--subtitle", default=None,
                    help="Short description shown under the title")
    args = ap.parse_args()

    if args.synthetic or not args.input:
        if not args.synthetic:
            print("No --input given; using synthetic data. "
                  "Pass --input <file> to run on your dataset.\n")
        df = make_synthetic(n=args.n)
    else:
        df = load_table(args.input, sheet=args.sheet)

    if args.diagnose or args.diagnose_only:
        report = diagnose(df)
        print_diagnosis(report)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "diagnose.json"), "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        if args.diagnose_only:
            return

    build_and_export(df, args.outdir, edge_threshold=args.edge_threshold,
                     title=args.title, subtitle=args.subtitle)


if __name__ == "__main__":
    main()
