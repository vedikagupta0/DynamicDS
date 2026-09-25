"""Standardised, evidence-based warnings. Each has severity, column, issue, evidence, recommendation."""
from __future__ import annotations

from app import config as C

ORDER = {"HIGH": 0, "WARNING": 1, "INFO": 2}


def _w(sev, col, issue, evidence, rec, cat):
    return {"severity": sev, "column": col, "issue": issue, "evidence": evidence, "recommendation": rec, "category": cat}


def build_warnings(profile: dict, quality: dict, eda: dict | None = None, target: str | None = None) -> list[dict]:
    W = []
    ov = profile["overview"]
    if ov["rows"] < 200:
        W.append(_w("WARNING", None, "Small dataset", f"{ov['rows']} rows.", "Model metrics will be unstable; prefer simple models and treat results as indicative.", "size"))
    for c in quality["missing"]["columns"]:
        p, col = c["missing_pct"], c["column"]
        if target and col == target and p > 0:
            sev = "HIGH" if p > 20 else ("WARNING" if p >= 5 else "INFO")
            W.append(_w(sev, col, f"Target contains {p:.1f}% missing values", f"{c['missing']} of {ov['rows']} rows.",
                        "Rows without a target are excluded from training. Confirm this is acceptable.", "missing"))
        elif c["level"] in ("high", "very_high"):
            W.append(_w("WARNING", col, f"{col} contains {p:.1f}% missing values", f"{c['missing']} missing of {ov['rows']}.",
                        c["recommendation"], "missing"))
    mr = quality["missing"]
    if mr["flagged_row_count"]:
        W.append(_w("WARNING", None, f"{mr['flagged_row_count']} rows have at least {mr['flagged_row_threshold']:.0%} of values missing",
                    f"Worst rows: {', '.join(str(r['row']) for r in mr['flagged_rows'][:5])}", "Consider dropping these rows. They are not dropped automatically.", "missing"))
    d = quality["duplicates"]
    if d["exact_duplicates"]:
        W.append(_w("WARNING", None, f"{d['exact_duplicates']} exact duplicate rows", f"Examples (row numbers): {d['example_rows'][:5]}", d["interpretation"], "duplicates"))
    elif d["excluding_strong_identifiers"]:
        W.append(_w("WARNING", None, f"{d['excluding_strong_identifiers']} duplicate rows once identifier columns are ignored",
                    f"Identifiers ignored: {', '.join(d['strong_identifiers_removed'])}", d["interpretation"], "duplicates"))
    for i in quality["identifiers"]:
        sev = "WARNING"
        W.append(_w(sev, i["column"], f"{i['column']} is a {i['label'].lower()}", "; ".join(i["evidence"]),
                    "Excluded from modelling by default. Keep it explicitly if it carries real signal.", "identifier"))
    for h in quality["high_cardinality"]:
        W.append(_w("WARNING", h["column"], f"{h['column']} is a high-cardinality categorical", f"{h['unique']} unique values ({h['unique_ratio']:.1%}).",
                    "Encoded by frequency, not one-hot. Consider target encoding, hashing or domain grouping.", "cardinality"))
    for k in quality["constants"]:
        if k["kind"] == "near_constant":
            W.append(_w("WARNING", k["column"], f"{k['column']} is near-constant", f"{k['top_share']:.1%} = '{k['top_value']}'", k["message"], "constant"))
        else:
            W.append(_w("WARNING", k["column"], f"{k['column']} is {k['kind'].replace('_', ' ')}", k["message"], "Excluded from modelling.", "constant"))
    for p in quality["placeholders"]:
        W.append(_w("INFO", p["column"], f"Placeholder values in {p['column']}", ", ".join(f"'{k}' ×{v}" for k, v in p["values"].items()),
                    "Converted to missing. " + p["note"], "placeholder"))
    for f in quality["formatting"]:
        W.append(_w("INFO", f["column"], f["message"], f"Examples: {', '.join(f['examples'])}", "Review formatting before use.", "formatting"))
    for m in profile["columns"]:
        if m["semantic_type"] == "datetime":
            W.append(_w("INFO", m["column"] if "column" in m else m["name"], "Datetime column detected",
                        "; ".join(m["evidence"]), "Can be used for time-series forecasting; otherwise expanded to calendar features.", "datetime"))
    if eda:
        for n in eda["numeric"]:
            if n.get("skew_class") and n["skew_class"].startswith("strongly"):
                W.append(_w("INFO", n["column"], f"{n['column']} is {n['skew_class']}", f"skewness = {n['skewness']:.2f}", "Consider a log1p transform for linear models.", "distribution"))
            o = n.get("outliers")
            if o and o["count"]:
                sev = "WARNING" if o["pct"] > C.OUTLIER_WARN_PCT else "INFO"
                W.append(_w(sev, n["column"], f"Potential outliers in {n['column']}", f"{o['count']} values ({o['pct']:.1f}%) outside [{o['lower_bound']:.3g}, {o['upper_bound']:.3g}]",
                            "Review whether they represent genuine observations before removing them.", "outliers"))
        for p in eda["correlation"]["high_pairs"]:
            sev = "WARNING" if abs(p["pearson"]) >= C.CORR_WARN else "INFO"
            W.append(_w(sev, f"{p['a']} / {p['b']}", f"{p['a']} and {p['b']} have correlation of {p['pearson']:.2f}", "Pearson correlation.", "Potential multicollinearity. Matters mainly for linear models.", "correlation"))
        t = eda.get("target")
        if t and t.get("problem_type") == "classification" and t.get("imbalanced"):
            W.append(_w("WARNING", t["target"], "Imbalanced target", f"Minority class is {t['distribution']['minority_pct']:.1f}% of rows.",
                        "Prefer PR-AUC, recall and F1 over accuracy. Class weights are applied; SMOTE is not used.", "target"))
        if t:
            for l in t["leakage_suspects"]:
                W.append(_w("HIGH", l["feature"], "Potential target leakage", l["reason"],
                            "Verify this feature is available at prediction time. Detection is heuristic and not exhaustive.", "leakage"))
    W.sort(key=lambda w: ORDER[w["severity"]])
    return W
