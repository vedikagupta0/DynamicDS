"""Self-contained HTML report (print to PDF from the browser). All dynamic text is HTML-escaped."""
from __future__ import annotations

from html import escape as e

CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1d2733}
h1{font-size:1.6rem}h2{margin-top:2.2rem;border-bottom:1px solid #d5dbe1;padding-bottom:.3rem}h3{margin:1.2rem 0 .3rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:13px}th,td{border:1px solid #d5dbe1;padding:4px 8px;text-align:left}th{background:#eef2f5}
.HIGH{color:#a12a3a;font-weight:600}.WARNING{color:#94640f;font-weight:600}.INFO{color:#4a5a6a}.muted{color:#5c6b7a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}svg{max-width:100%}
@media print{body{max-width:none}h2{break-after:avoid}}"""


def _f(v, d=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{d}g}"
    return e(str(v))


def _table(head, rows):
    return "<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{c if isinstance(c, _Raw) else _f(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


class _Raw(str):
    pass


def svg_hist(counts, w=270, h=90):
    m = max(counts) or 1
    bw = w / len(counts)
    bars = "".join(f'<rect x="{i*bw:.1f}" y="{h-c/m*h:.1f}" width="{bw-1:.1f}" height="{c/m*h:.1f}" fill="#3b7ea1"/>' for i, c in enumerate(counts))
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{bars}</svg>'


def svg_lines(series, w=700, h=220, colors=("#3b7ea1", "#c0682c", "#6a8f3a", "#8a4f9e")):
    ys = [v for s in series for v in s if v is not None]
    if not ys:
        return ""
    lo, hi = min(ys), max(ys)
    rng = (hi - lo) or 1
    paths = ""
    for k, s in enumerate(series):
        n = max(len(s) - 1, 1)
        pts = " ".join(f"{i/n*w:.1f},{h-(v-lo)/rng*h:.1f}" for i, v in enumerate(s) if v is not None)
        paths += f'<polyline fill="none" stroke="{colors[k % 4]}" stroke-width="1.5" points="{pts}"/>'
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="border:1px solid #d5dbe1">{paths}</svg>'


def svg_bars(items, w=600):
    if not items:
        return ""
    m = max(abs(v) for _, v in items) or 1
    rows = "".join(f'<text x="0" y="{i*20+14}" font-size="12">{e(str(k))[:28]}</text><rect x="190" y="{i*20+3}" width="{abs(v)/m*380:.1f}" height="14" fill="#3b7ea1"/>'
                   f'<text x="{195+abs(v)/m*380:.1f}" y="{i*20+14}" font-size="11">{v:.3g}</text>' for i, (k, v) in enumerate(items))
    return f'<svg viewBox="0 0 {w+60} {len(items)*20+4}" width="{w+60}">{rows}</svg>'


def build_report(meta: dict, warnings: list[dict], eda: dict, exp: dict | None = None) -> str:
    ov, q = meta["profile"]["overview"], meta["quality"]
    h = [f"<h1>AutoDS report: {e(meta['filename'])}</h1><p class='muted'>Dataset id {e(meta['id'])} · SHA-256 {e(meta['sha256'][:16])}… · uploaded {e(meta['uploaded_at'][:19])}</p>"]
    tc = ov["type_counts"]
    h.append("<h2>Dataset summary</h2>" + _table(["Rows", "Columns", "Memory (MB)", "Missing cells", "Duplicate rows"],
             [[ov["rows"], ov["columns"], ov["memory_mb"], f"{ov['missing_cell_pct']}%", ov["duplicate_rows"]]])
             + _table(list(tc), [list(tc.values())]))
    h.append("<h2>Warnings</h2>" + (_table(["Severity", "Column", "Issue", "Evidence", "Recommendation"],
             [[_Raw(f"<span class='{w['severity']}'>{w['severity']}</span>"), w["column"], w["issue"], w["evidence"], w["recommendation"]] for w in warnings]) if warnings else "<p>No warnings.</p>"))
    d = q["duplicates"]
    h.append("<h2>Data quality</h2><h3>Missing values</h3>" + _table(["Column", "Missing", "%", "Level"],
             [[c["column"], c["missing"], c["missing_pct"], c["label"]] for c in q["missing"]["columns"] if c["missing"]]) or "")
    h.append(f"<h3>Duplicates</h3><p>Exact: {d['exact_duplicates']} · excluding strong identifiers: {_f(d['excluding_strong_identifiers'])}. {e(d['interpretation'])}</p>")
    if q["identifiers"]:
        h.append("<h3>Identifiers</h3>" + _table(["Column", "Class", "Unique ratio", "Evidence"], [[i["column"], i["label"], i["unique_ratio"], "; ".join(i["evidence"])] for i in q["identifiers"]]))
    if q["placeholders"]:
        h.append("<h3>Placeholders converted to missing</h3>" + _table(["Column", "Count", "Values"], [[p["column"], p["count"], ", ".join(p["values"])] for p in q["placeholders"]]))
    h.append("<h2>Exploratory analysis</h2><p class='muted'>" + e(eda["sampling"]["note"]) + "</p>")
    if eda["numeric"]:
        h.append("<h3>Numerical columns</h3>" + _table(["Column", "Mean", "Median", "Std", "Min", "Max", "Skew", "Shape", "Outliers %"],
                 [[n["column"], n.get("mean"), n.get("median"), n.get("std"), n.get("min"), n.get("max"), n.get("skewness"), n.get("skew_class"),
                   n.get("outliers", {}).get("pct")] for n in eda["numeric"]]))
        h.append("<div class='grid'>" + "".join(f"<div><b>{e(n['column'])}</b><br>{svg_hist(n['histogram']['counts'])}</div>" for n in eda["numeric"][:12] if "histogram" in n) + "</div>")
    if eda["categorical"]:
        h.append("<h3>Categorical columns</h3>" + _table(["Column", "Unique", "Top value", "Top %", "Cardinality"],
                 [[c["column"], c["unique"], c.get("most_frequent"), c.get("frequency_pct"), c.get("cardinality_class")] for c in eda["categorical"]]))
    hp = eda["correlation"]["high_pairs"]
    if hp:
        h.append("<h3>High correlations</h3>" + _table(["A", "B", "Pearson"], [[p["a"], p["b"], p["pearson"]] for p in hp[:15]]))
    t = eda.get("target")
    if t and t.get("associations"):
        h.append(f"<h2>Target analysis: {e(t['target'])}</h2><p>{e(t['recommendation']['reason'])}</p>" +
                 svg_bars([(a["feature"], a["strength"]) for a in t["associations"][:12]]) + f"<p class='muted'>{e(t.get('association_note',''))}</p>")
    if exp and exp.get("results"):
        r = exp["results"]
        h.append(f"<h2>Model report ({e(exp['model']['name'])} {e(exp['model']['version'])})</h2>")
        cfg = {k: v for k, v in exp["config"].items() if v not in (None, [], "")}
        h.append("<h3>Configuration</h3>" + _table(["Setting", "Value"], [[k, str(v)] for k, v in cfg.items()]))
        if r.get("decisions"):
            h.append("<h3>Preprocessing decisions</h3>" + _table(["Column", "Action", "Reason"], [[x["column"], x["action"], x["reason"]] for x in r["decisions"]]))
        if r.get("preprocessing"):
            h.append("<h3>Preprocessing</h3>" + _table(["Step", "Detail"], [[k, str(v)] for k, v in r["preprocessing"].items()]))
        pm = r["primary_metric"]
        if r["mode"] == "tabular":
            h.append(f"<h3>Model comparison (primary metric: {e(pm)})</h3>" + _table(["Model", "CV mean", "CV std", "Test " + pm, "Train time (s)", "Status", "Leader"],
                     [[m["name"], m.get("primary_cv"), m.get("primary_cv_std"), m.get("primary_test"), m.get("train_time_s"), m["status"], "yes" if m.get("leader") else ""] for m in r["models"]]))
            lead = next(m for m in r["models"] if m.get("leader"))
            h.append("<h3>Leading model: test metrics</h3>" + _table(list(lead["test"]), [list(lead["test"].values())]))
            dg = lead["diagnostics"]
            if "confusion_matrix" in dg:
                h.append("<h3>Confusion matrix</h3>" + _table(["actual \\ predicted"] + dg["labels"], [[l] + row for l, row in zip(dg["labels"], dg["confusion_matrix"])]))
            elif "residuals" in dg:
                h.append("<h3>Error distribution</h3>" + svg_hist(dg["error_hist"]["counts"], 500, 120))
            xai = r.get("explainability") or {}
            if xai.get("permutation_importance"):
                h.append("<h3>Permutation importance</h3>" + svg_bars([(x["feature"], x["importance"]) for x in xai["permutation_importance"][:12]]))
        else:
            h.append(f"<h3>Split</h3><p>{e(r['split']['explanation'])}</p>" + _table(["Window", "Start", "End", "Periods"],
                     [[k, r["split"][k]["start"][:10], r["split"][k]["end"][:10], r["split"][k]["n"]] for k in ("train", "validation", "test")]))
            h.append(f"<h3>Model comparison (primary metric: {e(pm)})</h3>" + _table(["Model", "Validation", "Test", "vs best baseline %", "Leader"],
                     [[m["name"], m.get("primary_val"), m.get("primary_test"), m.get("improvement_vs_best_baseline_pct"), "yes" if m.get("leader") else ""] for m in r["models"] if m["status"] == "ok"]))
            ch = r["charts"]
            names = list(ch["test"]["pred"])
            h.append("<h3>Test window: actual vs forecast</h3>" + svg_lines([ch["test"]["actual"]] + [ch["test"]["pred"][k] for k in names]) +
                     "<p class='muted'>Series order: actual, " + ", ".join(e(k) for k in names) + "</p>")
            h.append("<h3>Future forecast</h3>" + svg_lines([ch["future"]["yhat"]]))
        h.append(f"<p><b>{e(r['baseline_note'])}</b></p>")
    h.append("<h2>Limitations</h2><ul><li>Automatic detection of identifiers, leakage and types is heuristic; review each flagged item.</li>"
             "<li>Baseline models are trained with default hyper-parameters and are not tuned.</li>"
             "<li>Feature importance and SHAP describe model behaviour, not causal effects.</li>"
             "<li>Performance on a single hold-out split can vary; small datasets give unstable estimates.</li>"
             "<li>Forecasts are point forecasts without prediction intervals.</li></ul>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Dynamic DS report</title><style>{CSS}</style></head><body>{''.join(h)}</body></html>"
