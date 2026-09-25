import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON } from "../lib/api";
import { useApp } from "../lib/store";
import { Tag } from "../components/ui";
import { fmt, pct } from "../lib/format";

interface Quality {
  missing: { columns: { column: string; missing: number; missing_pct: number; level: string; label: string; recommendation: string | null }[]; flagged_row_count: number; flagged_row_threshold: number; flagged_rows: { row: number; missing: number; total: number }[] };
  duplicates: { exact_duplicates: number; excluding_strong_identifiers: number | null; strong_identifiers_removed: string[]; interpretation: string; action: string };
  identifiers: { column: string; label: string; unique_ratio: number; evidence: string[] }[];
  high_cardinality: { column: string; label: string; unique_ratio: number; unique: number }[];
  constants: { column: string; kind: string; top_share?: number; top_value?: string; message: string; action: string }[];
  placeholders: { column: string; values: Record<string, number>; action: string; note: string }[];
  formatting: { column: string; kind: string; message: string; examples: string[] }[];
}
const sevTag = (l: string) => (({ none: "ok", low: "ok", moderate: "warn", high: "warn", very_high: "bad" } as Record<string, "ok" | "warn" | "bad">)[l] ?? "");

export default function QualityPage() {
  const { ds } = useApp();
  const nav = useNavigate();
  const [q, setQ] = useState<Quality | null>(null);
  useEffect(() => { if (ds) getJSON(`/datasets/${ds}/quality`).then(setQ); }, [ds]);
  if (!q) return <p className="muted">Loading…</p>;
  const m = q.missing, d = q.duplicates;
  return (
    <div>
      <h1>Data quality</h1>
      <p className="muted">Detected issue, recommendation and action are kept separate. Nothing here changes your data except converting placeholders to missing, which is listed below.</p>
      <h2>Duplicates</h2>
      <div className="card">
        <div className="row">
          <div className="stat"><b>{fmt(d.exact_duplicates)}</b><span className="muted small">exact duplicate rows</span></div>
          <div className="stat"><b>{fmt(d.excluding_strong_identifiers)}</b><span className="muted small">excluding strong identifiers</span></div>
        </div>
        <p>{d.interpretation} <span className="muted">Action: {d.action}</span></p>
        {d.strong_identifiers_removed.length > 0 && <p className="small muted">Identifiers ignored for the second count: {d.strong_identifiers_removed.join(", ")}</p>}
      </div>
      <h2>Missing values</h2>
      <div className="card tscroll">
        <table>
          <thead><tr><th>Column</th><th>Missing</th><th>%</th><th></th><th>Level</th><th>Recommendation</th></tr></thead>
          <tbody>
            {[...m.columns].filter((c) => c.missing).sort((a, b) => b.missing_pct - a.missing_pct).map((c) => (
              <tr key={c.column}>
                <td>{c.column}</td><td>{fmt(c.missing)}</td><td>{pct(c.missing_pct)}</td>
                <td style={{ width: 120 }}><div className="bar"><i style={{ width: `${c.missing_pct}%` }} /></div></td>
                <td><Tag kind={sevTag(c.level)}>{c.label}</Tag></td>
                <td className="small">{c.recommendation || ""}</td>
              </tr>
            ))}
            {m.columns.every((c) => !c.missing) && <tr><td colSpan={6}>No missing values.</td></tr>}
          </tbody>
        </table>
      </div>
      {m.flagged_row_count > 0 && (
        <div className="note">
          {m.flagged_row_count} rows have at least {m.flagged_row_threshold * 100}% of their values missing
          (worst: {m.flagged_rows.slice(0, 5).map((r) => `row ${r.row} (${r.missing}/${r.total})`).join(", ")}).
          Consider dropping them; they are not dropped automatically.
        </div>
      )}
      <h2>Identifiers and cardinality</h2>
      <div className="card tscroll">
        <table>
          <thead><tr><th>Column</th><th>Class</th><th>Unique</th><th>Evidence</th></tr></thead>
          <tbody>
            {q.identifiers.map((i) => (
              <tr key={i.column}><td>{i.column}</td><td><Tag kind="bad">{i.label}</Tag></td><td>{pct(i.unique_ratio * 100)}</td><td className="small">{i.evidence.join("; ")}</td></tr>
            ))}
            {q.high_cardinality.map((i) => (
              <tr key={i.column}><td>{i.column}</td><td><Tag kind="warn">{i.label}</Tag></td><td>{pct(i.unique_ratio * 100)} ({i.unique})</td><td className="small">Not an identifier by the rules; kept as a feature.</td></tr>
            ))}
            {!q.identifiers.length && !q.high_cardinality.length && <tr><td colSpan={4}>None detected.</td></tr>}
          </tbody>
        </table>
      </div>
      <h2>Constant and near-constant columns</h2>
      <div className="card">
        {q.constants.length ? q.constants.map((k) => (
          <p key={k.column}>{k.column}: <Tag kind={k.kind === "near_constant" ? "warn" : "bad"}>{k.kind.replace("_", " ")}</Tag>
            {k.top_share ? ` ${pct(k.top_share * 100)} = “${k.top_value}”. ` : " "}{k.message} <span className="muted">Action: {k.action}</span></p>
        )) : "None detected."}
      </div>
      <h2>Placeholder values</h2>
      <div className="card">
        {q.placeholders.length ? q.placeholders.map((p) => (
          <p key={p.column}><b>{p.column}</b> → {Object.entries(p.values).map(([k, v]) => `“${k}” ×${v}`).join(", ")} <span className="muted">{p.action}. {p.note}</span></p>
        )) : "None detected."}
      </div>
      <h2>Formatting</h2>
      <div className="card">
        {q.formatting.length ? q.formatting.map((f, i) => (
          <p key={i}><b>{f.column}</b> <Tag>{f.kind.replace(/_/g, " ")}</Tag> {f.message} <span className="muted small">e.g. {f.examples.join(", ")}</span></p>
        )) : "Nothing unusual."}
      </div>
      <p><button className="btn" onClick={() => nav("/eda")}>Continue to EDA</button></p>
    </div>
  );
}
