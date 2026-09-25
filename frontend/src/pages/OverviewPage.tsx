import { useNavigate } from "react-router-dom";
import { useApp } from "../lib/store";
import { Stat, Strip, Tag } from "../components/ui";
import { fmt, pct, COLORS } from "../lib/format";

export default function OverviewPage() {
  const { prof } = useApp();
  const nav = useNavigate();
  if (!prof) return <p className="muted">No dataset loaded.</p>;
  const o = prof.overview;
  return (
    <div>
      <h1>{prof.filename}</h1>
      <p className="muted small">SHA-256 {prof.sha256.slice(0, 16)}… · dataset id {prof.id}</p>
      <div className="card">
        <div className="row">
          <Stat label="rows" value={fmt(o.rows)} /><Stat label="columns" value={o.columns} />
          <Stat label="missing cells" value={pct(o.missing_cell_pct)} /><Stat label="duplicate rows" value={fmt(o.duplicate_rows)} />
          <Stat label="memory" value={o.memory_mb + " MB"} /><Stat label="file size" value={(o.file_size_bytes / 1e6).toFixed(2) + " MB"} />
        </div>
        <Strip counts={o.type_counts} />
      </div>
      <h2>Columns</h2>
      <p className="muted small">Types are inferred from values, not just pandas dtypes. Identifier and text columns are excluded from modelling by default, and every exclusion is listed later.</p>
      <div className="card tscroll">
        <table>
          <thead><tr><th>Column</th><th>Semantic type</th><th>Unique</th><th>Evidence</th></tr></thead>
          <tbody>
            {prof.columns.map((c) => (
              <tr key={c.name}>
                <td><b>{c.name}</b></td>
                <td>
                  <span className="tag" style={{ background: COLORS[c.semantic_type] + "22", color: COLORS[c.semantic_type] }}>{c.semantic_type}</span>
                  {c.identifier_level && <Tag kind="bad">{c.identifier_level} identifier</Tag>}
                  {c.high_cardinality && <Tag kind="warn">high-cardinality</Tag>}
                  {c.constant && <Tag kind="bad">constant</Tag>}
                </td>
                <td>{fmt(c.n_unique)} <span className="muted small">({pct(c.unique_ratio * 100)})</span></td>
                <td className="small muted">{c.evidence.join("; ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p><button className="btn" onClick={() => nav("/quality")}>Review data quality</button></p>
    </div>
  );
}
