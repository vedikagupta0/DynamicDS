import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON } from "../lib/api";
import { useApp } from "../lib/store";
import { Ledger, Tag } from "../components/ui";

interface W { severity: "HIGH" | "WARNING" | "INFO"; column: string | null; issue: string; evidence: string; recommendation: string }

export default function WarningsPage() {
  const { ds, target } = useApp();
  const nav = useNavigate();
  const [w, setW] = useState<W[] | null>(null);
  useEffect(() => {
    if (!ds) return;
    const q = target ? `?target=${encodeURIComponent(target)}` : "";
    getJSON(`/datasets/${ds}/warnings${q}`).then(setW);
  }, [ds, target]);
  if (!w) return <p className="muted">Loading…</p>;
  const n = (s: string) => w.filter((x) => x.severity === s).length;
  return (
    <div>
      <h1>Automated warnings</h1>
      <div className="row">
        <Tag kind="bad">{n("HIGH")} high</Tag><Tag kind="warn">{n("WARNING")} warning</Tag><Tag>{n("INFO")} info</Tag>
        <span className="muted small">{target ? `Includes target checks for "${target}"` : "Select a target on the EDA → Target tab to add target and leakage checks."}</span>
      </div>
      {w.length ? w.map((x, i) => (
        <Ledger key={i} severity={x.severity}>
          {x.column && <Tag>{x.column}</Tag>}
          <b>{x.issue}</b><br />
          <span className="small"><b>Evidence:</b> {x.evidence}</span><br />
          <span className="small"><b>Recommendation:</b> {x.recommendation}</span>
        </Ledger>
      )) : <p>No warnings.</p>}
      <p><button className="btn" onClick={() => nav("/model")}>Choose a prediction task</button></p>
    </div>
  );
}
