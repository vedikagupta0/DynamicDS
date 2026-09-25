import { useEffect, useState } from "react";
import { getJSON, patchJSON } from "../lib/api";
import { fmt } from "../lib/format";

export default function RegistryPage() {
  const [list, setList] = useState<any[]>([]);
  const [statuses, setStatuses] = useState<string[]>([]);
  const load = () => { getJSON("/models").then(setList); getJSON("/config").then((c) => setStatuses(c.model_statuses)); };
  useEffect(load, []);
  const change = async (id: string, status: string) => { await patchJSON(`/models/${id}/status`, { status }); load(); };
  return (
    <div>
      <h1>Model registry</h1>
      <p className="muted small">One Production version per model name; promoting a version archives the previous one.</p>
      <div className="card tscroll">
        <table>
          <thead><tr><th>Name</th><th>Version</th><th>Task</th><th>Dataset</th><th>Metric (test)</th><th>Created</th><th>Status</th></tr></thead>
          <tbody>
            {list.length ? list.map((m) => (
              <tr key={m.id}>
                <td>{m.name}</td><td>{m.version}</td><td>{m.task}</td><td>{m.dataset}</td>
                <td>{m.headline ? `${m.headline.primary_metric} ${fmt(m.headline.test?.[m.headline.primary_metric])}` : "–"}</td>
                <td>{m.created_at.slice(0, 16).replace("T", " ")}</td>
                <td><select value={m.status} onChange={(e) => change(m.id, e.target.value)}>{statuses.map((s) => <option key={s}>{s}</option>)}</select></td>
              </tr>
            )) : <tr><td colSpan={7}>No trained models yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
