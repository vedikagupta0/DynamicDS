import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON } from "../lib/api";
import { useApp } from "../lib/store";
import { Tag } from "../components/ui";
import { fmt } from "../lib/format";

export default function ExperimentsPage() {
  const { setExpId } = useApp();
  const nav = useNavigate();
  const [list, setList] = useState<any[]>([]);
  useEffect(() => { getJSON("/experiments").then(setList); }, []);
  return (
    <div>
      <h1>Experiment history</h1>
      <div className="card tscroll">
        <table>
          <thead><tr><th>When</th><th>Dataset</th><th>Target</th><th>Task</th><th>Model / version</th><th>Leader</th><th>Test score</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {list.length ? list.map((e) => {
              const h = e.headline;
              return (
                <tr key={e.id}>
                  <td>{e.created_at.slice(0, 16).replace("T", " ")}</td><td>{e.dataset_name}</td><td>{e.target}</td><td>{e.task}</td>
                  <td>{e.model.name} {e.model.version}</td><td>{h ? h.leader : "–"}</td>
                  <td>{h ? `${h.primary_metric} ${fmt(h.test?.[h.primary_metric])}` : "–"}</td>
                  <td><Tag kind={e.status === "completed" ? "ok" : e.status === "failed" ? "bad" : ""}>{e.status}</Tag></td>
                  <td>{e.status === "completed" && <button className="btn ghost" onClick={() => { setExpId(e.id); nav("/results"); }}>Open</button>}</td>
                </tr>
              );
            }) : <tr><td colSpan={9}>No experiments yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
