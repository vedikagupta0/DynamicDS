import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON, postForm, ApiError } from "../lib/api";
import { useApp, type Profile } from "../lib/store";
import { fmt } from "../lib/format";

interface DsRow { id: string; filename: string; rows: number; columns: number; uploaded_at: string }

export default function UploadPage() {
  const { setDataset } = useApp();
  const [list, setList] = useState<DsRow[]>([]);
  const [msg, setMsg] = useState("");
  const [over, setOver] = useState(false);
  const nav = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { getJSON("/datasets").then(setList).catch(() => {}); }, []);

  const upload = async (f: File) => {
    setMsg("Uploading and profiling…");
    try {
      const fd = new FormData();
      fd.append("file", f);
      const r = await postForm("/datasets/upload", fd);
      const j = await r.json();
      await openDataset(j.id);
    } catch (e) {
      setMsg((e as ApiError).message);
    }
  };
  const openDataset = async (id: string) => {
    const prof: Profile = await getJSON(`/datasets/${id}/profile`);
    setDataset(id, prof);
    nav("/overview");
  };

  return (
    <div>
      <h1>Upload a dataset</h1>
      <p className="muted">CSV or XLSX. Nothing is trained until you review the profile and choose a task.</p>
      <div
        className={`drop ${over ? "over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); }}
      >
        <p><b>Drop a file here</b> or</p>
        <input ref={fileRef} type="file" accept=".csv,.xlsx" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        <p className="small muted">Limits are set by the server (default 50 MB, 500,000 rows, 300 columns).</p>
        <p className={msg.toLowerCase().includes("could not") || msg.toLowerCase().includes("exceed") ? "err" : ""}>{msg}</p>
      </div>
      <h2>Recent datasets</h2>
      {list.length ? (
        <div className="card tscroll">
          <table>
            <thead><tr><th>File</th><th>Rows</th><th>Columns</th><th>Uploaded</th><th></th></tr></thead>
            <tbody>
              {list.slice(0, 10).map((d) => (
                <tr key={d.id}>
                  <td>{d.filename}</td><td>{fmt(d.rows)}</td><td>{d.columns}</td>
                  <td>{d.uploaded_at.slice(0, 16).replace("T", " ")}</td>
                  <td><button className="btn ghost" onClick={() => openDataset(d.id)}>Open</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="muted">No datasets yet.</p>}
    </div>
  );
}
