import { useEffect, useState } from "react";
import { getJSON, postForm, postJSON, ApiError } from "../lib/api";
import { useApp } from "../lib/store";
import { Bar } from "../components/ui";
import { LineSeriesChart } from "../components/charts";
import { fmt } from "../lib/format";

interface ModelRow { id: string; name: string; version: string; status: string }
interface InputField { name: string; kind: string; options?: string[]; min?: number; median?: number; max?: number; example?: string }

export default function PredictPage() {
  const { expId, setExpId } = useApp();
  const [models, setModels] = useState<ModelRow[]>([]);
  const [current, setCurrent] = useState<string>(expId || "");
  const [schema, setSchema] = useState<any>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [predOut, setPredOut] = useState<{ prediction: any; confidence?: number; probabilities?: Record<string, number>; explanation?: any; unparseable: string[] } | null>(null);
  const [predErr, setPredErr] = useState("");
  const [periods, setPeriods] = useState(12);
  const [fcOut, setFcOut] = useState<{ t: string[]; yhat: number[]; model: string; note: string } | null>(null);
  const [fcErr, setFcErr] = useState("");
  const [batchMsg, setBatchMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => { getJSON("/models").then((all: ModelRow[]) => { setModels(all); if (!current && all[0]) setCurrent(all[0].id); }); }, []);
  useEffect(() => { if (current) { setExpId(current); getJSON(`/models/${current}/schema`).then(setSchema); setPredOut(null); setFcOut(null); } }, [current]);

  if (!models.length) return <div className="note">Train a model first.</div>;
  if (!schema) return <p className="muted">Loading…</p>;

  if (schema.kind === "forecast") {
    const forecast = async () => {
      setFcErr(""); setFcOut(null);
      try { setFcOut(await postJSON("/forecast", { experiment_id: current, periods })); }
      catch (e) { setFcErr((e as ApiError).message); }
    };
    return (
      <div>
        <h1>Prediction</h1>
        <label htmlFor="ms">Model</label>
        <select id="ms" value={current} onChange={(e) => setCurrent(e.target.value)}>
          {models.map((m) => <option key={m.id} value={m.id}>{m.name} {m.version} ({m.status})</option>)}
        </select>
        <div className="card">
          <label htmlFor="np">Periods to forecast</label>
          <input type="number" id="np" value={periods} min={1} max={1000} onChange={(e) => setPeriods(+e.target.value)} />{" "}
          <button className="btn" onClick={forecast}>Forecast</button>
          {fcErr && <div className="note bad">{fcErr}</div>}
          {fcOut && (
            <div>
              <p className="small muted">{fcOut.note}</p>
              <LineSeriesChart height={180} series={[{ name: fcOut.model, data: fcOut.t.map((t, i) => ({ t, y: fcOut.yhat[i] })) }]} />
              <div className="tscroll"><table><thead><tr><th>Period</th><th>Forecast</th></tr></thead>
                <tbody>{fcOut.t.map((t, i) => <tr key={t}><td>{t.slice(0, 10)}</td><td>{fmt(fcOut.yhat[i], 6)}</td></tr>)}</tbody>
              </table></div>
            </div>
          )}
        </div>
      </div>
    );
  }

  const predict = async () => {
    setPredErr(""); setPredOut(null);
    try {
      const r = await postJSON(`/models/${current}/predict`, { records: [values], explain: true });
      const p = r.predictions[0];
      setPredOut({ prediction: p.prediction, confidence: p.confidence, probabilities: p.probabilities, explanation: r.explanation, unparseable: r.unparseable_columns });
    } catch (e) { setPredErr((e as ApiError).message); }
  };
  const batch = async (f: File) => {
    setBatchMsg(null);
    try {
      const fd = new FormData(); fd.append("file", f);
      const r = await postForm(`/models/${current}/batch-predict`, fd);
      const miss = r.headers.get("X-Missing-Columns");
      const blob = await r.blob();
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "predictions.csv"; a.click();
      setBatchMsg({ ok: true, text: "Downloaded predictions." + (miss ? " Columns not found in your file: " + miss : "") });
    } catch (e) { setBatchMsg({ ok: false, text: (e as ApiError).message }); }
  };

  return (
    <div>
      <h1>Prediction</h1>
      <label htmlFor="ms">Model</label>
      <select id="ms" value={current} onChange={(e) => setCurrent(e.target.value)}>
        {models.map((m) => <option key={m.id} value={m.id}>{m.name} {m.version} ({m.status})</option>)}
      </select>
      <div className="card">
        <h3>Single prediction</h3>
        <p className="small muted">Leave a field blank if the value is unknown; it is treated as missing, exactly as in training.</p>
        <div className="grid">
          {schema.inputs.map((f: InputField) => (
            <div key={f.name}>
              <label htmlFor={`f_${f.name}`}>{f.name}</label>
              {f.kind === "category" || f.kind === "bool" ? (
                <>
                  <input type="text" list={`dl_${f.name}`} id={`f_${f.name}`} value={values[f.name] || ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
                  <datalist id={`dl_${f.name}`}>{(f.options || []).map((o) => <option key={o} value={o} />)}</datalist>
                </>
              ) : (
                <input type="text" id={`f_${f.name}`} value={values[f.name] || ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                  placeholder={f.kind === "numeric" ? "e.g. " + fmt(f.median) : f.kind === "datetime" ? (f.example || "").slice(0, 19) : ""} />
              )}
            </div>
          ))}
        </div>
        <p><button className="btn" onClick={predict}>Predict</button></p>
        {predErr && <div className="note bad">{predErr}</div>}
        {predOut && (
          <div>
            {schema.problem_type === "classification" ? (
              <>
                <div className="big">{predOut.prediction}</div>
                <p>Confidence {fmt(predOut.confidence, 3)}</p>
                <Bar items={Object.entries(predOut.probabilities!).map(([k, v]) => [k, v])} max={1} fmtv={(v) => v.toFixed(3)} />
              </>
            ) : <div className="big">{fmt(predOut.prediction, 6)}</div>}
            {predOut.unparseable.length > 0 && <div className="note bad">Could not parse: {predOut.unparseable.join(", ")}. Treated as missing.</div>}
            {predOut.explanation && (
              <>
                <h3>Top contributing features</h3>
                <p className="small muted">{predOut.explanation.note}</p>
                <Bar items={predOut.explanation.top_features.map((f: any) => [f.feature + (f.value != null ? " = " + f.value : ""), f.contribution])} fmtv={(v) => (v > 0 ? "+" : "") + v.toFixed(3)} />
              </>
            )}
          </div>
        )}
      </div>
      <div className="card">
        <h3>Batch prediction</h3>
        <p className="small muted">Upload a CSV/XLSX containing the model's input columns. Columns that are missing are treated as missing values, and the response reports which.</p>
        <input type="file" accept=".csv,.xlsx" onChange={(e) => e.target.files?.[0] && batch(e.target.files[0])} />
        {batchMsg && <div className={`note ${batchMsg.ok ? "good" : "bad"}`}>{batchMsg.text}</div>}
      </div>
    </div>
  );
}
