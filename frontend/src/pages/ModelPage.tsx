import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON, postJSON, postEmpty, ApiError } from "../lib/api";
import { useApp } from "../lib/store";
import { Seg, Tag } from "../components/ui";

const M_STD = ["auto", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"];
const M_REG = ["auto", "rmse", "mae", "r2"];
const M_FC = ["mae", "rmse", "smape", "mape"];

export default function ModelPage() {
  const { ds, prof, mode, setMode, target, setTarget, setExpId } = useApp();
  const nav = useNavigate();
  const [pt, setPt] = useState<"" | "classification" | "regression">("");
  const [pm, setPm] = useState("auto");
  const [horizon, setHorizon] = useState(12);
  const [dtCol, setDtCol] = useState("");
  const [rec, setRec] = useState<{ problem_type: string | null; reason: string } | null>(null);
  const [keep, setKeep] = useState<Set<string>>(new Set());
  const [training, setTraining] = useState(false);
  const [err, setErr] = useState("");

  const [showParams, setShowParams] = useState(false);
  const [logregC, setLogregC] = useState(1.0);
  const [rfTrees, setRfTrees] = useState(200);
  const [rfDepth, setRfDepth] = useState(14);
  const [xgbTrees, setXgbTrees] = useState(250);
  const [xgbDepth, setXgbDepth] = useState(5);
  const [xgbLr, setXgbLr] = useState(0.08);

  const P = prof?.columns ?? [];
  const dts = P.filter((c) => c.semantic_type === "datetime");

  useEffect(() => {
    if (!ds || !target || mode !== "standard") { setRec(null); return; }
    getJSON(`/datasets/${ds}/eda?target=${encodeURIComponent(target)}`).then((e) => {
      setRec(e.target.recommendation);
      if (e.target.recommendation.problem_type) setPt(e.target.recommendation.problem_type);
    });
  }, [ds, target, mode]);

  useEffect(() => { setPm(mode === "forecast" ? "mae" : "auto"); }, [pt, mode]);

  if (!prof) return null;

  const excluded = P.filter((x) => x.name !== target && (x.identifier_level || x.constant || x.semantic_type === "text" || x.semantic_type === "empty"));

  const train = async () => {
    setErr(""); setTraining(true);
    try {
      const model_params = {
        logreg: { C: logregC },
        rf: { n_estimators: rfTrees, max_depth: rfDepth },
        xgb: { n_estimators: xgbTrees, max_depth: xgbDepth, learning_rate: xgbLr }
      };
      const body: Record<string, unknown> = mode === "forecast"
        ? { dataset_id: ds, mode: "forecast", target, datetime_column: dtCol, horizon, primary_metric: pm }
        : { dataset_id: ds, mode: "standard", target, problem_type: pt, primary_metric: pm === "auto" ? null : pm, keep_columns: [...keep], model_params };
      const exp = await postJSON("/experiments", body);
      await postEmpty(`/experiments/${exp.id}/train`);
      setExpId(exp.id);
      for (let i = 0; i < 120; i++) {
        await new Promise((r) => setTimeout(r, 1200));
        const st = await getJSON(`/experiments/${exp.id}`);
        if (st.status === "completed") { nav("/results"); return; }
        if (st.status === "failed") { setErr(`Training failed. ${st.error}`); setTraining(false); return; }
      }
      setErr("Training is taking longer than expected.");
      setTraining(false);
    } catch (e) {
      setErr((e as ApiError).message);
      setTraining(false);
    }
  };

  return (
    <div>
      <h1>Choose a prediction task</h1>
      <Seg options={[["standard", "Standard ML"], ["forecast", "Time-series forecasting"]]} value={mode} onChange={(m) => { setMode(m); setErr(""); }} />
      <div className="card">
        {mode === "forecast" ? (
          !dts.length ? <p className="err">No datetime column was detected in this dataset, so forecasting is unavailable.</p> : (
            <>
              <label htmlFor="dt">Datetime column</label>
              <select id="dt" value={dtCol} onChange={(e) => setDtCol(e.target.value)}>
                <option value="">Select…</option>
                {dts.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              <label htmlFor="tg">Target column (numerical)</label>
              <select id="tg" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">Select…</option>
                {P.filter((c) => c.semantic_type === "numerical" && !c.identifier_level).map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              <label htmlFor="hz">Forecast horizon (periods)</label>
              <input type="number" id="hz" value={horizon} min={1} max={1000} onChange={(e) => setHorizon(+e.target.value)} />
              <label htmlFor="pm">Primary metric</label>
              <select id="pm" value={pm} onChange={(e) => setPm(e.target.value)}>
                {M_FC.map((m) => <option key={m}>{m}</option>)}
              </select>
              <div className="note">Time-aware validation is used to prevent future observations from leaking into model training. Data is sorted by time and split 70% train, 15% validation, 15% test. There is never a random split. Naive and seasonal-naive baselines are always included.</div>
              <button className="btn" disabled={!dtCol || !target || training} onClick={train}>{training ? "Training…" : "Train baselines"}</button>
            </>
          )
        ) : (
          <>
            <label htmlFor="tg">Target column</label>
            <select id="tg" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Select…</option>
              {P.filter((c) => !["identifier", "text", "empty", "datetime"].includes(c.semantic_type)).map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
            {rec && (rec.problem_type ? (
              <div className="note">Recommended: <b>{rec.problem_type}</b>. {rec.reason} Confirm or change below.</div>
            ) : <div className="note bad">{rec.reason}</div>)}
            <label>Problem type</label>
            <label className="inline"><input type="radio" name="pt" checked={pt === "classification"} onChange={() => setPt("classification")} />Classification</label>
            <label className="inline"><input type="radio" name="pt" checked={pt === "regression"} onChange={() => setPt("regression")} />Regression</label>
            <label htmlFor="pmS">Primary metric</label>
            <select id="pmS" value={pm} onChange={(e) => setPm(e.target.value)}>
              {(pt === "regression" ? M_REG : M_STD).map((m) => <option key={m}>{m}</option>)}
            </select>
            <p className="small muted">"auto" picks PR-AUC for imbalanced binary targets, F1 for other classification, and RMSE for regression. There is no single best metric, so compare all columns in the results.</p>

            <div style={{ marginTop: "1rem", marginBottom: "1rem" }}>
              <button type="button" className="btn secondary" onClick={() => setShowParams(!showParams)}>
                {showParams ? "Hide Model Hyperparameters" : "⚙️ Customize Model Hyperparameters"}
              </button>
            </div>

            {showParams && (
              <div className="card" style={{ background: "var(--surface)", border: "1px solid var(--border)", marginBottom: "1rem" }}>
                <h3>Model Hyperparameters</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
                  {pt === "classification" && (
                    <div>
                      <h4>Logistic Regression</h4>
                      <label htmlFor="logregC">Regularization Strength (C)</label>
                      <input type="number" id="logregC" step="0.1" value={logregC} min={0.01} max={100} onChange={(e) => setLogregC(+e.target.value)} />
                    </div>
                  )}
                  <div>
                    <h4>Random Forest</h4>
                    <label htmlFor="rfTrees">Trees (n_estimators)</label>
                    <input type="number" id="rfTrees" value={rfTrees} min={10} max={1000} onChange={(e) => setRfTrees(+e.target.value)} />
                    <label htmlFor="rfDepth">Max Depth</label>
                    <input type="number" id="rfDepth" value={rfDepth} min={1} max={50} onChange={(e) => setRfDepth(+e.target.value)} />
                  </div>
                  <div>
                    <h4>XGBoost</h4>
                    <label htmlFor="xgbTrees">Trees (n_estimators)</label>
                    <input type="number" id="xgbTrees" value={xgbTrees} min={10} max={1000} onChange={(e) => setXgbTrees(+e.target.value)} />
                    <label htmlFor="xgbDepth">Max Depth</label>
                    <input type="number" id="xgbDepth" value={xgbDepth} min={1} max={30} onChange={(e) => setXgbDepth(+e.target.value)} />
                    <label htmlFor="xgbLr">Learning Rate</label>
                    <input type="number" id="xgbLr" step="0.01" value={xgbLr} min={0.001} max={1.0} onChange={(e) => setXgbLr(+e.target.value)} />
                  </div>
                </div>
              </div>
            )}

            {excluded.length > 0 && (
              <>
                <h3>Columns excluded by default</h3>
                {excluded.map((x) => (
                  <div className="small" key={x.name}>
                    {x.name} <Tag kind="bad">{x.identifier_level ? x.identifier_level + " identifier" : x.constant ? "constant" : x.semantic_type}</Tag>
                    {x.identifier_level && (
                      <label className="inline">
                        <input type="checkbox" checked={keep.has(x.name)} onChange={(e) => {
                          const s = new Set(keep); e.target.checked ? s.add(x.name) : s.delete(x.name); setKeep(s);
                        }} />keep as feature
                      </label>
                    )}
                  </div>
                ))}
              </>
            )}
            <p><button className="btn" disabled={!target || !pt || training} onClick={train}>{training ? "Training…" : "Train models"}</button></p>
          </>
        )}
      </div>
      {training && <div className="note">Training… cross-validating baseline, linear/statistical, Random Forest and XGBoost models.</div>}
      {err && <div className="note bad"><b>{err}</b></div>}
    </div>
  );
}
