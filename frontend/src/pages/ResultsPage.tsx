import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON } from "../lib/api";
import { useApp } from "../lib/store";
import { Bar, Note, Tag } from "../components/ui";
import { ConfusionMatrix, CurveChart, LineSeriesChart, ScatterXY, Beeswarm, HistogramChart } from "../components/charts";
import { fmt } from "../lib/format";

function DecisionsBlock({ r }: { r: any }) {
  return (
    <>
      <h2>Every data decision</h2>
      <div className="card">
        {r.decisions.length ? r.decisions.map((d: any, i: number) => (
          <p className="small" key={i}><Tag kind={d.action === "excluded" ? "warn" : ""}>{d.action.replace("_", " ")}</Tag><b>{d.column}</b> {d.reason}</p>
        )) : <p>No columns or rows were removed.</p>}
      </div>
    </>
  );
}

export default function ResultsPage() {
  const { expId } = useApp();
  const nav = useNavigate();
  const [exp, setExp] = useState<any>(null);
  const [diagModel, setDiagModel] = useState("");

  useEffect(() => { if (expId) getJSON(`/experiments/${expId}`).then(setExp); }, [expId]);

  if (!expId) return <Note>No experiment yet. Choose a task first.</Note>;
  if (!exp) return <p className="muted">Loading…</p>;
  if (exp.status !== "completed") return <Note bad={exp.status === "failed"}>Experiment is {exp.status}. {exp.error || ""}</Note>;

  const r = exp.results, pm = r.primary_metric, arrow = r.lower_is_better ? "lower is better" : "higher is better";
  const lead = r.models.find((m: any) => m.leader);
  const okModels = r.models.filter((m: any) => m.status === "ok");
  const dm = diagModel || lead.key;
  const dmRow = okModels.find((m: any) => m.key === dm) ?? lead;

  return (
    <div>
      <h1>Results</h1>
      <p className="muted">{exp.model.name} {exp.model.version} · primary metric <b>{pm}</b> ({arrow}) · <a href={`/api/experiments/${exp.id}/report`}>download model report</a></p>
      <Note good={r.beats_baseline} bad={!r.beats_baseline}><b>{r.baseline_note}</b> {r.mode === "tabular" ? r.metric_note : ""}</Note>

      {r.mode === "tabular" ? (
        <>
          <h2>Model comparison</h2>
          <div className="card tscroll">
            <table>
              <thead><tr><th>Model</th><th>{pm} (CV mean ± std)</th><th>{pm} (test)</th><th>Train time</th><th>vs baseline</th><th>Status</th></tr></thead>
              <tbody>
                {r.models.map((m: any) => m.status !== "ok" ? (
                  <tr key={m.key}><td>{m.name}</td><td colSpan={4} className="err small">{m.error}</td><td><Tag kind="bad">failed</Tag></td></tr>
                ) : (
                  <tr key={m.key}>
                    <td><b>{m.name}</b> {m.leader && <Tag kind="ok">leader on {pm}</Tag>}{m.is_baseline && <Tag>baseline</Tag>}</td>
                    <td>{fmt(m.primary_cv)} ± {fmt(m.primary_cv_std, 2)}</td><td>{fmt(m.primary_test)}</td><td>{m.train_time_s}s</td>
                    <td>{m.delta_vs_baseline == null ? "–" : (m.delta_vs_baseline > 0 ? "+" : "") + fmt(m.delta_vs_baseline, 3) + " better"}</td>
                    <td><Tag kind="ok">ok</Tag></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>All test metrics</h3>
          <div className="card tscroll">
            <table>
              <thead><tr><th>Model</th>{Object.keys(lead.test).map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{okModels.map((m: any) => <tr key={m.key}><td>{m.name}</td>{Object.keys(lead.test).map((k) => <td key={k}>{fmt(m.test[k])}</td>)}</tr>)}</tbody>
            </table>
          </div>
          <p className="small muted">Split: {r.split.method} · {fmt(r.split.train_rows)} train / {fmt(r.split.test_rows)} test rows · {r.split.cv_folds}-fold CV on the training split.{r.imbalance?.imbalanced ? ` Imbalanced target: ${r.imbalance.handling}` : ""}</p>

          <h2>Diagnostics</h2>
          <label htmlFor="dm">Model</label>
          <select id="dm" value={dm} onChange={(e) => setDiagModel(e.target.value)}>
            {okModels.map((m: any) => <option key={m.key} value={m.key}>{m.name}</option>)}
          </select>
          <div>
            {dmRow.diagnostics.confusion_matrix ? (
              <div className="grid">
                <div className="card"><b>Confusion matrix</b><p className="small muted">rows = actual, columns = predicted</p><ConfusionMatrix labels={dmRow.diagnostics.labels} matrix={dmRow.diagnostics.confusion_matrix} /></div>
                {dmRow.diagnostics.roc && (
                  <>
                    <div className="card"><b>ROC curve</b><CurveChart x={dmRow.diagnostics.roc.fpr} y={dmRow.diagnostics.roc.tpr} /><span className="small muted">AUC {fmt(dmRow.test.roc_auc)}</span></div>
                    <div className="card"><b>Precision–recall curve</b><CurveChart x={dmRow.diagnostics.pr.recall} y={dmRow.diagnostics.pr.precision} baseline={dmRow.diagnostics.pr.baseline} /><span className="small muted">PR-AUC {fmt(dmRow.test.pr_auc)} · dashed = prevalence</span></div>
                  </>
                )}
                <div className="card tscroll" style={{ gridColumn: "1 / -1" }}>
                  <b>Classification report</b>
                  <table>
                    <thead><tr><th>Class</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr></thead>
                    <tbody>{dmRow.diagnostics.labels.map((l: string) => { const q = dmRow.diagnostics.report[l]; return <tr key={l}><td>{l}</td><td>{fmt(q.precision, 3)}</td><td>{fmt(q.recall, 3)}</td><td>{fmt(q["f1-score"], 3)}</td><td>{q.support}</td></tr>; })}</tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="grid">
                <div className="card"><b>Actual vs predicted</b><ScatterXY x={dmRow.diagnostics.actual} y={dmRow.diagnostics.predicted} diag /></div>
                <div className="card"><b>Residuals vs predicted</b><ScatterXY x={dmRow.diagnostics.predicted} y={dmRow.diagnostics.residuals} zero /></div>
                <div className="card"><b>Error distribution</b><HistogramChart counts={dmRow.diagnostics.error_hist.counts} edges={dmRow.diagnostics.error_hist.edges} /></div>
              </div>
            )}
            {dmRow.diagnostics.sampled && <p className="small muted">Scatter plots show a random sample of test rows.</p>}
          </div>
        </>
      ) : (
        <>
          <h2>Validation design</h2>
          <div className="card">
            <p>{r.split.explanation}</p>
            <table>
              <thead><tr><th>Window</th><th>From</th><th>To</th><th>Periods</th></tr></thead>
              <tbody>{(["train", "validation", "test"] as const).map((k) => <tr key={k}><td>{k}</td><td>{r.split[k].start.slice(0, 10)}</td><td>{r.split[k].end.slice(0, 10)}</td><td>{r.split[k].n}</td></tr>)}</tbody>
            </table>
            <p className="small muted">Frequency: {r.frequency.label}{r.season_period ? ` · seasonal period ${r.season_period}` : " · no seasonality detected"}</p>
          </div>
          <h2>Model comparison</h2>
          <div className="card tscroll">
            <table>
              <thead><tr><th>Model</th><th>{pm} validation</th><th>{pm} test</th><th>vs best baseline</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>sMAPE</th></tr></thead>
              <tbody>
                {r.models.map((m: any) => m.status !== "ok" ? (
                  <tr key={m.key}><td>{m.name}</td><td colSpan={7} className="err small">{m.error}</td></tr>
                ) : (
                  <tr key={m.key}>
                    <td><b>{m.name}</b> {m.leader && <Tag kind="ok">leader on validation</Tag>}{m.is_baseline && <Tag>baseline</Tag>}</td>
                    <td>{fmt(m.primary_val)}</td><td>{fmt(m.primary_test)}</td>
                    <td>{m.improvement_vs_best_baseline_pct == null ? "–" : (m.improvement_vs_best_baseline_pct > 0 ? "+" : "") + m.improvement_vs_best_baseline_pct.toFixed(1) + "%"}</td>
                    <td>{fmt(m.test.mae)}</td><td>{fmt(m.test.rmse)}</td><td>{fmt(m.test.mape)}</td><td>{fmt(m.test.smape)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h2>Forecast</h2>
          <div className="card">
            {(() => {
              const c = r.charts;
              const series = [
                { name: "history", data: c.history.t.map((t: string, i: number) => ({ t, y: c.history.y[i] })), color: "#9aa7b3" },
                { name: "test actual", data: c.test.t.map((t: string, i: number) => ({ t, y: c.test.actual[i] })), color: "#16212b", width: 2 },
                ...Object.entries(c.test.pred).map(([k, v]: [string, any]) => ({ name: k, data: c.test.t.map((t: string, i: number) => ({ t, y: v[i] })), dashed: k === "naive" || k === "snaive" })),
                { name: `future (${c.future.model})`, data: c.future.t.map((t: string, i: number) => ({ t, y: c.future.yhat[i] })), color: "#b3283d", width: 2.2 },
              ];
              return <LineSeriesChart series={series} />;
            })()}
          </div>
          <h3>Test-window residuals (leader)</h3>
          <div className="card"><LineSeriesChart height={120} series={[{ name: "actual − forecast", data: r.charts.residuals.map((y: number, i: number) => ({ t: i, y })) }]} /></div>
        </>
      )}

      {r.explainability?.permutation_importance && (
        <>
          <h2>Feature importance</h2>
          <p className="small muted">Importance = how much the model's {pm} degrades when a feature is shuffled on held-out rows. It shows what the model relies on, not the direction of an effect and not causality.</p>
          <div className="card"><Bar items={r.explainability.permutation_importance.slice(0, 15).map((f: any) => [f.feature, Math.max(f.importance, 0), "±" + fmt(f.std, 2)])} /></div>
        </>
      )}
      {r.explainability?.linear && (
        <>
          <h2>Feature effect (linear coefficients)</h2>
          <p className="small muted">{r.explainability.linear.note}</p>
          <div className="card"><Bar items={r.explainability.linear.effects.slice(0, 15).map((f: any) => [f.feature, f.coefficient])} fmtv={(v) => fmt(v, 3)} /></div>
        </>
      )}
      {r.explainability?.shap && (
        <>
          <h2>SHAP summary</h2>
          <p className="small muted">{r.explainability.shap.note} Each dot is a test row; position is its SHAP value, colour is the feature value (low → high).</p>
          <div className="card">{r.explainability.shap.beeswarm.map((b: any, i: number) => <Beeswarm key={i} feature={b.feature} points={b.points} />)}</div>
        </>
      )}
      {(r.explainability?.shap_error || r.explainability?.permutation_error) && <Note bad>{r.explainability.shap_error || r.explainability.permutation_error}</Note>}
      {r.explainability?.feature_importance && (
        <>
          <h2>Lag-model feature importance</h2>
          <div className="card"><Bar items={r.explainability.feature_importance.map((f: any) => [f.feature, f.importance])} /></div>
        </>
      )}

      {r.mode === "tabular" ? (
        <>
          <DecisionsBlock r={r} />
          <h2>Preprocessing (fitted on training data only)</h2>
          <div className="card small">
            {r.preprocessing && Object.entries(r.preprocessing).map(([k, v]: [string, any]) => (
              <p key={k}><b>{k.replace(/_/g, " ")}:</b> {Array.isArray(v) ? (v.join(", ") || "none") : String(v)}</p>
            ))}
          </div>
        </>
      ) : <DecisionsBlock r={r} />}

      <h2>Notes</h2>
      <ul className="small">{r.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
      <p><button className="btn" onClick={() => nav("/predict")}>Use this model</button></p>
    </div>
  );
}
