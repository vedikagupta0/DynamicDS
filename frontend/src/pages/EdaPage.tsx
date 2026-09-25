import { useEffect, useState } from "react";
import { getJSON } from "../lib/api";
import { useApp } from "../lib/store";
import { Bar, Ledger, Tabs, Tag } from "../components/ui";
import { HistogramChart, HeatMatrix } from "../components/charts";
import { fmt, pct } from "../lib/format";

type Eda = any; // shape mirrors backend EDA payload; kept loose to avoid duplicating the whole schema

export default function EdaPage() {
  const { ds, target, setTarget, prof } = useApp();
  const [tab, setTab] = useState("numeric");
  const [eda, setEda] = useState<Eda | null>(null);

  useEffect(() => {
    if (!ds) return;
    const q = target ? `?target=${encodeURIComponent(target)}` : "";
    getJSON(`/datasets/${ds}/eda${q}`).then(setEda);
  }, [ds, target]);

  if (!eda || !prof) return <p className="muted">Loading…</p>;
  const tabs: [string, string][] = [["numeric", "Numerical"], ["categorical", "Categorical"], ["other", "Boolean and text"], ["datetime", "Datetime"], ["corr", "Correlation"], ["target", "Target"]];
  const colOpts = (f: (c: any) => boolean) => prof.columns.filter(f).map((c) => <option key={c.name} value={c.name}>{c.name}</option>);

  return (
    <div>
      <h1>Exploratory analysis</h1>
      <Tabs tabs={tabs} active={tab} onChange={setTab} />

      {tab === "numeric" && (
        <>
          <p className="muted small">Histogram (blue), KDE (orange), boxplot below. Skew classes use |skew| &lt; {eda.thresholds.skew_symmetric} symmetric, &lt; {eda.thresholds.skew_strong} moderate, otherwise strong. Outliers use {eda.thresholds.outlier_iqr_k}×IQR. {eda.sampling.note}</p>
          <div className="grid">
            {eda.numeric.length ? eda.numeric.map((c: any) => (
              <div className="card" key={c.column}>
                <b>{c.column}</b> {c.skew_class && <Tag>{c.skew_class}</Tag>}
                {c.outliers?.count ? <Tag kind={c.outliers.pct > 5 ? "warn" : ""}>{c.outliers.count} outliers ({pct(c.outliers.pct)})</Tag> : null}
                {c.histogram && <HistogramChart counts={c.histogram.counts} edges={c.histogram.edges} kde={c.kde} boxplot={c.boxplot} />}
                <table className="small"><tbody>
                  <tr><td>mean {fmt(c.mean)}</td><td>median {fmt(c.median)}</td><td>std {fmt(c.std)}</td></tr>
                  <tr><td>min {fmt(c.min)}</td><td>Q1 {fmt(c.q1)}</td><td>Q3 {fmt(c.q3)}</td></tr>
                  <tr><td>max {fmt(c.max)}</td><td>skew {fmt(c.skewness, 3)}</td><td>kurt {fmt(c.kurtosis, 3)}</td></tr>
                </tbody></table>
                {c.normality && <p className="small muted">{c.normality.test} (n={c.normality.n}{c.normality.sampled ? ", sampled" : ""}): {c.normality.conclusion}. The shape label above is descriptive, not a normality claim.</p>}
                {c.outliers?.count ? <p className="small muted">Bounds [{fmt(c.outliers.lower_bound, 3)}, {fmt(c.outliers.upper_bound, 3)}]. Review before removing; nothing is deleted.</p> : null}
              </div>
            )) : <p>No numerical columns.</p>}
          </div>
        </>
      )}

      {tab === "categorical" && (
        <div className="grid">
          {eda.categorical.length ? eda.categorical.map((c: any) => (
            <div className="card" key={c.column}>
              <b>{c.column}</b> <Tag>{c.cardinality_class}</Tag>
              <span className="muted small"> {c.unique} unique · most frequent “{c.most_frequent}” {pct(c.frequency_pct)} · {c.rare_categories} rare</span>
              {c.top_values && <Bar items={c.top_values.slice(0, c.plottable ? 20 : 10).map((v: any) => [v.value, v.pct, v.count])} max={100} fmtv={pct} />}
              {(c.showing_top_only || !c.plottable) && <p className="small muted">Showing top values only. Top-5 concentration {pct(c.top5_concentration_pct)}.</p>}
              {c.warning && <div className="note">{c.warning}</div>}
            </div>
          )) : <p>No categorical columns.</p>}
        </div>
      )}

      {tab === "other" && (
        <>
          <h3>Boolean</h3>
          <div className="grid">
            {eda.boolean.length ? eda.boolean.map((c: any) => (
              <div className="card" key={c.column}>
                <b>{c.column}</b>
                <div className="strip"><span style={{ width: `${c.true_pct}%`, background: "var(--good)" }} /><span style={{ width: `${c.false_pct}%`, background: "#9aa7b3" }} /></div>
                <span className="small">true {fmt(c.true)} ({pct(c.true_pct)}) · false {fmt(c.false)} ({pct(c.false_pct)}) · missing {fmt(c.missing)}</span>
              </div>
            )) : <p>None.</p>}
          </div>
          <h3>Text</h3>
          {eda.text.length ? eda.text.map((c: any) => (
            <div className="card" key={c.column}>
              <b>{c.column}</b>
              <p className="small">avg length {fmt(c.avg_length, 3)} · median {fmt(c.median_length, 3)} · unique {fmt(c.unique)} · missing {fmt(c.empty_or_missing)}</p>
              <Bar items={c.common_tokens.map((t: any) => [t.token, t.count])} />
              <p className="small muted">Common tokens are shown instead of a word cloud.</p>
            </div>
          )) : <p>None.</p>}
        </>
      )}

      {tab === "datetime" && (eda.datetime.length ? eda.datetime.map((c: any) => (
        <div className="card" key={c.column}>
          <b>{c.column}</b> <Tag kind="ok">forecast candidate</Tag>
          <p>{(c.min || "").slice(0, 19)} → {(c.max || "").slice(0, 19)} · {fmt(c.duration_days, 5)} days · missing {fmt(c.missing)}
            {c.frequency && <> · <b>{c.frequency.label}</b> {c.frequency.regular ? "(regular)" : `(${c.frequency.gaps} gaps)`}</>}
          </p>
          <div className="grid">
            <div><span className="small muted">by year</span><Bar items={Object.entries(c.by_year || {}) as [string, number][]} /></div>
            <div><span className="small muted">by month</span><Bar items={Object.entries(c.by_month || {}) as [string, number][]} /></div>
            <div><span className="small muted">by weekday (0=Mon)</span><Bar items={Object.entries(c.by_weekday || {}) as [string, number][]} /></div>
            {c.by_hour && <div><span className="small muted">by hour</span><Bar items={Object.entries(c.by_hour) as [string, number][]} /></div>}
          </div>
        </div>
      )) : <p>No datetime columns detected.</p>)}

      {tab === "corr" && (
        <>
          <p className="muted small">{eda.correlation.note} Bands: |r| &lt; {eda.thresholds.corr_low} low, &lt; {eda.thresholds.corr_high} moderate, otherwise high.{eda.correlation.truncated ? " Showing the first 40 numerical columns." : ""}</p>
          {eda.correlation.pearson ? <><h3>Pearson</h3><HeatMatrix labels={eda.correlation.columns} matrix={eda.correlation.pearson} /></> : <p>Need at least two numerical columns.</p>}
          <h3>High feature–feature correlations</h3>
          {eda.correlation.high_pairs.length ? eda.correlation.high_pairs.map((p: any, i: number) => (
            <Ledger key={i} severity={Math.abs(p.pearson) >= 0.9 ? "WARNING" : "INFO"}>
              <b>{p.a} ↔ {p.b}</b> r = {p.pearson.toFixed(2)} (Spearman {p.spearman == null ? "–" : p.spearman.toFixed(2)}) <span className="muted">Potential multicollinearity.</span>
            </Ledger>
          )) : <p>None at |r| ≥ 0.7.</p>}
          {eda.correlation.categorical_pairs.length > 0 && (
            <><h3>Categorical associations (Cramér's V ≥ 0.5)</h3>
              {eda.correlation.categorical_pairs.map((p: any, i: number) => <p key={i}>{p.a} ↔ {p.b}: V = {p.cramers_v.toFixed(2)}</p>)}
            </>
          )}
        </>
      )}

      {tab === "target" && (
        <>
          <div className="card">
            <label htmlFor="tsel">Target column</label>
            <select id="tsel" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">(none)</option>
              {colOpts((c) => !["identifier", "text", "empty", "datetime"].includes(c.semantic_type))}
            </select>
          </div>
          {eda.target && (
            <>
              {eda.target.problem_type ? (
                <div className="note">Recommended problem type: <b>{eda.target.problem_type}</b>. {eda.target.recommendation.reason} You confirm this on the next step.</div>
              ) : <div className="note bad">{eda.target.recommendation.reason}</div>}
              {eda.target.distribution && eda.target.problem_type === "classification" && (
                <>
                  <h3>Class balance</h3>
                  <Bar items={eda.target.distribution.classes.map((c: any) => [c.value, c.pct, c.count])} max={100} fmtv={pct} />
                  <p className="small">Minority class {pct(eda.target.distribution.minority_pct)} · imbalance ratio {eda.target.distribution.imbalance_ratio}
                    {eda.target.imbalanced && " · imbalanced: prefer PR-AUC, recall and F1 over accuracy"}</p>
                </>
              )}
              {eda.target.distribution && eda.target.problem_type === "regression" && (
                <><h3>Target distribution</h3><div className="card" style={{ maxWidth: 420 }}>
                  <HistogramChart counts={eda.target.distribution.histogram.counts} edges={eda.target.distribution.histogram.edges} />
                  <span className="small">{eda.target.distribution.skew_class || ""}</span>
                </div></>
              )}
              {eda.target.associations.length > 0 && (
                <><h3>Feature ↔ target association</h3><p className="small muted">{eda.target.association_note}</p>
                  <Bar items={eda.target.associations.slice(0, 15).map((a: any) => [a.feature, a.strength, <><Tag>{a.measure.replace("_", " ")}</Tag><Tag>{a.band}</Tag></>])} max={1} />
                </>
              )}
              {eda.target.leakage_suspects.map((l: any, i: number) => (
                <Ledger key={i} severity="HIGH"><b>POTENTIAL TARGET LEAKAGE — {l.feature}</b>: {l.reason}</Ledger>
              ))}
            </>
          )}
        </>
      )}

      <p>
        <a className="btn" href={`/api/datasets/${ds}/report${target ? "?target=" + encodeURIComponent(target) : ""}`}>Download EDA report (HTML)</a>{" "}
        <button className="btn" onClick={() => (window.location.hash = "#/warnings")}>See warnings</button>
      </p>
    </div>
  );
}
