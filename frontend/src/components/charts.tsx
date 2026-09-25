import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar as RBar, XAxis, YAxis, Tooltip, CartesianGrid,
  ScatterChart, Scatter, ReferenceLine, Cell, Legend,
} from "recharts";
import { fmt } from "../lib/format";

export function HistogramChart({ counts, edges, kde, boxplot }: { counts: number[]; edges: number[]; kde?: { x: number[]; y: number[] }; boxplot?: { whisker_low: number; q1: number; median: number; q3: number; whisker_high: number } }) {
  const data = counts.map((c, i) => ({ x: (edges[i] + edges[i + 1]) / 2, count: c, label: fmt(edges[i], 3) }));
  const kdeMax = kde ? Math.max(...kde.y) : 1;
  const countMax = Math.max(...counts) || 1;
  const merged = data.map((d, i) => ({ ...d, kde: kde ? (kde.y[i] ?? kde.y[kde.y.length - 1]) / kdeMax * countMax : undefined }));
  return (
    <div>
      <ResponsiveContainer width="100%" height={130}>
        <BarChart data={merged} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
          <XAxis dataKey="label" hide />
          <YAxis hide />
          <Tooltip formatter={(v) => fmt(v as number)} labelFormatter={(l) => `≈ ${l}`} />
          <RBar dataKey="count" fill="#23517a" opacity={0.85} />
          {kde && <Line dataKey="kde" stroke="#c0682c" dot={false} strokeWidth={1.5} type="monotone" />}
        </BarChart>
      </ResponsiveContainer>
      {boxplot && (
        <svg viewBox="0 0 300 26" width="100%" height={26}>
          {(() => {
            const lo = edges[0], hi = edges[edges.length - 1], r = hi - lo || 1;
            const sx = (v: number) => ((v - lo) / r) * 300;
            const b = boxplot;
            return (
              <>
                <line x1={sx(b.whisker_low)} x2={sx(b.whisker_high)} y1={13} y2={13} stroke="#16212b" />
                <rect x={sx(b.q1)} y={7} width={Math.max(sx(b.q3) - sx(b.q1), 1)} height={12} fill="#dbe6ef" stroke="#16212b" />
                <line x1={sx(b.median)} x2={sx(b.median)} y1={7} y2={19} stroke="#c0682c" strokeWidth={2} />
              </>
            );
          })()}
        </svg>
      )}
    </div>
  );
}

export function LineSeriesChart({ series, height = 240 }: { series: { name: string; data: { t: number | string; y: number | null }[]; color?: string; dashed?: boolean; width?: number }[]; height?: number }) {
  const pts: Record<string | number, Record<string, number | string>> = {};
  series.forEach((s) => s.data.forEach((d) => {
    pts[d.t] ??= { t: d.t };
    if (d.y != null) pts[d.t][s.name] = d.y;
  }));
  const data = Object.values(pts).sort((a, b) => (a.t > b.t ? 1 : -1));
  const colors = ["#9aa7b3", "#16212b", "#c0682c", "#23517a", "#6a8f3a", "#b3283d"];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e4eaf0" />
        <XAxis dataKey="t" tick={{ fontSize: 10 }} minTickGap={40} />
        <YAxis tick={{ fontSize: 10 }} width={54} />
        <Tooltip formatter={(v) => fmt(v as number)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s, i) => (
          <Line key={s.name} dataKey={s.name} name={s.name} stroke={s.color || colors[i % colors.length]} strokeWidth={s.width || 1.5} dot={false} strokeDasharray={s.dashed ? "5 4" : undefined} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ScatterXY({ x, y, diag, zero, height = 240 }: { x: number[]; y: number[]; diag?: boolean; zero?: boolean; height?: number }) {
  const data = x.map((v, i) => ({ x: v, y: y[i] }));
  const lo = Math.min(...x, ...y), hi = Math.max(...x, ...y);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e4eaf0" />
        <XAxis dataKey="x" tick={{ fontSize: 10 }} type="number" domain={["auto", "auto"]} />
        <YAxis dataKey="y" tick={{ fontSize: 10 }} width={54} type="number" domain={["auto", "auto"]} />
        <Tooltip formatter={(v) => fmt(v as number)} />
        {diag && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#c0682c" strokeDasharray="4 4" />}
        {zero && <ReferenceLine y={0} stroke="#c0682c" strokeDasharray="4 4" />}
        <Scatter data={data} fill="#23517a" fillOpacity={0.5} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function CurveChart({ x, y, baseline }: { x: number[]; y: number[]; baseline?: number | null }) {
  const data = x.map((v, i) => ({ x: v, y: y[i] }));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e4eaf0" />
        <XAxis dataKey="x" tick={{ fontSize: 10 }} type="number" domain={[0, 1]} />
        <YAxis tick={{ fontSize: 10 }} width={40} type="number" domain={[0, 1]} />
        <Tooltip formatter={(v) => fmt(v as number)} />
        {baseline != null && <ReferenceLine y={baseline} stroke="#9aa7b3" strokeDasharray="4 4" />}
        <Line dataKey="y" stroke="#23517a" strokeWidth={2} dot={false} type="monotone" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function HeatMatrix({ labels, matrix, max = 1 }: { labels: string[]; matrix: number[][]; max?: number }) {
  return (
    <div className="tscroll">
      <table>
        <thead>
          <tr>
            <th></th>
            {labels.map((l) => <th key={l}>{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th>{labels[i]}</th>
              {row.map((v, j) => {
                const a = v == null ? 0 : Math.min(Math.abs(v) / max, 1);
                return (
                  <td key={j} style={{ background: `rgba(35,81,122,${a * 0.85})`, color: a > 0.5 ? "#fff" : "inherit" }}>
                    {v == null ? "–" : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const max = Math.max(...matrix.flat()) || 1;
  return (
    <div className="tscroll">
      <table>
        <thead>
          <tr><th></th>{labels.map((l) => <th key={l}>{l}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th>{labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} style={{ background: i === j ? `rgba(47,125,91,${(v / max) * 0.7})` : `rgba(179,40,61,${(v / max) * 0.5})` }}>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Beeswarm({ feature, points }: { feature: string; points: { shap: number; value: number }[] }) {
  const m = Math.max(...points.map((p) => Math.abs(p.shap))) || 1;
  return (
    <div className="row" style={{ gap: ".5rem", alignItems: "center" }}>
      <span className="small" style={{ width: 150 }}>{feature}</span>
      <svg viewBox="0 0 400 22" width={400} height={22}>
        <line x1={200} x2={200} y1={0} y2={22} stroke="#d3dae1" />
        {points.map((p, j) => (
          <circle key={j} cx={200 + (p.shap / m) * 190} cy={11 + ((j * 7) % 13) - 6} r={2.4} fill={`hsl(${220 - p.value * 200},65%,50%)`} opacity={0.8} />
        ))}
      </svg>
    </div>
  );
}
export { Cell };
