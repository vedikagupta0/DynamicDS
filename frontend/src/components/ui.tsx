import type { ReactNode } from "react";

export const esc = (s: unknown) => String(s ?? "");

export function Tag({ children, kind = "" }: { children: ReactNode; kind?: "" | "ok" | "warn" | "bad" }) {
  return <span className={`tag ${kind}`}>{children}</span>;
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="stat">
      <b>{value}</b>
      <span className="muted small">{label}</span>
    </div>
  );
}

export function Ledger({ severity, children }: { severity: "HIGH" | "WARNING" | "INFO"; children: ReactNode }) {
  return (
    <div className={`ledger ${severity}`}>
      <span className="sev">{severity}</span> {children}
    </div>
  );
}

export function Note({ good, bad, children }: { good?: boolean; bad?: boolean; children: ReactNode }) {
  return <div className={`note ${good ? "good" : ""} ${bad ? "bad" : ""}`}>{children}</div>;
}

export function Bar({ items, max, fmtv, color = "var(--accent)" }: { items: [string, number, ReactNode?][]; max?: number; fmtv?: (v: number) => string; color?: string }) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i[1]))) ?? 1;
  const f = fmtv ?? ((v: number) => String(v));
  return (
    <table>
      <tbody>
        {items.map(([k, v, extra], i) => (
          <tr key={i}>
            <td style={{ width: "34%" }}>{k}</td>
            <td>
              <div className="bar">
                <i style={{ width: `${Math.min(100, (Math.abs(v) / (m || 1)) * 100)}%`, background: color }} />
              </div>
            </td>
            <td style={{ width: 70 }}>{f(v)}</td>
            <td className="muted small">{extra}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Strip({ counts }: { counts: Record<string, number> }) {
  const COLORS: Record<string, string> = { numerical: "#23517a", categorical: "#c0682c", boolean: "#6a8f3a", datetime: "#8a4f9e", identifier: "#b3283d", text: "#2a9d9d", empty: "#9aa7b3" };
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const entries = Object.entries(counts).filter(([, v]) => v);
  return (
    <>
      <div className="strip">
        {entries.map(([k, v]) => (
          <span key={k} title={`${k}: ${v}`} style={{ width: `${(v / total) * 100}%`, background: COLORS[k] }} />
        ))}
      </div>
      <div className="legend">
        {entries.map(([k, v]) => (
          <span key={k}>
            <i style={{ background: COLORS[k] }} /> {k} {v}
          </span>
        ))}
      </div>
    </>
  );
}

export function Tabs({ tabs, active, onChange }: { tabs: [string, string][]; active: string; onChange: (k: string) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(([k, l]) => (
        <button key={k} role="tab" aria-selected={active === k} onClick={() => onChange(k)}>
          {l}
        </button>
      ))}
    </div>
  );
}

export function Seg<T extends string>({ options, value, onChange }: { options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="seg" role="group">
      {options.map(([v, l]) => (
        <button key={v} aria-pressed={value === v} onClick={() => onChange(v)}>
          {l}
        </button>
      ))}
    </div>
  );
}
