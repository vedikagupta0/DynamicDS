export const fmt = (v: number | string | null | undefined, d = 4): string => {
  if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "–";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return v.toLocaleString();
  return Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : String(+v.toPrecision(d));
};
export const pct = (v: number | null | undefined): string => (v == null ? "–" : v.toFixed(1) + "%");
export const COLORS: Record<string, string> = {
  numerical: "#23517a", categorical: "#c0682c", boolean: "#6a8f3a", datetime: "#8a4f9e",
  identifier: "#b3283d", text: "#2a9d9d", empty: "#9aa7b3",
};
