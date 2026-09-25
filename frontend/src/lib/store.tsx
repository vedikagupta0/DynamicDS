import { createContext, useContext, useState, type ReactNode } from "react";

export interface ColumnMeta {
  name: string; semantic_type: string; identifier_level: string | null; n_unique: number;
  unique_ratio: number; constant: boolean; high_cardinality: boolean; evidence: string[];
  is_integer: boolean; formatted_numeric: boolean; [k: string]: unknown;
}
export interface Profile {
  id: string; filename: string; sha256: string; uploaded_at: string;
  overview: { rows: number; columns: number; memory_mb: number; file_size_bytes: number; missing_cell_pct: number; duplicate_rows: number; type_counts: Record<string, number> };
  columns: ColumnMeta[];
}

interface AppState {
  ds: string | null; prof: Profile | null; target: string; mode: "standard" | "forecast"; expId: string | null;
  setDataset: (id: string, prof: Profile) => void; setTarget: (t: string) => void; setMode: (m: "standard" | "forecast") => void; setExpId: (id: string) => void;
}
const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [ds, setDs] = useState<string | null>(null);
  const [prof, setProf] = useState<Profile | null>(null);
  const [target, setTargetS] = useState("");
  const [mode, setMode] = useState<"standard" | "forecast">("standard");
  const [expId, setExpId] = useState<string | null>(null);
  const value: AppState = {
    ds, prof, target, mode, expId,
    setDataset: (id, p) => { setDs(id); setProf(p); setTargetS(""); setExpId(null); },
    setTarget: setTargetS, setMode, setExpId,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
export const useApp = () => {
  const c = useContext(Ctx);
  if (!c) throw new Error("useApp outside provider");
  return c;
};
