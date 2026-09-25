import { useLocation, useNavigate } from "react-router-dom";
import { useApp } from "../lib/store";

const STEPS: [string, string][] = [
  ["upload", "Upload"], ["overview", "Profile"], ["quality", "Data quality"], ["eda", "EDA"],
  ["warnings", "Warnings"], ["model", "Choose task"], ["results", "Results"], ["predict", "Predict"],
];

export default function Nav() {
  const { ds } = useApp();
  const loc = useLocation();
  const nav = useNavigate();
  const current = loc.pathname.slice(1) || "upload";
  return (
    <nav>
      <div className="brand">Dynamic DS</div>
      <small>EDA and baseline tester</small>
      {STEPS.map(([k, l], i) => (
        <button key={k} aria-current={current === k ? "page" : undefined} disabled={!ds && i > 0} onClick={() => nav("/" + k)}>
          <span className="n">{i + 1}</span>
          {l}
        </button>
      ))}
      <hr />
      <button aria-current={current === "experiments" ? "page" : undefined} onClick={() => nav("/experiments")}>Experiments</button>
      <button aria-current={current === "registry" ? "page" : undefined} onClick={() => nav("/registry")}>Model registry</button>
    </nav>
  );
}
