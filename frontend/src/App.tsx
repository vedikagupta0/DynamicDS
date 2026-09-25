import { Routes, Route, Navigate } from "react-router-dom";
import { AppStateProvider } from "./lib/store";
import Nav from "./components/Nav";
import UploadPage from "./pages/UploadPage";
import OverviewPage from "./pages/OverviewPage";
import QualityPage from "./pages/QualityPage";
import EdaPage from "./pages/EdaPage";
import WarningsPage from "./pages/WarningsPage";
import ModelPage from "./pages/ModelPage";
import ResultsPage from "./pages/ResultsPage";
import PredictPage from "./pages/PredictPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import RegistryPage from "./pages/RegistryPage";

export default function App() {
  return (
    <AppStateProvider>
      <div className="app">
        <Nav />
        <main>
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/quality" element={<QualityPage />} />
            <Route path="/eda" element={<EdaPage />} />
            <Route path="/warnings" element={<WarningsPage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/registry" element={<RegistryPage />} />
          </Routes>
        </main>
      </div>
    </AppStateProvider>
  );
}
