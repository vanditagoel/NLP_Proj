import { useCallback, useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { DatasetPage } from './pages/DatasetPage';
import { HomePage } from './pages/HomePage';
import { ModelTestingPage } from './pages/ModelTestingPage';
import { ModelsPage } from './pages/ModelsPage';
import { api } from './services/api';
import type {
  DatasetStats,
  EvaluationSummary,
  NavTab,
  ProjectOverview,
  RegisteredModel,
} from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('home');
  const [project, setProject] = useState<ProjectOverview | null>(null);
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);

  const loadAll = useCallback(async () => {
    const [projRes, modRes, evalRes, statsRes] = await Promise.allSettled([
      api.getProject(),
      api.getModels(),
      api.getEvaluation(),
      api.getDatasetStats(),
    ]);
    if (projRes.status === 'fulfilled') setProject(projRes.value);
    if (modRes.status === 'fulfilled') setModels(modRes.value.models);
    if (evalRes.status === 'fulfilled') setEvaluation(evalRes.value);
    if (statsRes.status === 'fulfilled') setDatasetStats(statsRes.value);
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <Navbar activeTab={activeTab} onSelect={setActiveTab} />
      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        {activeTab === 'home' && <HomePage project={project} onNavigate={setActiveTab} />}
        {activeTab === 'testing' && <ModelTestingPage models={models} />}
        {activeTab === 'models' && (
          <ModelsPage models={models} evaluation={evaluation} onRefresh={loadAll} />
        )}
        {activeTab === 'dataset' && <DatasetPage stats={datasetStats} />}
      </main>
    </div>
  );
}
