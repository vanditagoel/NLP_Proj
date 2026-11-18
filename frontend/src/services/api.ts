import type {
  BatchResponse,
  CompareResponse,
  DatasetSample,
  DatasetStats,
  EvaluationSummary,
  PredictionResult,
  ProjectOverview,
  RegisteredModel,
} from '../types';

async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((data as { detail?: string; error?: string }).detail || (data as { error?: string }).error || `Request failed (${res.status})`);
  }
  return data as T;
}

export const api = {
  getProject: () => apiJson<ProjectOverview>('/api/project'),
  getModels: () => apiJson<{ models: RegisteredModel[] }>('/api/models'),
  getEvaluation: () => apiJson<EvaluationSummary>('/api/evaluation'),
  getDatasetStats: () => apiJson<DatasetStats>('/api/dataset/stats'),
  getDatasetSamples: (params: { language?: string; label?: string; split?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params.language) qs.set('language', params.language);
    if (params.label) qs.set('label', params.label);
    if (params.split) qs.set('split', params.split);
    if (params.limit) qs.set('limit', String(params.limit));
    return apiJson<{ count: number; samples: DatasetSample[] }>(`/api/dataset/samples?${qs.toString()}`);
  },
  predict: (text: string, language: string, model_key: string) =>
    apiJson<PredictionResult>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language, model_key }),
    }),
  compare: (text: string, language: string, model_keys?: string[]) =>
    apiJson<CompareResponse>('/api/predict/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language, model_keys }),
    }),
  predictBatchJson: async (file: File, textColumn: string, modelKeys: string[]): Promise<BatchResponse> => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('text_column', textColumn);
    fd.append('model_keys', modelKeys.join(','));
    fd.append('return_csv', 'false');
    return apiJson<BatchResponse>('/api/predict/batch', { method: 'POST', body: fd });
  },
  downloadBatchCsv: async (file: File, textColumn: string, modelKeys: string[]): Promise<Blob> => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('text_column', textColumn);
    fd.append('model_keys', modelKeys.join(','));
    fd.append('return_csv', 'true');
    const res = await fetch('/api/predict/batch', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail || 'Batch CSV download failed');
    }
    return res.blob();
  },
  triggerTraining: (model_keys: string[], epochs?: number) =>
    apiJson<{ started: boolean; reason?: string; status: Record<string, unknown> }>('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_keys, epochs }),
    }),
  getTrainingStatus: () =>
    apiJson<{
      running: boolean;
      current_model: string | null;
      last_completed: string[];
      last_error: string | null;
    }>('/api/train/status'),
};
