import { useState } from 'react';
import { Award, CheckCircle2, Clock, Loader2, Play } from 'lucide-react';
import { api } from '../services/api';
import type { EvaluationSummary, RegisteredModel } from '../types';

interface ModelsPageProps {
  models: RegisteredModel[];
  evaluation: EvaluationSummary | null;
  onRefresh: () => void;
}

export function ModelsPage({ models, evaluation, onRefresh }: ModelsPageProps) {
  const [selectedKey, setSelectedKey] = useState<string>(
    evaluation?.best_model_key || evaluation?.models?.[0]?.model_key || 'muril'
  );
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainMsg, setTrainMsg] = useState<string | null>(null);

  const evalModels = evaluation?.models || [];
  const activeEval = evalModels.find((m) => m.model_key === selectedKey) || evalModels[0] || null;

  async function handleRetrainTransformers() {
    setTrainingBusy(true);
    setTrainMsg('Training all 4 Transformer models on 20k+ dataset splits...');
    try {
      const res = await api.triggerTraining(['muril', 'mbert', 'xlmr', 'indicbert']);
      if (!res.started) {
        setTrainMsg(res.reason || 'Training already running.');
      } else {
        const timer = setInterval(async () => {
          const st = await api.getTrainingStatus();
          if (!st.running) {
            clearInterval(timer);
            setTrainingBusy(false);
            setTrainMsg(st.last_error ? `Error: ${st.last_error}` : 'All 4 Transformer models retrained, saved & evaluated.');
            onRefresh();
          }
        }, 1500);
      }
    } catch (err) {
      setTrainingBusy(false);
      setTrainMsg(err instanceof Error ? err.message : 'Failed to start training.');
    }
  }

  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold text-slate-900">Models &amp; Benchmark Comparison</h1>
            <p className="mt-1 text-xs text-slate-500">
              All metrics are computed on the held-out Test split with strict group-based leakage prevention. Untrained models report their honest status.
            </p>
          </div>

          <button
            type="button"
            disabled={trainingBusy}
            onClick={handleRetrainTransformers}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
          >
            {trainingBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Re-train Transformer Models
          </button>
        </div>

        {trainMsg && (
          <div className="mt-3 rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-xs text-teal-900">
            {trainMsg}
          </div>
        )}

        <div className="mt-5 overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-2.5">Model</th>
                <th className="px-4 py-2.5">Family</th>
                <th className="px-4 py-2.5">Status</th>
                <th className="px-4 py-2.5">Macro F1</th>
                <th className="px-4 py-2.5">Accuracy</th>
                <th className="px-4 py-2.5">Hindi F1</th>
                <th className="px-4 py-2.5">Bengali F1</th>
                <th className="px-4 py-2.5">English F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {models.map((m) => {
                const isBest = evaluation?.best_model_key === m.model_key;
                const lm = m.metrics?.language_metrics || {};
                return (
                  <tr
                    key={m.model_key}
                    onClick={() => m.trained && setSelectedKey(m.model_key)}
                    className={m.trained ? 'cursor-pointer hover:bg-slate-50' : 'bg-slate-50/40 text-slate-500'}
                  >
                    <td className="px-4 py-2.5 font-medium text-slate-900">
                      <div className="flex items-center gap-1.5">
                        {isBest && <Award className="h-4 w-4 text-teal-700" />}
                        <span>{m.name}</span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-xs text-slate-600">{m.family}</td>
                    <td className="px-4 py-2.5 text-xs">
                      {m.trained ? (
                        <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                          <CheckCircle2 className="h-3.5 w-3.5" /> Trained
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-amber-700">
                          <Clock className="h-3.5 w-3.5" /> Model not trained
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 font-semibold tabular-nums">
                      {m.metrics ? `${(m.metrics.macro_f1 * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums">
                      {m.metrics ? `${(m.metrics.accuracy * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums">
                      {lm.Hindi ? `${(lm.Hindi.macro_f1 * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums">
                      {lm.Bengali ? `${(lm.Bengali.macro_f1 * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums">
                      {lm.English ? `${(lm.English.macro_f1 * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {activeEval && (
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="text-base font-bold text-slate-900">
              Confusion Matrix — {activeEval.model_name}
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Rows = True Label, Columns = Predicted Label ({activeEval.test_samples} test samples)
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-center text-xs">
                <thead>
                  <tr>
                    <th className="border border-slate-200 bg-slate-50 p-2">True \ Pred</th>
                    {activeEval.labels.map((lbl) => (
                      <th key={lbl} className="border border-slate-200 bg-slate-50 p-2 font-semibold">
                        {lbl}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {activeEval.confusion_matrix.map((row, rIdx) => (
                    <tr key={rIdx}>
                      <td className="border border-slate-200 bg-slate-50 p-2 font-semibold">
                        {activeEval.labels[rIdx]}
                      </td>
                      {row.map((val, cIdx) => (
                        <td
                          key={cIdx}
                          className={`border border-slate-200 p-2.5 font-mono font-semibold ${
                            rIdx === cIdx ? 'bg-teal-50 text-teal-900' : 'bg-white text-slate-700'
                          }`}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-5 space-y-2">
              <p className="text-xs font-semibold uppercase text-slate-500">Per-Class F1 Breakdown</p>
              {activeEval.labels.map((lbl) => {
                const pc = activeEval.per_class[lbl];
                return (
                  <div key={lbl} className="flex items-center justify-between text-xs">
                    <span className="font-medium text-slate-700">
                      {lbl} (n={pc?.support ?? 0})
                    </span>
                    <span className="font-mono text-slate-800">
                      P={(pc?.precision * 100).toFixed(1)}% · R={(pc?.recall * 100).toFixed(1)}% · F1=
                      {(pc?.f1 * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="text-base font-bold text-slate-900">Benchmark Visualizations</h2>
            <img
              src="/charts/model_comparison.png"
              alt="Model Performance Comparison Chart"
              className="w-full rounded-lg border border-slate-200"
            />
            <img
              src="/charts/language_f1_comparison.png"
              alt="Language-wise Macro F1 Comparison"
              className="w-full rounded-lg border border-slate-200"
            />
          </div>
        </section>
      )}
    </div>
  );
}
