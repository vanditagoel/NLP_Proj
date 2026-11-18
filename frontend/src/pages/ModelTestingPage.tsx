import { useEffect, useState } from 'react';
import { Download, Loader2, Play, Sparkles, Upload } from 'lucide-react';
import { StanceBadge } from '../components/StanceBadge';
import { api } from '../services/api';
import type { BatchResponse, CompareResponse, PredictionResult, RegisteredModel } from '../types';

const EXAMPLES = [
  {
    lang: 'Hindi',
    text: 'जलवायु परिवर्तन एक गंभीर संकट है और हमें जीवाश्म ईंधन उत्सर्जन तुरंत कम करना चाहिए।',
  },
  {
    lang: 'Bengali',
    text: 'বৈশ্বিক উষ্ণতা বৃদ্ধির ফলে হিমবাহ গলছে এবং উপকূলীয় শহরগুলো ঝুঁকির মুখে পড়ছে।',
  },
  {
    lang: 'Hindi (Against)',
    text: 'जलवायु सदियों से प्राकृतिक रूप से बदलती रही है, मानव उत्सर्जन कोई संकट नहीं है।',
  },
  {
    lang: 'Bengali (None)',
    text: 'আবহাওয়া দপ্তর আগামীকাল সন্ধ্যায় কলকাতার কিছু অংশে হালকা বৃষ্টির পূর্বাভাস দিয়েছে।',
  },
  {
    lang: 'English',
    text: 'Climate change is an urgent global emergency driven by greenhouse gas emissions.',
  },
];

export function ModelTestingPage({ models }: { models: RegisteredModel[] }) {
  const trainedModels = models.filter((m) => m.trained);
  const [text, setText] = useState(EXAMPLES[0].text);
  const [language, setLanguage] = useState('Auto');
  const [selectedModel, setSelectedModel] = useState<string>(trainedModels[0]?.model_key || 'tfidf_logreg');
  const [compareMode, setCompareMode] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  // Batch CSV testing state
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [csvPreviewRows, setCsvPreviewRows] = useState<string[][]>([]);
  const [textColumn, setTextColumn] = useState('text');
  const [batchModelKeys, setBatchModelKeys] = useState<string[]>(
    trainedModels.length > 0 ? [trainedModels[0].model_key] : ['tfidf_logreg']
  );
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);

  useEffect(() => {
    if (trainedModels.length > 0 && !trainedModels.some((m) => m.model_key === selectedModel)) {
      setSelectedModel(trainedModels[0].model_key);
    }
    if (trainedModels.length > 0 && batchModelKeys.every((k) => !trainedModels.some((m) => m.model_key === k))) {
      setBatchModelKeys([trainedModels[0].model_key]);
    }
  }, [models]);

  async function handleRunInference() {
    if (!text.trim()) {
      setError('Please enter a Hindi, Bengali, or English comment.');
      return;
    }
    setLoading(true);
    setError(null);
    setSingleResult(null);
    setCompareResult(null);
    try {
      if (compareMode) {
        const res = await api.compare(text.trim(), language);
        setCompareResult(res);
      } else {
        const res = await api.predict(text.trim(), language, selectedModel);
        setSingleResult(res);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(file: File | null) {
    setCsvFile(file);
    setBatchResult(null);
    setBatchError(null);
    if (!file) {
      setCsvColumns([]);
      setCsvPreviewRows([]);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const content = String(reader.result || '');
      const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0).slice(0, 6);
      if (lines.length > 0) {
        const header = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
        setCsvColumns(header);
        const preferred = header.find((c) => ['text', 'comment', 'sentence', 'premise'].includes(c.toLowerCase()));
        setTextColumn(preferred || header[0]);
        setCsvPreviewRows(lines.slice(1).map((l) => l.split(',').map((c) => c.trim().replace(/^"|"$/g, ''))));
      }
    };
    reader.readAsText(file);
  }

  async function handleRunBatch() {
    if (!csvFile) return;
    setBatchLoading(true);
    setBatchError(null);
    try {
      const res = await api.predictBatchJson(csvFile, textColumn, batchModelKeys);
      setBatchResult(res);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'Batch prediction failed.');
    } finally {
      setBatchLoading(false);
    }
  }

  async function handleDownloadBatchCsv() {
    if (!csvFile) return;
    try {
      const blob = await api.downloadBatchCsv(csvFile, textColumn, batchModelKeys);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'climate_stance_predictions.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'CSV download failed.');
    }
  }

  function toggleBatchModel(key: string) {
    setBatchModelKeys((prev) =>
      prev.includes(key) ? (prev.length > 1 ? prev.filter((k) => k !== key) : prev) : [...prev, key]
    );
  }

  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold text-slate-900">Single Comment Testing &amp; Comparison</h1>
            <p className="mt-1 text-xs text-slate-500">
              Predicts stance toward: &ldquo;Climate change and global warming is a serious concern.&rdquo;
            </p>
          </div>
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-700">
            <input
              type="checkbox"
              checked={compareMode}
              onChange={(e) => setCompareMode(e.target.checked)}
              className="rounded border-slate-300 text-teal-700"
            />
            Compare all trained models
          </label>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setText(ex.text)}
              className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700 hover:bg-slate-100"
            >
              <Sparkles className="h-3 w-3 text-teal-700" />
              {ex.lang}
            </button>
          ))}
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          placeholder="Enter a Hindi, Bengali, or English comment..."
          className="mt-3 w-full rounded-lg border border-slate-300 p-3 text-sm text-slate-900 focus:border-teal-600 focus:outline-none"
        />

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div>
            <label className="block text-xs font-medium text-slate-600">Language</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800"
            >
              <option value="Auto">Auto Detect (Script + LangID)</option>
              <option value="Hindi">Hindi</option>
              <option value="Bengali">Bengali</option>
              <option value="English">English</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={compareMode}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 disabled:bg-slate-100"
            >
              {models.map((m) => (
                <option key={m.model_key} value={m.model_key}>
                  {m.name} {m.trained ? '' : '(Not trained)'}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              type="button"
              onClick={handleRunInference}
              disabled={loading}
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-teal-700 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-800 disabled:opacity-60"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {compareMode ? 'Compare Models' : 'Predict Stance'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            {error}
          </div>
        )}

        {singleResult && (
          <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <StanceBadge stance={singleResult.stance} />
                <span className="text-sm font-semibold text-slate-800">
                  Confidence: {(singleResult.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="text-xs text-slate-500">
                Detected: <strong>{singleResult.detected_language}</strong> · Model: {singleResult.model_name} ·{' '}
                {singleResult.processing_time_ms} ms
              </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              {(['Favour', 'Against', 'None'] as const).map((lbl) => {
                const prob = (singleResult.probabilities[lbl] ?? 0) * 100;
                return (
                  <div key={lbl} className="rounded-md border border-slate-200 bg-white p-2.5">
                    <div className="flex justify-between text-xs font-medium text-slate-700">
                      <span>{lbl}</span>
                      <span className="tabular-nums">{prob.toFixed(1)}%</span>
                    </div>
                    <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className={`h-full ${
                          lbl === 'Favour' ? 'bg-emerald-600' : lbl === 'Against' ? 'bg-rose-600' : 'bg-slate-500'
                        }`}
                        style={{ width: `${Math.min(100, prob)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {compareResult && (
          <div className="mt-5 space-y-3">
            <div className="rounded-lg border border-teal-200 bg-teal-50/70 px-4 py-2.5 text-sm text-teal-900">
              <strong>{compareResult.agreement.summary}</strong> · Detected Language: {compareResult.detected_language}
            </div>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-4 py-2.5">Model</th>
                    <th className="px-4 py-2.5">Prediction</th>
                    <th className="px-4 py-2.5">Confidence</th>
                    <th className="px-4 py-2.5">Time (ms)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  {compareResult.predictions.map((p) => (
                    <tr key={p.model_key}>
                      <td className="px-4 py-2.5 font-medium text-slate-800">{p.model_name}</td>
                      <td className="px-4 py-2.5">
                        <StanceBadge stance={p.stance} />
                      </td>
                      <td className="px-4 py-2.5 tabular-nums">{(p.confidence * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 tabular-nums text-slate-500">{p.processing_time_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="text-lg font-bold text-slate-900">Batch CSV Testing</h2>
        <p className="mt-1 text-xs text-slate-500">
          Upload a UTF-8 CSV file with Hindi, Bengali, or English comments, choose the text column, run predictions, and download the annotated CSV.
        </p>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <label className="block text-xs font-medium text-slate-600">1. Upload CSV</label>
            <label className="mt-1 flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-700 hover:bg-slate-100">
              <Upload className="h-4 w-4 text-teal-700" />
              <span>{csvFile ? csvFile.name : 'Choose a .csv file...'}</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                className="hidden"
              />
            </label>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600">2. Select Text Column</label>
            <select
              value={textColumn}
              onChange={(e) => setTextColumn(e.target.value)}
              disabled={csvColumns.length === 0}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 disabled:bg-slate-100"
            >
              {csvColumns.length === 0 ? (
                <option value="text">Upload CSV to detect columns</option>
              ) : (
                csvColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>

        {csvPreviewRows.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500">
                <tr>
                  {csvColumns.map((c) => (
                    <th key={c} className="px-3 py-2">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {csvPreviewRows.map((row, idx) => (
                  <tr key={idx}>
                    {csvColumns.map((_, cIdx) => (
                      <td key={cIdx} className="max-w-xs truncate px-3 py-1.5 text-slate-700">
                        {row[cIdx] || ''}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-4">
          <p className="text-xs font-medium text-slate-600">3. Select Trained Model(s)</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {trainedModels.map((m) => (
              <button
                key={m.model_key}
                type="button"
                onClick={() => toggleBatchModel(m.model_key)}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium ${
                  batchModelKeys.includes(m.model_key)
                    ? 'border-teal-700 bg-teal-700 text-white'
                    : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-50'
                }`}
              >
                {m.name}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            disabled={!csvFile || batchLoading || trainedModels.length === 0}
            onClick={handleRunBatch}
            className="inline-flex items-center gap-2 rounded-lg bg-teal-700 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-800 disabled:opacity-50"
          >
            {batchLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Batch Analysis
          </button>

          {batchResult && (
            <button
              type="button"
              onClick={handleDownloadBatchCsv}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              <Download className="h-4 w-4" />
              Download Annotated CSV
            </button>
          )}
        </div>

        {batchError && (
          <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            {batchError}
          </div>
        )}

        {batchResult && (
          <div className="mt-5 space-y-3">
            <div className="flex flex-wrap gap-4 text-xs font-medium text-slate-700">
              <span>Total Rows: {batchResult.total_rows}</span>
              <span className="text-emerald-700">Favour: {batchResult.counts.Favour}</span>
              <span className="text-rose-700">Against: {batchResult.counts.Against}</span>
              <span className="text-slate-600">None: {batchResult.counts.None}</span>
            </div>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Text</th>
                    <th className="px-3 py-2">Language</th>
                    <th className="px-3 py-2">Prediction</th>
                    <th className="px-3 py-2">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  {batchResult.rows.slice(0, 25).map((r, idx) => (
                    <tr key={idx}>
                      <td className="max-w-md truncate px-3 py-2 text-slate-800">{String(r[textColumn] || '')}</td>
                      <td className="px-3 py-2 text-slate-600">{String(r.detected_language || '')}</td>
                      <td className="px-3 py-2">
                        <StanceBadge stance={String(r.prediction)} />
                      </td>
                      <td className="px-3 py-2 tabular-nums">
                        {(Number(r.confidence || 0) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
