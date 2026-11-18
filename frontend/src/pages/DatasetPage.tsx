import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { api } from '../services/api';
import type { DatasetSample, DatasetStats } from '../types';

export function DatasetPage({ stats }: { stats: DatasetStats | null }) {
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [langFilter, setLangFilter] = useState('All');
  const [labelFilter, setLabelFilter] = useState('All');
  const [splitFilter, setSplitFilter] = useState('All');

  useEffect(() => {
    api
      .getDatasetSamples({ language: langFilter, label: labelFilter, split: splitFilter, limit: 20 })
      .then((res) => setSamples(res.samples))
      .catch(() => setSamples([]));
  }, [langFilter, labelFilter, splitFilter]);

  if (!stats) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600">
        Dataset statistics are not generated yet. Run the dataset preparation and analysis pipeline.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-xl font-bold text-slate-900">Dataset Audit &amp; Multilingual Distribution</h1>
        <p className="mt-1 text-xs text-slate-500">
          Origin: {stats.prepare_meta?.dataset_origin || stats.dataset_path} · Total Rows:{' '}
          {stats.total_samples.toLocaleString()} · Unique Groups:{' '}
          {stats.prepare_meta?.unique_groups?.toLocaleString() ?? 'N/A'}
        </p>

        {stats.prepare_meta?.leakage_verified && (
          <div className="mt-3 inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            Group-stratified split verified: zero overlap of source group_id across Train, Validation, and Test
          </div>
        )}

        {stats.pol_nli_inspection && (
          <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50/70 p-4 text-xs text-amber-950">
            <div className="flex items-center gap-2 font-semibold">
              <AlertTriangle className="h-4 w-4 text-amber-700" />
              Pol_NLI Task Compatibility Audit ({stats.pol_nli_inspection.total_rows.toLocaleString()} total raw NLI rows)
            </div>
            <p className="mt-1.5 leading-relaxed">
              {stats.pol_nli_inspection.compatibility_analysis?.reason}
            </p>
            <p className="mt-1.5 leading-relaxed font-medium">
              Resolution: {stats.pol_nli_inspection.compatibility_analysis?.legitimate_resolution}
            </p>
          </div>
        )}

        <div className="mt-5 grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Class Distribution</p>
            <div className="mt-2 space-y-1 text-sm">
              {Object.entries(stats.class_distribution).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span>{k}</span>
                  <span className="font-semibold tabular-nums">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Language Distribution</p>
            <div className="mt-2 space-y-1 text-sm">
              {Object.entries(stats.language_distribution).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span>{k}</span>
                  <span className="font-semibold tabular-nums">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Split &amp; Length Stats</p>
            <div className="mt-2 space-y-1 text-sm">
              {Object.entries(stats.split_distribution).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span>{k}</span>
                  <span className="font-semibold tabular-nums">{v}</span>
                </div>
              ))}
              <div className="flex justify-between border-t border-slate-200 pt-1 text-xs text-slate-600">
                <span>Mean / Median Words</span>
                <span className="tabular-nums">
                  {stats.text_length_words.mean} / {stats.text_length_words.median}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <img src="/charts/class_distribution.png" alt="Class Distribution" className="w-full rounded-lg" />
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <img src="/charts/language_distribution.png" alt="Language Distribution" className="w-full rounded-lg" />
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <img src="/charts/language_x_class_distribution.png" alt="Language by Class" className="w-full rounded-lg" />
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <img src="/charts/text_length_distribution.png" alt="Text Length Distribution" className="w-full rounded-lg" />
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-bold text-slate-900">Sample Records Inspector</h2>
          <div className="flex flex-wrap gap-2 text-xs">
            <select
              value={langFilter}
              onChange={(e) => setLangFilter(e.target.value)}
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5"
            >
              <option value="All">All Languages</option>
              <option value="Hindi">Hindi</option>
              <option value="Bengali">Bengali</option>
              <option value="English">English</option>
            </select>
            <select
              value={labelFilter}
              onChange={(e) => setLabelFilter(e.target.value)}
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5"
            >
              <option value="All">All Stances</option>
              <option value="Favour">Favour</option>
              <option value="Against">Against</option>
              <option value="None">None</option>
            </select>
            <select
              value={splitFilter}
              onChange={(e) => setSplitFilter(e.target.value)}
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5"
            >
              <option value="All">All Splits</option>
              <option value="Train">Train</option>
              <option value="Validation">Validation</option>
              <option value="Test">Test</option>
            </select>
          </div>
        </div>

        <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500">
              <tr>
                <th className="px-3 py-2">Text</th>
                <th className="px-3 py-2">Language</th>
                <th className="px-3 py-2">Label</th>
                <th className="px-3 py-2">Split</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {samples.map((s) => (
                <tr key={s.id}>
                  <td className="max-w-lg px-3 py-2 text-slate-800">{s.text}</td>
                  <td className="px-3 py-2 text-slate-600">{s.language}</td>
                  <td className="px-3 py-2 font-medium text-slate-800">{s.label}</td>
                  <td className="px-3 py-2 text-slate-600">{s.split}</td>
                  <td className="px-3 py-2 text-slate-500">{s.translation_status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
