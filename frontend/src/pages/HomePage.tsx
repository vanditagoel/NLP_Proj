import { ArrowRight, CheckCircle2, Database, FlaskConical, MinusCircle, XCircle } from 'lucide-react';
import type { NavTab, ProjectOverview } from '../types';

interface HomePageProps {
  project: ProjectOverview | null;
  onNavigate: (tab: NavTab) => void;
}

export function HomePage({ project, onNavigate }: HomePageProps) {
  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs sm:p-8">
        <p className="text-xs font-semibold uppercase tracking-wider text-teal-700">
          {project?.course_code || 'CSET 346'} · University NLP &amp; ML Project
        </p>
        <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
          {project?.project_title || 'Detecting Stance in Media on Global Warming for Indic Languages'}
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-600 sm:text-base">
          Multilingual stance classification pipeline evaluating Hindi, Bengali, and English social media comments
          toward the target claim:
        </p>
        <blockquote className="mt-3 rounded-lg border-l-4 border-teal-700 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-800">
          &ldquo;{project?.target_claim || 'Climate change and global warming is a serious concern.'}&rdquo;
        </blockquote>

        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => onNavigate('testing')}
            className="inline-flex items-center gap-2 rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-teal-800"
          >
            <FlaskConical className="h-4 w-4" />
            Test Models &amp; Batch CSV
            <ArrowRight className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onNavigate('models')}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            View Model Benchmarks
          </button>
          <button
            type="button"
            onClick={() => onNavigate('dataset')}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            <Database className="h-4 w-4" />
            Dataset &amp; Inspection
          </button>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium text-slate-500">Prepared Dataset Size</p>
          <p className="mt-1.5 text-2xl font-bold text-slate-900 tabular-nums">
            {project && project.total_samples > 0 ? `${project.total_samples.toLocaleString()} rows` : 'Not prepared'}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Source: {project?.dataset_source || 'mlburnham/Pol_NLI'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium text-slate-500">Trained Models Ready</p>
          <p className="mt-1.5 text-2xl font-bold text-slate-900 tabular-nums">
            {project ? `${project.trained_models_count} / ${project.total_models_count ?? 4}` : '0 / 4'}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            MuRIL, mBERT, XLM-RoBERTa &amp; IndicBERTv2 Transformers
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium text-slate-500">Best Model (Macro F1)</p>
          <p className="mt-1.5 text-lg font-bold text-teal-800">
            {project?.best_model_name || 'No model evaluated yet'}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Ranked strictly by held-out Test Macro F1
          </p>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-emerald-900">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            Favour
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-emerald-800">
            Comment supports or agrees that climate change and global warming is a serious concern.
          </p>
        </div>
        <div className="rounded-xl border border-rose-200 bg-rose-50/50 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-rose-900">
            <XCircle className="h-4 w-4 text-rose-600" />
            Against
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-rose-800">
            Comment rejects, denies, or minimizes the claim that global warming is a serious concern.
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <MinusCircle className="h-4 w-4 text-slate-500" />
            None
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
            Comment is neutral, descriptive, off-topic, or expresses no clear stance on the claim.
          </p>
        </div>
      </section>
    </div>
  );
}
