import fs from 'node:fs';
import path from 'node:path';
import Papa from 'papaparse';
import {
  CHART_COLORS,
  saveConfusionMatrixChartPng,
  saveGroupedBarChartPng,
  saveModelComparisonChartPng,
} from './server_charts.ts';
import {
  LABELS,
  NeuralTransformerClassifier,
  TRANSFORMER_SPECS,
  downloadHfVocabTokens,
  trainTransformerOnSplits,
} from './server_transformer.ts';
import type { StanceLabel } from './server_transformer.ts';

export { LABELS };
export type { StanceLabel };

const ROOT_DIR = process.cwd();
const CONFIG_PATH = path.join(ROOT_DIR, 'configs', 'default_config.json');
const PROCESSED_CSV_PATH = path.join(ROOT_DIR, 'data', 'processed', 'climate_stance_dataset.csv');
const SPLITS_DIR = path.join(ROOT_DIR, 'data', 'splits');
const MODELS_DIR = path.join(ROOT_DIR, 'models');
const HF_VOCAB_CACHE_DIR = path.join(MODELS_DIR, '_hf_vocabs');
const METRICS_DIR = path.join(ROOT_DIR, 'results', 'metrics');
export const CHARTS_DIR = path.join(ROOT_DIR, 'results', 'charts');

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetrics {
  model_key: string;
  model_name: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  precision_macro: number;
  recall_macro: number;
  per_class: Record<StanceLabel, PerClassMetric>;
  confusion_matrix: number[][];
  labels: StanceLabel[];
  language_metrics: Record<string, { samples: number; accuracy: number; macro_f1: number }>;
  avg_inference_ms: number;
  test_samples: number;
  training_seconds?: number;
}

export interface RegisteredModel {
  model_key: string;
  name: string;
  family: string;
  kind: 'classical' | 'transformer';
  hf_id?: string;
  description: string;
  trained: boolean;
  status: string;
  metrics: ModelMetrics | null;
}

export interface EvaluationSummary {
  best_model_key: string | null;
  best_model_name: string | null;
  models: ModelMetrics[];
}

const URL_RE = /https?:\/\/\S+|www\.\S+/gi;
const MENTION_RE = /@[A-Za-z0-9_]+/g;
const HASHTAG_RE = /#(\S+)/g;
const MULTISPACE_RE = /\s+/g;

const DEVANAGARI_RE = /[\u0900-\u097F]/g;
const BENGALI_RE = /[\u0980-\u09FF]/g;
const LATIN_RE = /[A-Za-z]/g;

export function cleanText(text: string): string {
  if (typeof text !== 'string') return '';
  return text
    .normalize('NFC')
    .replace(URL_RE, ' ')
    .replace(MENTION_RE, ' ')
    .replace(HASHTAG_RE, '$1')
    .replace(MULTISPACE_RE, ' ')
    .trim();
}

export function detectScriptLanguage(text: string): { language: string; code: string } {
  if (typeof text !== 'string' || !text.trim()) {
    return { language: 'English', code: 'en' };
  }
  const hiCount = (text.match(DEVANAGARI_RE) || []).length;
  const bnCount = (text.match(BENGALI_RE) || []).length;
  const enCount = (text.match(LATIN_RE) || []).length;

  if (hiCount >= 2 && hiCount >= bnCount) return { language: 'Hindi', code: 'hi' };
  if (bnCount >= 2 && bnCount > hiCount) return { language: 'Bengali', code: 'bn' };
  if (hiCount > 0 && hiCount >= bnCount && enCount === 0) return { language: 'Hindi', code: 'hi' };
  if (bnCount > 0 && bnCount > hiCount && enCount === 0) return { language: 'Bengali', code: 'bn' };
  return { language: 'English', code: 'en' };
}

function round4(val: number): number {
  return Math.round(val * 10000) / 10000;
}

function argmaxLabel(probMap: Record<StanceLabel, number>): StanceLabel {
  let best: StanceLabel = 'Favour';
  let bestVal = -1;
  for (const lbl of LABELS) {
    if (probMap[lbl] > bestVal) {
      bestVal = probMap[lbl];
      best = lbl;
    }
  }
  return best;
}

interface DatasetCsvRow {
  id?: string;
  text?: string;
  original_text?: string;
  language?: string;
  label?: string;
  split?: string;
  translation_status?: string;
  original_source?: string;
}

function readCsvFile<T = Record<string, string>>(filePath: string): T[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, 'utf-8');
  const parsed = Papa.parse<T>(raw, {
    header: true,
    skipEmptyLines: true,
  });
  return parsed.data;
}

function readJsonFile<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function computeClassificationMetrics(
  yTrue: StanceLabel[],
  yPred: StanceLabel[],
  languages: string[] | null,
  modelKey: string,
  modelName: string,
  avgInferenceMs: number,
  trainingSeconds: number
): ModelMetrics {
  const n = Math.max(1, yTrue.length);
  let correct = 0;
  const cm: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];

  for (let i = 0; i < yTrue.length; i++) {
    const tIdx = LABELS.indexOf(yTrue[i]);
    const pIdx = LABELS.indexOf(yPred[i]);
    if (tIdx >= 0 && pIdx >= 0) {
      cm[tIdx][pIdx] += 1;
      if (tIdx === pIdx) correct += 1;
    }
  }

  const perClass = {} as Record<StanceLabel, PerClassMetric>;
  let macroF1Sum = 0;
  let weightedF1Sum = 0;
  let macroPrecSum = 0;
  let macroRecSum = 0;

  for (let c = 0; c < 3; c++) {
    const lbl = LABELS[c];
    const tp = cm[c][c];
    const fp = cm[0][c] + cm[1][c] + cm[2][c] - tp;
    const fn = cm[c][0] + cm[c][1] + cm[c][2] - tp;
    const support = cm[c][0] + cm[c][1] + cm[c][2];

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    perClass[lbl] = {
      precision: round4(precision),
      recall: round4(recall),
      f1: round4(f1),
      support,
    };

    macroF1Sum += f1;
    weightedF1Sum += f1 * support;
    macroPrecSum += precision;
    macroRecSum += recall;
  }

  const languageMetrics: Record<string, { samples: number; accuracy: number; macro_f1: number }> = {};
  if (languages && languages.length === yTrue.length) {
    const uniqueLangs = Array.from(new Set(languages)).sort();
    for (const lang of uniqueLangs) {
      const idxs: number[] = [];
      for (let i = 0; i < languages.length; i++) {
        if (languages[i] === lang) idxs.push(i);
      }
      if (idxs.length === 0) continue;
      let langCorrect = 0;
      const lcm = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ];
      for (const idx of idxs) {
        const tIdx = LABELS.indexOf(yTrue[idx]);
        const pIdx = LABELS.indexOf(yPred[idx]);
        if (tIdx >= 0 && pIdx >= 0) {
          lcm[tIdx][pIdx] += 1;
          if (tIdx === pIdx) langCorrect += 1;
        }
      }
      let lMacroF1 = 0;
      for (let c = 0; c < 3; c++) {
        const tp = lcm[c][c];
        const fp = lcm[0][c] + lcm[1][c] + lcm[2][c] - tp;
        const fn = lcm[c][0] + lcm[c][1] + lcm[c][2] - tp;
        const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
        const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
        const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
        lMacroF1 += f1;
      }
      languageMetrics[lang] = {
        samples: idxs.length,
        accuracy: round4(langCorrect / idxs.length),
        macro_f1: round4(lMacroF1 / 3),
      };
    }
  }

  return {
    model_key: modelKey,
    model_name: modelName,
    accuracy: round4(correct / n),
    macro_f1: round4(macroF1Sum / 3),
    weighted_f1: round4(weightedF1Sum / n),
    precision_macro: round4(macroPrecSum / 3),
    recall_macro: round4(macroRecSum / 3),
    per_class: perClass,
    confusion_matrix: cm,
    labels: [...LABELS],
    language_metrics: languageMetrics,
    avg_inference_ms: round4(avgInferenceMs),
    test_samples: yTrue.length,
    training_seconds: trainingSeconds,
  };
}

const loadedTransformers = new Map<string, NeuralTransformerClassifier>();
let evaluationSummaryCache: EvaluationSummary | null = null;

export const trainStatus = {
  running: false,
  current_model: null as string | null,
  last_completed: [] as string[],
  last_error: null as string | null,
  started_at: null as number | null,
  finished_at: null as number | null,
};

function isModelSaved(modelKey: string): boolean {
  const dir = path.join(MODELS_DIR, modelKey);
  return (
    fs.existsSync(path.join(dir, 'config.json')) &&
    fs.existsSync(path.join(dir, 'weights.bin'))
  );
}

function getOrLoadTransformer(modelKey: string): NeuralTransformerClassifier {
  const cached = loadedTransformers.get(modelKey);
  if (cached) return cached;

  const spec = TRANSFORMER_SPECS[modelKey];
  if (!spec) {
    throw new Error(`Unknown model key '${modelKey}'.`);
  }
  const modelDir = path.join(MODELS_DIR, modelKey);
  if (!isModelSaved(modelKey)) {
    throw new ModelNotTrainedError('Model not trained');
  }
  const model = NeuralTransformerClassifier.load(modelDir, spec);
  loadedTransformers.set(modelKey, model);
  return model;
}

export function initializeMlService(): void {
  const precomputed = readJsonFile<EvaluationSummary>(
    path.join(METRICS_DIR, 'evaluation_summary.json')
  );
  if (precomputed) {
    evaluationSummaryCache = precomputed;
  }

  for (const key of Object.keys(TRANSFORMER_SPECS)) {
    if (isModelSaved(key)) {
      try {
        getOrLoadTransformer(key);
      } catch (err) {
        console.warn(`Failed to load saved model ${key}:`, err);
      }
    }
  }
}

export function getEvaluationSummary(): EvaluationSummary {
  if (evaluationSummaryCache) return evaluationSummaryCache;
  const loaded = readJsonFile<EvaluationSummary>(
    path.join(METRICS_DIR, 'evaluation_summary.json')
  );
  if (loaded) {
    evaluationSummaryCache = loaded;
    return loaded;
  }
  return { best_model_key: null, best_model_name: null, models: [] };
}

export function listModels(): RegisteredModel[] {
  const evalSummary = getEvaluationSummary();
  const metricsByKey = new Map(evalSummary.models.map((m) => [m.model_key, m]));
  const models: RegisteredModel[] = [];

  for (const [key, spec] of Object.entries(TRANSFORMER_SPECS)) {
    const trained = loadedTransformers.has(key) || isModelSaved(key);
    models.push({
      model_key: key,
      name: spec.name,
      family: spec.family,
      kind: 'transformer',
      hf_id: spec.hf_id,
      description: spec.description,
      trained,
      status: trained ? 'Trained' : 'Model not trained',
      metrics: metricsByKey.get(key) || null,
    });
  }

  return models;
}

export function getDatasetStats(): Record<string, unknown> | null {
  const stats = readJsonFile<Record<string, unknown>>(path.join(METRICS_DIR, 'dataset_stats.json'));
  if (!stats) return null;

  const prepareMeta = readJsonFile<Record<string, unknown>>(
    path.join(METRICS_DIR, 'dataset_prepare_meta.json')
  );
  const polInspect = readJsonFile<Record<string, unknown>>(
    path.join(METRICS_DIR, 'pol_nli_inspection.json')
  );
  const polAux = readJsonFile<Record<string, unknown>>(
    path.join(METRICS_DIR, 'pol_nli_prepare_meta.json')
  );

  if (prepareMeta) stats.prepare_meta = prepareMeta;
  if (polInspect) stats.pol_nli_inspection = polInspect;
  if (polAux) stats.pol_nli_auxiliary = polAux;

  const charts = fs.existsSync(CHARTS_DIR)
    ? fs.readdirSync(CHARTS_DIR).filter((f) => f.endsWith('.png')).sort()
    : [];
  stats.charts = charts;
  return stats;
}

export function getProjectOverview(): Record<string, unknown> {
  const cfg = readJsonFile<Record<string, unknown>>(CONFIG_PATH) || {
    project_title: 'Detecting Stance in Media on Global Warming for Indic Languages',
    course_code: 'CSET 346',
    target_claim: 'Climate change and global warming is a serious concern.',
    labels: LABELS,
    languages: ['English', 'Hindi', 'Bengali'],
    dataset_source: 'mlburnham/Pol_NLI',
  };
  const evalData = getEvaluationSummary();
  const stats = getDatasetStats();
  const allModels = listModels();
  const trainedCount = allModels.filter((m) => m.trained).length;

  return {
    project_title: cfg.project_title,
    course_code: cfg.course_code,
    target_claim: cfg.target_claim,
    labels: cfg.labels,
    languages: cfg.languages,
    dataset_source:
      (stats?.prepare_meta as { dataset_origin?: string } | undefined)?.dataset_origin ||
      cfg.dataset_source,
    best_model_key: evalData.best_model_key,
    best_model_name: evalData.best_model_name,
    total_samples: typeof stats?.total_samples === 'number' ? stats.total_samples : 0,
    trained_models_count: trainedCount,
    total_models_count: allModels.length,
  };
}

export function getDatasetSamples(params: {
  language?: string;
  label?: string;
  split?: string;
  limit?: number;
}): Array<Record<string, string>> {
  const rows = readCsvFile<DatasetCsvRow>(PROCESSED_CSV_PATH);
  const limit = Number.isFinite(params.limit) && (params.limit || 0) > 0 ? Number(params.limit) : 25;

  return rows
    .filter((r) => {
      if (params.language && params.language !== 'All' && r.language !== params.language) return false;
      if (params.label && params.label !== 'All' && r.label !== params.label) return false;
      if (params.split && params.split !== 'All' && r.split !== params.split) return false;
      return true;
    })
    .slice(0, limit)
    .map((r) => ({
      id: r.id || '',
      text: r.text || '',
      original_text: r.original_text || '',
      language: r.language || '',
      label: r.label || '',
      split: r.split || '',
      translation_status: r.translation_status || '',
      original_source: r.original_source || '',
    }));
}

export class ModelNotTrainedError extends Error {
  constructor(message = 'Model not trained') {
    super(message);
    this.name = 'ModelNotTrainedError';
  }
}

export function predictSingle(
  text: string,
  language = 'Auto',
  modelKey = 'muril'
): Record<string, unknown> {
  const cleaned = cleanText(text);
  if (!cleaned) {
    throw new Error('Input text cannot be empty.');
  }

  const { language: detLang, code: detCode } = detectScriptLanguage(cleaned);
  const resolvedLang = !language || language === 'Auto' ? detLang : language;

  const spec = TRANSFORMER_SPECS[modelKey];
  if (!spec) {
    throw new Error(`Unknown model key '${modelKey}'.`);
  }
  const model = getOrLoadTransformer(modelKey);

  const t0 = performance.now();
  const probMap = model.predictProba(cleaned);
  const stance = argmaxLabel(probMap);
  const elapsedMs = Math.round((performance.now() - t0) * 100) / 100;
  const confidence = round4(Math.max(...Object.values(probMap)));

  return {
    text,
    cleaned_text: cleaned,
    detected_language: detLang,
    detected_language_code: detCode,
    selected_language: resolvedLang,
    model_key: modelKey,
    model_name: spec.name,
    stance,
    confidence,
    probabilities: probMap,
    processing_time_ms: elapsedMs,
  };
}

export function predictBatchTexts(
  texts: string[],
  modelKey: string
): Array<{
  text: string;
  detected_language: string;
  stance: StanceLabel;
  confidence: number;
  probabilities: Record<StanceLabel, number>;
  processing_time_ms: number;
}> {
  if (!(modelKey in TRANSFORMER_SPECS)) {
    throw new Error(`Unknown model key '${modelKey}'.`);
  }
  const model = getOrLoadTransformer(modelKey);

  const t0 = performance.now();
  const probsList = texts.map((raw) => {
    const cleaned = cleanText(raw);
    return model.predictProba(cleaned || raw);
  });
  const totalMs = performance.now() - t0;
  const perRowMs = Math.round((totalMs / Math.max(1, texts.length)) * 100) / 100;

  return texts.map((raw, idx) => {
    const cleaned = cleanText(raw);
    const { language: detLang } = detectScriptLanguage(cleaned);
    const probMap = probsList[idx];
    const stance = argmaxLabel(probMap);
    return {
      text: raw,
      detected_language: detLang,
      stance,
      confidence: round4(Math.max(...Object.values(probMap))),
      probabilities: probMap,
      processing_time_ms: perRowMs,
    };
  });
}

export function predictCompare(
  text: string,
  language = 'Auto',
  modelKeys?: string[] | null
): Record<string, unknown> {
  const trainedKeys = listModels()
    .filter((m) => m.trained)
    .map((m) => m.model_key);
  const targetKeys = modelKeys && modelKeys.length > 0 ? modelKeys : trainedKeys;
  if (targetKeys.length === 0) {
    throw new ModelNotTrainedError('Model not trained');
  }

  const predictions: Array<Record<string, unknown>> = [];
  const untrained: Array<{ model_key: string; error: string }> = [];

  for (const key of targetKeys) {
    try {
      predictions.push(predictSingle(text, language, key));
    } catch (err) {
      if (err instanceof ModelNotTrainedError) {
        untrained.push({ model_key: key, error: 'Model not trained' });
      } else {
        throw err;
      }
    }
  }

  if (predictions.length === 0) {
    throw new ModelNotTrainedError('Model not trained');
  }

  const stances = predictions.map((p) => String(p.stance));
  const uniqueStances = Array.from(new Set(stances));
  let agreement: {
    status: 'unanimous' | 'disagreement';
    consensus: string | null;
    summary: string;
  };

  if (uniqueStances.length === 1) {
    agreement = {
      status: 'unanimous',
      consensus: stances[0],
      summary: `All ${predictions.length} models agree on '${stances[0]}'.`,
    };
  } else {
    const counts = new Map<string, number>();
    for (const s of stances) counts.set(s, (counts.get(s) || 0) + 1);
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    const [topS, topC] = sorted[0];
    agreement = {
      status: 'disagreement',
      consensus: topC > predictions.length / 2 ? topS : null,
      summary: `Models disagree (${sorted.map(([k, v]) => `${k}: ${v}`).join(', ')}).`,
    };
  }

  return {
    text,
    detected_language: predictions[0].detected_language,
    predictions,
    untrained,
    agreement,
  };
}

export function startBackgroundTraining(modelKeys: string[]): Record<string, unknown> {
  if (trainStatus.running) {
    return {
      started: false,
      reason: 'Training already in progress',
      status: { ...trainStatus },
    };
  }

  const targetKeys = modelKeys.filter((k) => k in TRANSFORMER_SPECS);
  const resolvedKeys = targetKeys.length > 0 ? targetKeys : Object.keys(TRANSFORMER_SPECS);

  trainStatus.running = true;
  trainStatus.current_model = resolvedKeys.join(', ');
  trainStatus.last_completed = [];
  trainStatus.last_error = null;
  trainStatus.started_at = Date.now() / 1000;
  trainStatus.finished_at = null;

  void (async () => {
    try {
      const trainRows = readCsvFile<DatasetCsvRow>(path.join(SPLITS_DIR, 'climate_train.csv'));
      const valRows = readCsvFile<DatasetCsvRow>(path.join(SPLITS_DIR, 'climate_validation.csv'));
      const testRows = readCsvFile<DatasetCsvRow>(path.join(SPLITS_DIR, 'climate_test.csv'));
      if (trainRows.length === 0 || testRows.length === 0) {
        throw new Error('Prepared climate_train.csv or climate_test.csv splits not found.');
      }

      const trainTexts = trainRows.map((r) => String(r.text || ''));
      const trainLabels = trainRows.map((r) => (r.label as StanceLabel) || 'None');
      const valTexts = valRows.map((r) => String(r.text || ''));
      const valLabels = valRows.map((r) => (r.label as StanceLabel) || 'None');
      const testTexts = testRows.map((r) => String(r.text || ''));
      const testLabels = testRows.map((r) => (r.label as StanceLabel) || 'None');
      const testLangs = testRows.map((r) => String(r.language || 'English'));

      for (const mk of resolvedKeys) {
        trainStatus.current_model = mk;
        const spec = TRANSFORMER_SPECS[mk];
        const hfVocab = await downloadHfVocabTokens(spec, HF_VOCAB_CACHE_DIR);
        const outDir = path.join(MODELS_DIR, mk);
        const trainRes = trainTransformerOnSplits(
          spec,
          hfVocab,
          trainTexts,
          trainLabels,
          valTexts,
          valLabels,
          outDir
        );
        loadedTransformers.set(mk, trainRes.model);
        trainStatus.last_completed.push(mk);
      }

      const evaluatedModels: ModelMetrics[] = [];
      for (const [key, spec] of Object.entries(TRANSFORMER_SPECS)) {
        if (!isModelSaved(key)) continue;
        const model = getOrLoadTransformer(key);
        const t0 = performance.now();
        const preds = testTexts.map((txt) => argmaxLabel(model.predictProba(cleanText(txt) || txt)));
        const avgInfMs = (performance.now() - t0) / Math.max(1, testTexts.length);
        const meta = readJsonFile<{ training_seconds?: number }>(
          path.join(MODELS_DIR, key, 'training_meta.json')
        );
        const m = computeClassificationMetrics(
          testLabels,
          preds,
          testLangs,
          key,
          spec.name,
          avgInfMs,
          meta?.training_seconds || 0
        );
        evaluatedModels.push(m);
        saveConfusionMatrixChartPng(
          path.join(CHARTS_DIR, `cm_${key}.png`),
          spec.name,
          LABELS,
          m.confusion_matrix
        );
      }

      evaluatedModels.sort((a, b) => b.macro_f1 - a.macro_f1);
      const best = evaluatedModels[0] || null;
      evaluationSummaryCache = {
        best_model_key: best ? best.model_key : null,
        best_model_name: best ? best.model_name : null,
        models: evaluatedModels,
      };
      fs.writeFileSync(
        path.join(METRICS_DIR, 'evaluation_summary.json'),
        JSON.stringify(evaluationSummaryCache, null, 2)
      );
      saveModelComparisonChartPng(path.join(CHARTS_DIR, 'model_comparison.png'), evaluatedModels);
      saveGroupedBarChartPng(
        path.join(CHARTS_DIR, 'language_f1_comparison.png'),
        'Language-Wise Macro F1 (%)',
        evaluatedModels.map((m) => m.model_key.toUpperCase()),
        [
          {
            name: 'English',
            values: evaluatedModels.map((m) => (m.language_metrics.English?.macro_f1 || 0) * 100),
            color: CHART_COLORS.SLATE_600,
          },
          {
            name: 'Hindi',
            values: evaluatedModels.map((m) => (m.language_metrics.Hindi?.macro_f1 || 0) * 100),
            color: CHART_COLORS.TEAL_700,
          },
          {
            name: 'Bengali',
            values: evaluatedModels.map((m) => (m.language_metrics.Bengali?.macro_f1 || 0) * 100),
            color: CHART_COLORS.SKY_600,
          },
        ],
        '%'
      );
    } catch (err) {
      trainStatus.last_error = err instanceof Error ? err.message : String(err);
    } finally {
      trainStatus.running = false;
      trainStatus.current_model = null;
      trainStatus.finished_at = Date.now() / 1000;
    }
  })();

  return {
    started: true,
    status: { ...trainStatus },
  };
}
