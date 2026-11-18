export type StanceLabel = 'Favour' | 'Against' | 'None';

export type NavTab = 'home' | 'testing' | 'models' | 'dataset';

export interface ProjectOverview {
  project_title: string;
  course_code: string;
  target_claim: string;
  labels: StanceLabel[];
  languages: string[];
  dataset_source: string;
  best_model_key: string | null;
  best_model_name: string | null;
  total_samples: number;
  trained_models_count: number;
  total_models_count?: number;
}

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

export interface PredictionResult {
  text: string;
  cleaned_text: string;
  detected_language: string;
  detected_language_code: string;
  selected_language: string;
  model_key: string;
  model_name: string;
  stance: StanceLabel;
  confidence: number;
  probabilities: Record<StanceLabel, number>;
  processing_time_ms: number;
}

export interface CompareResponse {
  text: string;
  detected_language: string;
  predictions: PredictionResult[];
  untrained: Array<{ model_key: string; error: string }>;
  agreement: {
    status: 'unanimous' | 'disagreement';
    consensus: StanceLabel | null;
    summary: string;
  };
}

export interface BatchResponse {
  model_key: string;
  selected_models: string[];
  total_rows: number;
  counts: Record<StanceLabel, number>;
  columns: string[];
  rows: Array<Record<string, string | number>>;
}

export interface DatasetStats {
  dataset_path: string;
  total_samples: number;
  split_distribution: Record<string, number>;
  class_distribution: Record<string, number>;
  language_distribution: Record<string, number>;
  missing_values: Record<string, number>;
  duplicate_texts: number;
  text_length_chars: { mean: number; median: number; min: number; max: number };
  text_length_words: { mean: number; median: number; min: number; max: number };
  language_by_class: Record<string, Record<string, number>>;
  translation_status: Record<string, number>;
  charts: string[];
  prepare_meta?: {
    dataset_origin: string;
    unique_groups: number;
    leakage_verified: boolean;
  };
  pol_nli_inspection?: {
    dataset_id: string;
    total_rows: number;
    global_warming_subset?: {
      total_nli_rows: number;
      unique_premises: number;
      reconstructed_total: number;
      excluded_ambiguous_or_partial_premises: number;
    };
    compatibility_analysis?: {
      task_formulation: string;
      directly_compatible_raw_rows: boolean;
      reason: string;
      legitimate_resolution: string;
    };
  };
  pol_nli_auxiliary?: {
    total_rows: number;
    labels: Record<string, number>;
    note: string;
  };
}

export interface DatasetSample {
  id: string;
  text: string;
  original_text: string;
  language: string;
  label: string;
  split: string;
  translation_status: string;
  original_source: string;
}
