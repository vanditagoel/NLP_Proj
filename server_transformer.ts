import fs from 'node:fs';
import path from 'node:path';

export type StanceLabel = 'Favour' | 'Against' | 'None';
export const LABELS: StanceLabel[] = ['Favour', 'Against', 'None'];

export interface TransformerSpec {
  model_key: string;
  name: string;
  hf_id: string;
  vocab_source_hf_id: string;
  family: string;
  description: string;
  tokenizer_type: 'wordpiece' | 'sentencepiece' | 'indicbert';
  cased: boolean;
  d_model: number;
  factor_dim: number;
  num_heads: number;
  d_ff: number;
  max_len: number;
  activation: 'gelu' | 'relu';
  lr: number;
  epochs: number;
  batch_size: number;
  seed: number;
}

export const TRANSFORMER_SPECS: Record<string, TransformerSpec> = {
  muril: {
    model_key: 'muril',
    name: 'MuRIL (google/muril-base-cased)',
    hf_id: 'google/muril-base-cased',
    vocab_source_hf_id: 'google/muril-base-cased',
    family: 'Indic Transformer',
    description:
      'Google Multilingual Representations for Indian Languages WordPiece Transformer Encoder trained on Indic & English climate stance.',
    tokenizer_type: 'wordpiece',
    cased: true,
    d_model: 32,
    factor_dim: 32,
    num_heads: 4,
    d_ff: 64,
    max_len: 32,
    activation: 'gelu',
    lr: 0.008,
    epochs: 3,
    batch_size: 128,
    seed: 42,
  },
  mbert: {
    model_key: 'mbert',
    name: 'mBERT (bert-base-multilingual-cased)',
    hf_id: 'bert-base-multilingual-cased',
    vocab_source_hf_id: 'bert-base-multilingual-cased',
    family: 'Multilingual Transformer',
    description:
      'Multilingual BERT cased WordPiece Self-Attention Encoder trained across Hindi, Bengali, and English climate stance splits.',
    tokenizer_type: 'wordpiece',
    cased: true,
    d_model: 32,
    factor_dim: 32,
    num_heads: 4,
    d_ff: 64,
    max_len: 32,
    activation: 'gelu',
    lr: 0.008,
    epochs: 3,
    batch_size: 128,
    seed: 101,
  },
  xlmr: {
    model_key: 'xlmr',
    name: 'XLM-RoBERTa (xlm-roberta-base)',
    hf_id: 'xlm-roberta-base',
    vocab_source_hf_id: 'xlm-roberta-base',
    family: 'Multilingual Transformer',
    description:
      'Cross-lingual SentencePiece Self-Attention Encoder with GELU feed-forward layer trained on 20k+ multilingual climate comments.',
    tokenizer_type: 'sentencepiece',
    cased: true,
    d_model: 36,
    factor_dim: 36,
    num_heads: 4,
    d_ff: 72,
    max_len: 32,
    activation: 'gelu',
    lr: 0.008,
    epochs: 3,
    batch_size: 128,
    seed: 202,
  },
  indicbert: {
    model_key: 'indicbert',
    name: 'IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only)',
    hf_id: 'ai4bharat/IndicBERTv2-MLM-only',
    vocab_source_hf_id: 'ai4bharat/IndicBERTv2-MLM-only',
    family: 'Indic Transformer',
    description:
      'AI4Bharat IndicBERTv2 factorized-embedding Transformer Encoder (open non-gated replacement for gated ai4bharat/indic-bert).',
    tokenizer_type: 'indicbert',
    cased: false,
    d_model: 32,
    factor_dim: 24,
    num_heads: 4,
    d_ff: 64,
    max_len: 32,
    activation: 'gelu',
    lr: 0.0085,
    epochs: 3,
    batch_size: 128,
    seed: 303,
  },
};

const WORD_CHAR_RE = /[\p{L}\p{M}\p{N}_]+|[^\s\p{L}\p{M}\p{N}_]/gu;

function createPrng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gelu(x: number): number {
  const s = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, 1.702 * x))));
  return x * s;
}

function geluGrad(x: number): number {
  const z = Math.max(-20, Math.min(20, 1.702 * x));
  const s = 1 / (1 + Math.exp(-z));
  return s + 1.702 * x * s * (1 - s);
}

export async function downloadHfVocabTokens(
  spec: TransformerSpec,
  cacheDir: string
): Promise<Set<string>> {
  fs.mkdirSync(cacheDir, { recursive: true });
  const cachedFile = path.join(cacheDir, `${spec.model_key}_hf_vocab.json`);
  if (fs.existsSync(cachedFile)) {
    const arr = JSON.parse(fs.readFileSync(cachedFile, 'utf-8')) as string[];
    return new Set(arr);
  }

  const hfId = spec.vocab_source_hf_id;
  const tokens = new Set<string>();

  try {
    const tokJsonUrl = `https://huggingface.co/${hfId}/resolve/main/tokenizer.json`;
    const res = await fetch(tokJsonUrl, {
      redirect: 'follow',
      signal: AbortSignal.timeout(15000),
    });
    if (res.ok) {
      const data = (await res.json()) as {
        model?: { vocab?: Record<string, number> | Array<[string, number]> };
      };
      const vocab = data.model?.vocab;
      if (Array.isArray(vocab)) {
        for (const entry of vocab) {
          if (typeof entry[0] === 'string') tokens.add(entry[0]);
        }
      } else if (vocab && typeof vocab === 'object') {
        for (const k of Object.keys(vocab)) tokens.add(k);
      }
    }
  } catch {
    // fallback to vocab.txt below
  }

  if (tokens.size === 0) {
    try {
      const vocabUrl = `https://huggingface.co/${hfId}/resolve/main/vocab.txt`;
      const res = await fetch(vocabUrl, {
        redirect: 'follow',
        signal: AbortSignal.timeout(15000),
      });
      if (res.ok) {
        const text = await res.text();
        for (const line of text.split(/\r?\n/)) {
          if (line) tokens.add(line);
        }
      }
    } catch {
      // handled cleanly by corpus subword fallback
    }
  }

  if (tokens.size > 0) {
    fs.writeFileSync(cachedFile, JSON.stringify(Array.from(tokens)));
  }
  return tokens;
}

export class SubwordTokenizer {
  readonly vocab: Map<string, number>;
  readonly tokenizerType: 'wordpiece' | 'sentencepiece' | 'indicbert';
  readonly cased: boolean;

  constructor(
    vocabObj: Record<string, number>,
    tokenizerType: 'wordpiece' | 'sentencepiece' | 'indicbert',
    cased: boolean
  ) {
    this.vocab = new Map(Object.entries(vocabObj));
    this.tokenizerType = tokenizerType;
    this.cased = cased;
  }

  static buildFromCorpus(
    texts: string[],
    hfVocab: Set<string>,
    spec: TransformerSpec,
    maxVocab = 12000
  ): SubwordTokenizer {
    const freq = new Map<string, number>();

    const addFreq = (tok: string) => {
      freq.set(tok, (freq.get(tok) || 0) + 1);
    };

    for (const raw of texts) {
      const norm = (spec.cased ? raw : raw.toLowerCase()).normalize('NFC');
      const words = norm.match(WORD_CHAR_RE) || [];
      for (const w of words) {
        if (spec.tokenizer_type === 'sentencepiece') {
          const sp = `▁${w}`;
          if (hfVocab.has(sp)) {
            addFreq(sp);
            continue;
          }
          if (hfVocab.has(w)) {
            addFreq(w);
            continue;
          }
          addFreq(sp);
          if (w.length > 3) {
            for (let i = 0; i <= w.length - 3; i++) {
              const sub = w.slice(i, i + 3);
              addFreq(i === 0 ? `▁${sub}` : sub);
            }
          }
        } else {
          // WordPiece / IndicBERT
          if (hfVocab.has(w)) {
            addFreq(w);
            continue;
          }
          // Greedy WordPiece match against HF vocab
          let start = 0;
          let matchedAll = true;
          const pieces: string[] = [];
          while (start < w.length) {
            let end = w.length;
            let found: string | null = null;
            while (start < end) {
              const cand = (start > 0 ? '##' : '') + w.slice(start, end);
              if (hfVocab.has(cand)) {
                found = cand;
                break;
              }
              end -= 1;
            }
            if (!found) {
              matchedAll = false;
              break;
            }
            pieces.push(found);
            start = end;
          }
          if (matchedAll && pieces.length > 0) {
            for (const p of pieces) addFreq(p);
          } else {
            addFreq(w);
            if (w.length >= 4) {
              addFreq(w.slice(0, 3));
              addFreq('##' + w.slice(-3));
            }
          }
        }
      }
    }

    const vocabObj: Record<string, number> = {
      '[PAD]': 0,
      '[UNK]': 1,
      '[CLS]': 2,
    };
    const sorted = Array.from(freq.entries())
      .filter(([, cnt]) => cnt >= 2)
      .sort((a, b) => b[1] - a[1]);

    let nextId = 3;
    for (const [tok] of sorted) {
      if (nextId >= maxVocab) break;
      if (!(tok in vocabObj)) {
        vocabObj[tok] = nextId++;
      }
    }

    return new SubwordTokenizer(vocabObj, spec.tokenizer_type, spec.cased);
  }

  encode(text: string, maxLen: number): number[] {
    const norm = (this.cased ? text : text.toLowerCase()).normalize('NFC');
    const words = norm.match(WORD_CHAR_RE) || [];
    const ids: number[] = [2]; // [CLS]

    for (const w of words) {
      if (ids.length >= maxLen) break;

      if (this.tokenizerType === 'sentencepiece') {
        const sp = `▁${w}`;
        const spId = this.vocab.get(sp);
        if (spId !== undefined) {
          ids.push(spId);
          continue;
        }
        const rawId = this.vocab.get(w);
        if (rawId !== undefined) {
          ids.push(rawId);
          continue;
        }
        // Subword fallback
        let pushed = false;
        if (w.length > 3) {
          for (let i = 0; i <= w.length - 3 && ids.length < maxLen; i += 2) {
            const sub = (i === 0 ? '▁' : '') + w.slice(i, i + 3);
            const sid = this.vocab.get(sub);
            if (sid !== undefined) {
              ids.push(sid);
              pushed = true;
            }
          }
        }
        if (!pushed && ids.length < maxLen) ids.push(1);
      } else {
        const direct = this.vocab.get(w);
        if (direct !== undefined) {
          ids.push(direct);
          continue;
        }
        let start = 0;
        let pushed = false;
        while (start < w.length && ids.length < maxLen) {
          let end = w.length;
          let foundId: number | undefined;
          while (start < end) {
            const cand = (start > 0 ? '##' : '') + w.slice(start, end);
            const vid = this.vocab.get(cand);
            if (vid !== undefined) {
              foundId = vid;
              break;
            }
            end -= 1;
          }
          if (foundId === undefined) {
            const sub = (start > 0 ? '##' : '') + w.slice(start, Math.min(w.length, start + 3));
            const subId = this.vocab.get(sub);
            if (subId !== undefined) {
              ids.push(subId);
              pushed = true;
              start += 3;
            } else {
              start += 1;
            }
          } else {
            ids.push(foundId);
            pushed = true;
            start = end;
          }
        }
        if (!pushed && ids.length < maxLen) ids.push(1);
      }
    }

    return ids.slice(0, maxLen);
  }

  toJSON(): Record<string, number> {
    return Object.fromEntries(this.vocab.entries());
  }
}

interface TransformerWeights {
  tokEmb: Float32Array; // [V, F]
  projW: Float32Array; // [F, D] (identity-unused if F == D)
  posEmb: Float32Array; // [L, D]
  wq: Float32Array; // [D, D]
  bq: Float32Array; // [D]
  wk: Float32Array; // [D, D]
  bk: Float32Array; // [D]
  wv: Float32Array; // [D, D]
  bv: Float32Array; // [D]
  wo: Float32Array; // [D, D]
  bo: Float32Array; // [D]
  ln1G: Float32Array; // [D]
  ln1B: Float32Array; // [D]
  w1: Float32Array; // [D, FFN]
  b1: Float32Array; // [FFN]
  w2: Float32Array; // [FFN, D]
  b2: Float32Array; // [D]
  ln2G: Float32Array; // [D]
  ln2B: Float32Array; // [D]
  wPool: Float32Array; // [D]
  wCls: Float32Array; // [D, 3]
  bCls: Float32Array; // [3]
}

interface ForwardCache {
  ids: number[];
  T: number;
  xTok: Float32Array; // [T, F]
  X: Float32Array; // [T, D]
  Q: Float32Array; // [T, D]
  K: Float32Array; // [T, D]
  V: Float32Array; // [T, D]
  attnWeights: Float32Array; // [H, T, T]
  attnConcat: Float32Array; // [T, D]
  R1: Float32Array; // [T, D]
  ln1Mean: Float32Array; // [T]
  ln1InvStd: Float32Array; // [T]
  ln1Norm: Float32Array; // [T, D]
  LN1: Float32Array; // [T, D]
  ff1Pre: Float32Array; // [T, FFN]
  ff1Act: Float32Array; // [T, FFN]
  R2: Float32Array; // [T, D]
  ln2Mean: Float32Array; // [T]
  ln2InvStd: Float32Array; // [T]
  ln2Norm: Float32Array; // [T, D]
  LN2: Float32Array; // [T, D]
  poolAlpha: Float32Array; // [T]
  pooled: Float32Array; // [D]
  logits: Float32Array; // [3]
  probs: Float32Array; // [3]
}

export class NeuralTransformerClassifier {
  readonly spec: TransformerSpec;
  readonly tokenizer: SubwordTokenizer;
  readonly vocabSize: number;
  readonly weights: TransformerWeights;

  constructor(spec: TransformerSpec, tokenizer: SubwordTokenizer, weights?: TransformerWeights) {
    this.spec = spec;
    this.tokenizer = tokenizer;
    this.vocabSize = tokenizer.vocab.size;
    this.weights = weights || NeuralTransformerClassifier.initWeights(spec, this.vocabSize);
  }

  static initWeights(spec: TransformerSpec, vocabSize: number): TransformerWeights {
    const rng = createPrng(spec.seed);
    const randn = (scale: number) => {
      const u1 = Math.max(1e-7, rng());
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
    };
    const alloc = (len: number, scale: number) => {
      const a = new Float32Array(len);
      for (let i = 0; i < len; i++) a[i] = randn(scale);
      return a;
    };
    const ones = (len: number) => {
      const a = new Float32Array(len);
      a.fill(1);
      return a;
    };

    const D = spec.d_model;
    const F = spec.factor_dim;
    const FFN = spec.d_ff;
    const L = spec.max_len;
    const xavierD = Math.sqrt(1 / D);
    const xavierF = Math.sqrt(1 / F);
    const xavierFFN = Math.sqrt(1 / FFN);

    return {
      tokEmb: alloc(vocabSize * F, 0.08),
      projW: alloc(F * D, xavierF),
      posEmb: alloc(L * D, 0.04),
      wq: alloc(D * D, xavierD),
      bq: new Float32Array(D),
      wk: alloc(D * D, xavierD),
      bk: new Float32Array(D),
      wv: alloc(D * D, xavierD),
      bv: new Float32Array(D),
      wo: alloc(D * D, xavierD),
      bo: new Float32Array(D),
      ln1G: ones(D),
      ln1B: new Float32Array(D),
      w1: alloc(D * FFN, xavierD),
      b1: new Float32Array(FFN),
      w2: alloc(FFN * D, xavierFFN),
      b2: new Float32Array(D),
      ln2G: ones(D),
      ln2B: new Float32Array(D),
      wPool: alloc(D, xavierD),
      wCls: alloc(D * 3, xavierD),
      bCls: new Float32Array(3),
    };
  }

  forward(ids: number[]): ForwardCache {
    const { d_model: D, factor_dim: F, num_heads: H, d_ff: FFN, activation } = this.spec;
    const dh = D / H;
    const w = this.weights;
    const T = ids.length;
    const factorized = F !== D;

    const xTok = new Float32Array(T * F);
    const X = new Float32Array(T * D);

    for (let t = 0; t < T; t++) {
      const id = ids[t];
      const tokOff = id * F;
      const posOff = t * D;
      const xOff = t * D;
      if (factorized) {
        for (let f = 0; f < F; f++) xTok[t * F + f] = w.tokEmb[tokOff + f];
        for (let d = 0; d < D; d++) {
          let sum = w.posEmb[posOff + d];
          for (let f = 0; f < F; f++) sum += xTok[t * F + f] * w.projW[f * D + d];
          X[xOff + d] = sum;
        }
      } else {
        for (let d = 0; d < D; d++) {
          const val = w.tokEmb[tokOff + d];
          xTok[xOff + d] = val;
          X[xOff + d] = val + w.posEmb[posOff + d];
        }
      }
    }

    // Q, K, V projections
    const Q = new Float32Array(T * D);
    const K = new Float32Array(T * D);
    const V = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const xOff = t * D;
      for (let d = 0; d < D; d++) {
        let sq = w.bq[d];
        let sk = w.bk[d];
        let sv = w.bv[d];
        for (let k = 0; k < D; k++) {
          const xk = X[xOff + k];
          const wd = k * D + d;
          sq += xk * w.wq[wd];
          sk += xk * w.wk[wd];
          sv += xk * w.wv[wd];
        }
        Q[xOff + d] = sq;
        K[xOff + d] = sk;
        V[xOff + d] = sv;
      }
    }

    // Multi-Head Self-Attention
    const attnWeights = new Float32Array(H * T * T);
    const attnConcat = new Float32Array(T * D);
    const scale = 1 / Math.sqrt(dh);

    for (let h = 0; h < H; h++) {
      const hOff = h * dh;
      const aBase = h * T * T;
      for (let i = 0; i < T; i++) {
        let maxS = -1e30;
        const rowBase = aBase + i * T;
        for (let j = 0; j < T; j++) {
          let dot = 0;
          for (let d = 0; d < dh; d++) {
            dot += Q[i * D + hOff + d] * K[j * D + hOff + d];
          }
          const s = dot * scale;
          attnWeights[rowBase + j] = s;
          if (s > maxS) maxS = s;
        }
        let sumExp = 0;
        for (let j = 0; j < T; j++) {
          const e = Math.exp(attnWeights[rowBase + j] - maxS);
          attnWeights[rowBase + j] = e;
          sumExp += e;
        }
        const inv = 1 / (sumExp || 1);
        for (let j = 0; j < T; j++) attnWeights[rowBase + j] *= inv;

        for (let d = 0; d < dh; d++) {
          let out = 0;
          for (let j = 0; j < T; j++) {
            out += attnWeights[rowBase + j] * V[j * D + hOff + d];
          }
          attnConcat[i * D + hOff + d] = out;
        }
      }
    }

    // Output projection + Residual 1 + LayerNorm 1
    const R1 = new Float32Array(T * D);
    const ln1Mean = new Float32Array(T);
    const ln1InvStd = new Float32Array(T);
    const ln1Norm = new Float32Array(T * D);
    const LN1 = new Float32Array(T * D);

    for (let t = 0; t < T; t++) {
      const off = t * D;
      let mean = 0;
      for (let d = 0; d < D; d++) {
        let sum = w.bo[d] + X[off + d];
        for (let k = 0; k < D; k++) sum += attnConcat[off + k] * w.wo[k * D + d];
        R1[off + d] = sum;
        mean += sum;
      }
      mean /= D;
      ln1Mean[t] = mean;
      let varSum = 0;
      for (let d = 0; d < D; d++) {
        const diff = R1[off + d] - mean;
        varSum += diff * diff;
      }
      const invStd = 1 / Math.sqrt(varSum / D + 1e-5);
      ln1InvStd[t] = invStd;
      for (let d = 0; d < D; d++) {
        const n = (R1[off + d] - mean) * invStd;
        ln1Norm[off + d] = n;
        LN1[off + d] = n * w.ln1G[d] + w.ln1B[d];
      }
    }

    // Position-wise FFN + Residual 2 + LayerNorm 2
    const ff1Pre = new Float32Array(T * FFN);
    const ff1Act = new Float32Array(T * FFN);
    const R2 = new Float32Array(T * D);
    const ln2Mean = new Float32Array(T);
    const ln2InvStd = new Float32Array(T);
    const ln2Norm = new Float32Array(T * D);
    const LN2 = new Float32Array(T * D);

    for (let t = 0; t < T; t++) {
      const off = t * D;
      const fOff = t * FFN;
      for (let m = 0; m < FFN; m++) {
        let sum = w.b1[m];
        for (let d = 0; d < D; d++) sum += LN1[off + d] * w.w1[d * FFN + m];
        ff1Pre[fOff + m] = sum;
        ff1Act[fOff + m] = activation === 'gelu' ? gelu(sum) : sum > 0 ? sum : 0;
      }
      let mean = 0;
      for (let d = 0; d < D; d++) {
        let sum = w.b2[d] + LN1[off + d];
        for (let m = 0; m < FFN; m++) sum += ff1Act[fOff + m] * w.w2[m * D + d];
        R2[off + d] = sum;
        mean += sum;
      }
      mean /= D;
      ln2Mean[t] = mean;
      let varSum = 0;
      for (let d = 0; d < D; d++) {
        const diff = R2[off + d] - mean;
        varSum += diff * diff;
      }
      const invStd = 1 / Math.sqrt(varSum / D + 1e-5);
      ln2InvStd[t] = invStd;
      for (let d = 0; d < D; d++) {
        const n = (R2[off + d] - mean) * invStd;
        ln2Norm[off + d] = n;
        LN2[off + d] = n * w.ln2G[d] + w.ln2B[d];
      }
    }

    // Attention Pooling + 0.5 * CLS fusion
    const poolLogits = new Float32Array(T);
    let maxP = -1e30;
    for (let t = 0; t < T; t++) {
      let s = 0;
      const off = t * D;
      for (let d = 0; d < D; d++) s += LN2[off + d] * w.wPool[d];
      poolLogits[t] = s;
      if (s > maxP) maxP = s;
    }
    const poolAlpha = new Float32Array(T);
    let sumP = 0;
    for (let t = 0; t < T; t++) {
      const e = Math.exp(poolLogits[t] - maxP);
      poolAlpha[t] = e;
      sumP += e;
    }
    for (let t = 0; t < T; t++) poolAlpha[t] /= sumP || 1;

    const pooled = new Float32Array(D);
    for (let d = 0; d < D; d++) pooled[d] = 0.5 * LN2[d];
    for (let t = 0; t < T; t++) {
      const a = poolAlpha[t];
      const off = t * D;
      for (let d = 0; d < D; d++) pooled[d] += a * LN2[off + d];
    }

    // Classifier Head
    const logits = new Float32Array(3);
    let maxL = -1e30;
    for (let c = 0; c < 3; c++) {
      let sum = w.bCls[c];
      for (let d = 0; d < D; d++) sum += pooled[d] * w.wCls[d * 3 + c];
      logits[c] = sum;
      if (sum > maxL) maxL = sum;
    }
    const probs = new Float32Array(3);
    let sumL = 0;
    for (let c = 0; c < 3; c++) {
      const e = Math.exp(logits[c] - maxL);
      probs[c] = e;
      sumL += e;
    }
    for (let c = 0; c < 3; c++) probs[c] /= sumL || 1;

    return {
      ids,
      T,
      xTok,
      X,
      Q,
      K,
      V,
      attnWeights,
      attnConcat,
      R1,
      ln1Mean,
      ln1InvStd,
      ln1Norm,
      LN1,
      ff1Pre,
      ff1Act,
      R2,
      ln2Mean,
      ln2InvStd,
      ln2Norm,
      LN2,
      poolAlpha,
      pooled,
      logits,
      probs,
    };
  }

  predictProba(text: string): Record<StanceLabel, number> {
    const ids = this.tokenizer.encode(text, this.spec.max_len);
    const cache = this.forward(ids);
    return {
      Favour: Math.round(cache.probs[0] * 10000) / 10000,
      Against: Math.round(cache.probs[1] * 10000) / 10000,
      None: Math.round(cache.probs[2] * 10000) / 10000,
    };
  }

  createZeroGrads(): TransformerWeights {
    const { d_model: D, factor_dim: F, d_ff: FFN, max_len: L } = this.spec;
    return {
      tokEmb: new Float32Array(this.vocabSize * F),
      projW: new Float32Array(F * D),
      posEmb: new Float32Array(L * D),
      wq: new Float32Array(D * D),
      bq: new Float32Array(D),
      wk: new Float32Array(D * D),
      bk: new Float32Array(D),
      wv: new Float32Array(D * D),
      bv: new Float32Array(D),
      wo: new Float32Array(D * D),
      bo: new Float32Array(D),
      ln1G: new Float32Array(D),
      ln1B: new Float32Array(D),
      w1: new Float32Array(D * FFN),
      b1: new Float32Array(FFN),
      w2: new Float32Array(FFN * D),
      b2: new Float32Array(D),
      ln2G: new Float32Array(D),
      ln2B: new Float32Array(D),
      wPool: new Float32Array(D),
      wCls: new Float32Array(D * 3),
      bCls: new Float32Array(3),
    };
  }

  accumulateBackward(
    cache: ForwardCache,
    targetClass: number,
    sampleWeight: number,
    grads: TransformerWeights
  ): number {
    const { d_model: D, factor_dim: F, num_heads: H, d_ff: FFN, activation } = this.spec;
    const dh = D / H;
    const w = this.weights;
    const { T, ids } = cache;
    const factorized = F !== D;

    const loss = -sampleWeight * Math.log(Math.max(1e-12, cache.probs[targetClass]));

    // dLoss / dLogits
    const dLogits = new Float32Array(3);
    for (let c = 0; c < 3; c++) {
      dLogits[c] = (cache.probs[c] - (c === targetClass ? 1 : 0)) * sampleWeight;
      grads.bCls[c] += dLogits[c];
    }

    const dPooled = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      let g = 0;
      for (let c = 0; c < 3; c++) {
        grads.wCls[d * 3 + c] += cache.pooled[d] * dLogits[c];
        g += w.wCls[d * 3 + c] * dLogits[c];
      }
      dPooled[d] = g;
    }

    // Backprop through pooling
    const dAlpha = new Float32Array(T);
    const dLN2 = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const a = cache.poolAlpha[t];
      const off = t * D;
      let dot = 0;
      for (let d = 0; d < D; d++) {
        dot += dPooled[d] * cache.LN2[off + d];
        dLN2[off + d] = a * dPooled[d] + (t === 0 ? 0.5 * dPooled[d] : 0);
      }
      dAlpha[t] = dot;
    }
    let sumAlphaDAlpha = 0;
    for (let t = 0; t < T; t++) sumAlphaDAlpha += cache.poolAlpha[t] * dAlpha[t];
    for (let t = 0; t < T; t++) {
      const dPoolLogit = cache.poolAlpha[t] * (dAlpha[t] - sumAlphaDAlpha);
      const off = t * D;
      for (let d = 0; d < D; d++) {
        grads.wPool[d] += cache.LN2[off + d] * dPoolLogit;
        dLN2[off + d] += w.wPool[d] * dPoolLogit;
      }
    }

    // Backprop LayerNorm 2 + Residual 2 + FFN
    const dLN1 = new Float32Array(T * D);
    const dR2 = new Float32Array(D);
    const dNorm2 = new Float32Array(D);
    const dFF1Act = new Float32Array(FFN);

    for (let t = 0; t < T; t++) {
      const off = t * D;
      const fOff = t * FFN;
      let sumDNorm = 0;
      let sumDNormXNorm = 0;
      for (let d = 0; d < D; d++) {
        const gOut = dLN2[off + d];
        const xNorm = cache.ln2Norm[off + d];
        grads.ln2G[d] += gOut * xNorm;
        grads.ln2B[d] += gOut;
        const dn = gOut * w.ln2G[d];
        dNorm2[d] = dn;
        sumDNorm += dn;
        sumDNormXNorm += dn * xNorm;
      }
      const invStd = cache.ln2InvStd[t];
      for (let d = 0; d < D; d++) {
        const dr =
          (invStd / D) * (D * dNorm2[d] - sumDNorm - cache.ln2Norm[off + d] * sumDNormXNorm);
        dR2[d] = dr;
        dLN1[off + d] += dr; // residual skip
        grads.b2[d] += dr;
      }
      for (let m = 0; m < FFN; m++) {
        let gAct = 0;
        const aVal = cache.ff1Act[fOff + m];
        for (let d = 0; d < D; d++) {
          grads.w2[m * D + d] += aVal * dR2[d];
          gAct += w.w2[m * D + d] * dR2[d];
        }
        const pre = cache.ff1Pre[fOff + m];
        const dPre =
          gAct * (activation === 'gelu' ? geluGrad(pre) : pre > 0 ? 1 : 0);
        dFF1Act[m] = dPre;
        grads.b1[m] += dPre;
      }
      for (let d = 0; d < D; d++) {
        const ln1Val = cache.LN1[off + d];
        let gLn1 = 0;
        for (let m = 0; m < FFN; m++) {
          const dPre = dFF1Act[m];
          grads.w1[d * FFN + m] += ln1Val * dPre;
          gLn1 += w.w1[d * FFN + m] * dPre;
        }
        dLN1[off + d] += gLn1;
      }
    }

    // Backprop LayerNorm 1 + Residual 1 + Wo
    const dX = new Float32Array(T * D);
    const dAttnConcat = new Float32Array(T * D);
    const dNorm1 = new Float32Array(D);

    for (let t = 0; t < T; t++) {
      const off = t * D;
      let sumDNorm = 0;
      let sumDNormXNorm = 0;
      for (let d = 0; d < D; d++) {
        const gOut = dLN1[off + d];
        const xNorm = cache.ln1Norm[off + d];
        grads.ln1G[d] += gOut * xNorm;
        grads.ln1B[d] += gOut;
        const dn = gOut * w.ln1G[d];
        dNorm1[d] = dn;
        sumDNorm += dn;
        sumDNormXNorm += dn * xNorm;
      }
      const invStd = cache.ln1InvStd[t];
      for (let d = 0; d < D; d++) {
        const dr =
          (invStd / D) * (D * dNorm1[d] - sumDNorm - cache.ln1Norm[off + d] * sumDNormXNorm);
        dX[off + d] += dr; // residual skip
        grads.bo[d] += dr;
        for (let k = 0; k < D; k++) {
          grads.wo[k * D + d] += cache.attnConcat[off + k] * dr;
          dAttnConcat[off + k] += w.wo[k * D + d] * dr;
        }
      }
    }

    // Backprop Multi-Head Self-Attention
    const dQ = new Float32Array(T * D);
    const dK = new Float32Array(T * D);
    const dV = new Float32Array(T * D);
    const scale = 1 / Math.sqrt(dh);
    const dA = new Float32Array(T);
    const dS = new Float32Array(T);

    for (let h = 0; h < H; h++) {
      const hOff = h * dh;
      const aBase = h * T * T;
      for (let i = 0; i < T; i++) {
        const rowBase = aBase + i * T;
        let sumAda = 0;
        for (let j = 0; j < T; j++) {
          let dot = 0;
          const aij = cache.attnWeights[rowBase + j];
          for (let d = 0; d < dh; d++) {
            const gOut = dAttnConcat[i * D + hOff + d];
            dot += gOut * cache.V[j * D + hOff + d];
            dV[j * D + hOff + d] += aij * gOut;
          }
          dA[j] = dot;
          sumAda += aij * dot;
        }
        for (let j = 0; j < T; j++) {
          dS[j] = cache.attnWeights[rowBase + j] * (dA[j] - sumAda) * scale;
        }
        for (let d = 0; d < dh; d++) {
          let gq = 0;
          const qid = cache.Q[i * D + hOff + d];
          for (let j = 0; j < T; j++) {
            const ds = dS[j];
            gq += ds * cache.K[j * D + hOff + d];
            dK[j * D + hOff + d] += ds * qid;
          }
          dQ[i * D + hOff + d] += gq;
        }
      }
    }

    // Backprop Q, K, V projections
    for (let t = 0; t < T; t++) {
      const off = t * D;
      for (let d = 0; d < D; d++) {
        const gq = dQ[off + d];
        const gk = dK[off + d];
        const gv = dV[off + d];
        grads.bq[d] += gq;
        grads.bk[d] += gk;
        grads.bv[d] += gv;
        for (let k = 0; k < D; k++) {
          const xk = cache.X[off + k];
          const wd = k * D + d;
          grads.wq[wd] += xk * gq;
          grads.wk[wd] += xk * gk;
          grads.wv[wd] += xk * gv;
          dX[off + k] += w.wq[wd] * gq + w.wk[wd] * gk + w.wv[wd] * gv;
        }
      }
    }

    // Backprop Pos + Token Embeddings
    for (let t = 0; t < T; t++) {
      const id = ids[t];
      const tokOff = id * F;
      const posOff = t * D;
      const xOff = t * D;
      for (let d = 0; d < D; d++) {
        grads.posEmb[posOff + d] += dX[xOff + d];
      }
      if (factorized) {
        for (let f = 0; f < F; f++) {
          const xf = cache.xTok[t * F + f];
          let gf = 0;
          for (let d = 0; d < D; d++) {
            const dx = dX[xOff + d];
            grads.projW[f * D + d] += xf * dx;
            gf += w.projW[f * D + d] * dx;
          }
          grads.tokEmb[tokOff + f] += gf;
        }
      } else {
        for (let d = 0; d < D; d++) {
          grads.tokEmb[tokOff + d] += dX[xOff + d];
        }
      }
    }

    return loss;
  }

  save(outDir: string, meta: Record<string, unknown>): void {
    fs.mkdirSync(outDir, { recursive: true });
    const config = {
      model_key: this.spec.model_key,
      name: this.spec.name,
      hf_id: this.spec.hf_id,
      family: this.spec.family,
      tokenizer_type: this.spec.tokenizer_type,
      cased: this.spec.cased,
      d_model: this.spec.d_model,
      factor_dim: this.spec.factor_dim,
      num_heads: this.spec.num_heads,
      d_ff: this.spec.d_ff,
      max_len: this.spec.max_len,
      activation: this.spec.activation,
      vocab_size: this.vocabSize,
      id2label: { '0': 'Favour', '1': 'Against', '2': 'None' },
      label2id: { Favour: 0, Against: 1, None: 2 },
    };
    fs.writeFileSync(path.join(outDir, 'config.json'), JSON.stringify(config, null, 2));
    fs.writeFileSync(
      path.join(outDir, 'vocab.json'),
      JSON.stringify(this.tokenizer.toJSON())
    );
    fs.writeFileSync(path.join(outDir, 'training_meta.json'), JSON.stringify(meta, null, 2));

    const keys = Object.keys(this.weights) as Array<keyof TransformerWeights>;
    const chunks: Buffer[] = [];
    const manifest: Record<string, number> = {};
    for (const k of keys) {
      const arr = this.weights[k];
      manifest[k] = arr.length;
      chunks.push(Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
    }
    fs.writeFileSync(path.join(outDir, 'weights_manifest.json'), JSON.stringify(manifest));
    fs.writeFileSync(path.join(outDir, 'weights.bin'), Buffer.concat(chunks));
  }

  static load(modelDir: string, spec: TransformerSpec): NeuralTransformerClassifier {
    const config = JSON.parse(
      fs.readFileSync(path.join(modelDir, 'config.json'), 'utf-8')
    ) as {
      tokenizer_type: 'wordpiece' | 'sentencepiece' | 'indicbert';
      cased: boolean;
      d_model: number;
      factor_dim: number;
      num_heads: number;
      d_ff: number;
      max_len: number;
      activation: 'gelu' | 'relu';
    };
    const vocabObj = JSON.parse(
      fs.readFileSync(path.join(modelDir, 'vocab.json'), 'utf-8')
    ) as Record<string, number>;
    const manifest = JSON.parse(
      fs.readFileSync(path.join(modelDir, 'weights_manifest.json'), 'utf-8')
    ) as Record<string, number>;
    const bin = fs.readFileSync(path.join(modelDir, 'weights.bin'));

    const tokenizer = new SubwordTokenizer(vocabObj, config.tokenizer_type, config.cased);
    const weights = {} as Record<string, Float32Array>;
    let byteOff = 0;
    for (const [k, len] of Object.entries(manifest)) {
      const bytes = len * 4;
      const arr = new Float32Array(len);
      Buffer.from(arr.buffer).set(bin.subarray(byteOff, byteOff + bytes));
      weights[k] = arr;
      byteOff += bytes;
    }

    const hydratedSpec: TransformerSpec = {
      ...spec,
      tokenizer_type: config.tokenizer_type,
      cased: config.cased,
      d_model: config.d_model,
      factor_dim: config.factor_dim,
      num_heads: config.num_heads,
      d_ff: config.d_ff,
      max_len: config.max_len,
      activation: config.activation,
    };

    return new NeuralTransformerClassifier(
      hydratedSpec,
      tokenizer,
      weights as unknown as TransformerWeights
    );
  }
}

export function trainTransformerOnSplits(
  spec: TransformerSpec,
  hfVocab: Set<string>,
  trainTexts: string[],
  trainLabels: StanceLabel[],
  valTexts: string[],
  valLabels: StanceLabel[],
  outputDir: string
): {
  model: NeuralTransformerClassifier;
  best_val_macro_f1: number;
  training_seconds: number;
  history: Array<{ epoch: number; train_loss: number; val_macro_f1: number }>;
} {
  const tStart = performance.now();
  const tokenizer = SubwordTokenizer.buildFromCorpus(trainTexts, hfVocab, spec, 10000);
  const model = new NeuralTransformerClassifier(spec, tokenizer);

  const trainIds = trainTexts.map((t) => tokenizer.encode(t, spec.max_len));
  const trainY = trainLabels.map((l) => LABELS.indexOf(l));
  const valIds = valTexts.map((t) => tokenizer.encode(t, spec.max_len));
  const valY = valLabels.map((l) => LABELS.indexOf(l));

  // Class-balanced weights
  const counts = [0, 0, 0];
  for (const y of trainY) if (y >= 0) counts[y] += 1;
  const totalN = Math.max(1, trainY.length);
  const classWeights = counts.map((c) => (c > 0 ? totalN / (3 * c) : 1));

  const keys = Object.keys(model.weights) as Array<keyof TransformerWeights>;
  const adamM = {} as Record<keyof TransformerWeights, Float32Array>;
  const adamV = {} as Record<keyof TransformerWeights, Float32Array>;
  const tokStep = new Int32Array(model.vocabSize);
  for (const k of keys) {
    adamM[k] = new Float32Array(model.weights[k].length);
    adamV[k] = new Float32Array(model.weights[k].length);
  }

  let bestValF1 = -1;
  let bestSnapshot: Record<string, Float32Array> | null = null;
  const history: Array<{ epoch: number; train_loss: number; val_macro_f1: number }> = [];

  const order = Array.from({ length: trainIds.length }, (_, i) => i);
  const rng = createPrng(spec.seed + 99);
  const batchSize = spec.batch_size;
  const totalSteps = Math.max(1, Math.ceil(trainIds.length / batchSize) * spec.epochs);
  let step = 0;

  for (let epoch = 1; epoch <= spec.epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let epochLoss = 0;
    let sampleCount = 0;

    for (let bStart = 0; bStart < order.length; bStart += batchSize) {
      const bEnd = Math.min(order.length, bStart + batchSize);
      const grads = model.createZeroGrads();
      const touchedTokIds = new Set<number>();
      let batchWeightSum = 0;

      for (let idx = bStart; idx < bEnd; idx++) {
        const sIdx = order[idx];
        const y = trainY[sIdx];
        if (y < 0) continue;
        const sw = classWeights[y];
        const seq = trainIds[sIdx];
        for (let t = 0; t < seq.length; t++) touchedTokIds.add(seq[t]);
        const cache = model.forward(seq);
        epochLoss += model.accumulateBackward(cache, y, sw, grads);
        batchWeightSum += sw;
        sampleCount += 1;
      }

      step += 1;
      const warmup = Math.max(1, Math.floor(0.08 * totalSteps));
      const lrMult =
        step < warmup
          ? step / warmup
          : 0.5 * (1 + Math.cos((Math.PI * (step - warmup)) / Math.max(1, totalSteps - warmup)));
      const lr = spec.lr * Math.max(0.08, lrMult);
      const invBatch = 1 / Math.max(1e-6, batchWeightSum);
      const beta1 = 0.9;
      const beta2 = 0.999;
      const bc1 = 1 - Math.pow(beta1, step);
      const bc2 = 1 - Math.pow(beta2, step);
      const weightDecay = 0.0015;

      // Sparse AdamW update ONLY for token embedding rows touched in this batch
      const F = spec.factor_dim;
      const tokP = model.weights.tokEmb;
      const tokG = grads.tokEmb;
      const tokM = adamM.tokEmb;
      const tokV = adamV.tokEmb;
      for (const tid of touchedTokIds) {
        tokStep[tid] += 1;
        const tBc1 = 1 - Math.pow(beta1, tokStep[tid]);
        const tBc2 = 1 - Math.pow(beta2, tokStep[tid]);
        const base = tid * F;
        for (let f = 0; f < F; f++) {
          const i = base + f;
          let g = tokG[i] * invBatch;
          if (g > 2.0) g = 2.0;
          else if (g < -2.0) g = -2.0;
          const m = (tokM[i] = beta1 * tokM[i] + (1 - beta1) * g);
          const v = (tokV[i] = beta2 * tokV[i] + (1 - beta2) * g * g);
          tokP[i] -= lr * ((m / tBc1) / (Math.sqrt(v / tBc2) + 1e-8) + weightDecay * tokP[i]);
        }
      }

      for (const k of keys) {
        if (k === 'tokEmb') continue;
        const param = model.weights[k];
        const gArr = grads[k];
        const mArr = adamM[k];
        const vArr = adamV[k];
        const applyWd =
          k !== 'bq' &&
          k !== 'bk' &&
          k !== 'bv' &&
          k !== 'bo' &&
          k !== 'b1' &&
          k !== 'b2' &&
          k !== 'bCls' &&
          k !== 'ln1G' &&
          k !== 'ln1B' &&
          k !== 'ln2G' &&
          k !== 'ln2B';

        for (let i = 0; i < param.length; i++) {
          let g = gArr[i] * invBatch;
          if (g > 2.0) g = 2.0;
          else if (g < -2.0) g = -2.0;
          if (g === 0 && !applyWd) continue;

          const m = (mArr[i] = beta1 * mArr[i] + (1 - beta1) * g);
          const v = (vArr[i] = beta2 * vArr[i] + (1 - beta2) * g * g);
          const mHat = m / bc1;
          const vHat = v / bc2;
          const wdTerm = applyWd ? weightDecay * param[i] : 0;
          param[i] -= lr * (mHat / (Math.sqrt(vHat) + 1e-8) + wdTerm);
        }
      }
    }

    // Evaluate on Validation split
    const cm = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < valIds.length; i++) {
      const cache = model.forward(valIds[i]);
      let pred = 0;
      if (cache.probs[1] > cache.probs[pred]) pred = 1;
      if (cache.probs[2] > cache.probs[pred]) pred = 2;
      const yt = valY[i];
      if (yt >= 0) cm[yt][pred] += 1;
    }
    let valMacroF1 = 0;
    for (let c = 0; c < 3; c++) {
      const tp = cm[c][c];
      const fp = cm[0][c] + cm[1][c] + cm[2][c] - tp;
      const fn = cm[c][0] + cm[c][1] + cm[c][2] - tp;
      const p = tp + fp > 0 ? tp / (tp + fp) : 0;
      const r = tp + fn > 0 ? tp / (tp + fn) : 0;
      valMacroF1 += p + r > 0 ? (2 * p * r) / (p + r) : 0;
    }
    valMacroF1 /= 3;

    const avgLoss = epochLoss / Math.max(1, sampleCount);
    history.push({
      epoch,
      train_loss: Math.round(avgLoss * 10000) / 10000,
      val_macro_f1: Math.round(valMacroF1 * 10000) / 10000,
    });
    console.log(
      `[${spec.model_key}] Epoch ${epoch}/${spec.epochs} | loss=${avgLoss.toFixed(4)} | val_macro_f1=${valMacroF1.toFixed(4)}`
    );

    if (valMacroF1 > bestValF1) {
      bestValF1 = valMacroF1;
      bestSnapshot = {};
      for (const k of keys) bestSnapshot[k] = new Float32Array(model.weights[k]);
    }
  }

  if (bestSnapshot) {
    for (const k of keys) model.weights[k].set(bestSnapshot[k]);
  }

  const trainingSeconds = Math.round(((performance.now() - tStart) / 1000) * 100) / 100;
  model.save(outputDir, {
    model_key: spec.model_key,
    hf_id: spec.hf_id,
    vocab_size: model.vocabSize,
    train_samples: trainTexts.length,
    val_samples: valTexts.length,
    best_val_macro_f1: Math.round(bestValF1 * 10000) / 10000,
    training_seconds: trainingSeconds,
    history,
  });

  return {
    model,
    best_val_macro_f1: Math.round(bestValF1 * 10000) / 10000,
    training_seconds: trainingSeconds,
    history,
  };
}
