import 'dotenv/config';
import express from 'express';
import type { Request, Response } from 'express';
import path from 'node:path';
import Papa from 'papaparse';
import { createServer as createViteServer } from 'vite';
import {
  CHARTS_DIR,
  LABELS,
  ModelNotTrainedError,
  getDatasetSamples,
  getDatasetStats,
  getEvaluationSummary,
  getProjectOverview,
  initializeMlService,
  listModels,
  predictBatchTexts,
  predictCompare,
  predictSingle,
  startBackgroundTraining,
  trainStatus,
} from './server_ml.ts';

const PORT = Number(process.env.PORT) || 3000;

async function startServer() {
  initializeMlService();

  const app = express();
  app.use(express.json({ limit: '5mb' }));

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({ status: 'ok' });
  });

  app.get('/api/project', (_req: Request, res: Response) => {
    res.json(getProjectOverview());
  });

  app.get('/api/dataset/stats', (_req: Request, res: Response) => {
    const stats = getDatasetStats();
    if (!stats) {
      res.status(404).json({
        detail: 'Dataset stats not generated yet. Run dataset preparation and analysis first.',
      });
      return;
    }
    res.json(stats);
  });

  app.get('/api/dataset/samples', (req: Request, res: Response) => {
    const language = typeof req.query.language === 'string' ? req.query.language : 'All';
    const label = typeof req.query.label === 'string' ? req.query.label : 'All';
    const split = typeof req.query.split === 'string' ? req.query.split : 'All';
    const limit = typeof req.query.limit === 'string' ? Number(req.query.limit) : 25;

    const samples = getDatasetSamples({ language, label, split, limit });
    res.json({ count: samples.length, samples });
  });

  app.get('/api/models', (_req: Request, res: Response) => {
    res.json({ models: listModels() });
  });

  app.get('/api/evaluation', (_req: Request, res: Response) => {
    res.json(getEvaluationSummary());
  });

  app.post('/api/train', (req: Request, res: Response) => {
    const body = (req.body || {}) as { model_keys?: string[] };
    const modelKeys =
      Array.isArray(body.model_keys) && body.model_keys.length > 0
        ? body.model_keys
        : ['muril', 'mbert', 'xlmr', 'indicbert'];
    res.json(startBackgroundTraining(modelKeys));
  });

  app.get('/api/train/status', (_req: Request, res: Response) => {
    res.json({ ...trainStatus });
  });

  app.post('/api/predict', (req: Request, res: Response) => {
    const body = (req.body || {}) as {
      text?: string;
      language?: string;
      model_key?: string;
    };
    const text = typeof body.text === 'string' ? body.text : '';
    const language = typeof body.language === 'string' ? body.language : 'Auto';
    const modelKey = typeof body.model_key === 'string' ? body.model_key : 'muril';

    if (!text.trim()) {
      res.status(400).json({ detail: 'Input text cannot be empty.' });
      return;
    }

    try {
      res.json(predictSingle(text, language, modelKey));
    } catch (err) {
      if (err instanceof ModelNotTrainedError) {
        res.status(400).json({ detail: 'Model not trained' });
        return;
      }
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Prediction failed' });
    }
  });

  app.post('/api/predict/compare', (req: Request, res: Response) => {
    const body = (req.body || {}) as {
      text?: string;
      language?: string;
      model_keys?: string[] | null;
    };
    const text = typeof body.text === 'string' ? body.text : '';
    const language = typeof body.language === 'string' ? body.language : 'Auto';
    const modelKeys = Array.isArray(body.model_keys) ? body.model_keys : null;

    if (!text.trim()) {
      res.status(400).json({ detail: 'Input text cannot be empty.' });
      return;
    }

    try {
      res.json(predictCompare(text, language, modelKeys));
    } catch (err) {
      if (err instanceof ModelNotTrainedError) {
        res.status(400).json({ detail: 'Model not trained' });
        return;
      }
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Comparison failed' });
    }
  });

  app.post(
    '/api/predict/batch',
    express.raw({ type: 'multipart/form-data', limit: '15mb' }),
    async (req: Request, res: Response) => {
      try {
        const contentType = req.headers['content-type'] || '';
        if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
          res.status(400).json({ detail: 'Uploaded CSV file is empty.' });
          return;
        }

        const formData = await new globalThis.Response(new Uint8Array(req.body), {
          headers: { 'content-type': contentType },
        }).formData();

        const fileEntry = formData.get('file');
        if (!fileEntry || typeof fileEntry === 'string') {
          res.status(400).json({ detail: 'Uploaded CSV file is missing.' });
          return;
        }

        const rawCsv = await fileEntry.text();
        if (!rawCsv.trim()) {
          res.status(400).json({ detail: 'Uploaded CSV file is empty.' });
          return;
        }

        const textColumn = String(formData.get('text_column') || 'text');
        const modelKeysRaw = String(formData.get('model_keys') || 'muril');
        const returnCsv = String(formData.get('return_csv') || 'false').toLowerCase() === 'true';

        const parsed = Papa.parse<Record<string, string>>(rawCsv, {
          header: true,
          skipEmptyLines: true,
        });

        const fields = parsed.meta.fields || [];
        if (!fields.includes(textColumn)) {
          res.status(400).json({
            detail: `Column '${textColumn}' not found. Available columns: ${JSON.stringify(fields)}`,
          });
          return;
        }

        const selectedKeys = modelKeysRaw
          .split(',')
          .map((k) => k.trim())
          .filter(Boolean);
        if (selectedKeys.length === 0) {
          res.status(400).json({ detail: 'Select at least one model.' });
          return;
        }

        const rows = parsed.data.slice(0, 500);
        const texts = rows.map((r) => String(r[textColumn] ?? ''));
        const primaryKey = selectedKeys[0];

        const primaryPreds = predictBatchTexts(texts, primaryKey);
        const outRows = rows.map((row, idx) => ({
          ...row,
          detected_language: primaryPreds[idx].detected_language,
          prediction: primaryPreds[idx].stance,
          confidence: primaryPreds[idx].confidence,
        })) as Array<Record<string, string | number>>;

        for (const extraKey of selectedKeys.slice(1)) {
          const extraPreds = predictBatchTexts(texts, extraKey);
          for (let i = 0; i < outRows.length; i++) {
            outRows[i][`pred_${extraKey}`] = extraPreds[i].stance;
          }
        }

        if (returnCsv) {
          const csvOut = '\uFEFF' + Papa.unparse(outRows);
          res.setHeader('Content-Type', 'text/csv; charset=utf-8');
          res.setHeader(
            'Content-Disposition',
            'attachment; filename="climate_stance_predictions.csv"'
          );
          res.send(csvOut);
          return;
        }

        const counts: Record<string, number> = { Favour: 0, Against: 0, None: 0 };
        for (const r of outRows) {
          const pred = String(r.prediction);
          if (LABELS.includes(pred as 'Favour' | 'Against' | 'None')) {
            counts[pred] += 1;
          }
        }

        const outColumns = outRows.length > 0 ? Object.keys(outRows[0]) : fields;
        res.json({
          model_key: primaryKey,
          selected_models: selectedKeys,
          total_rows: outRows.length,
          counts,
          columns: outColumns,
          rows: outRows,
        });
      } catch (err) {
        if (err instanceof ModelNotTrainedError) {
          res.status(400).json({ detail: 'Model not trained' });
          return;
        }
        res.status(400).json({
          detail: err instanceof Error ? err.message : 'Batch prediction failed',
        });
      }
    }
  );

  app.use('/charts', express.static(CHARTS_DIR));

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`IndicClimate AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
