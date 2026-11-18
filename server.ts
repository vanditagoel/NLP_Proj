import 'dotenv/config';
import express from 'express';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { createServer as createViteServer } from 'vite';

const PORT = Number(process.env.PORT) || 3000;
const FASTAPI_HOST = '127.0.0.1';
const FASTAPI_PORT = Number(process.env.FASTAPI_PORT) || 8001;

function resolvePythonBin(): string {
  const venvPy = path.resolve(process.cwd(), '.venv', 'bin', 'python');
  if (fs.existsSync(venvPy)) {
    return venvPy;
  }
  return process.env.PYTHON_BIN || 'python3';
}

function startFastApiProcess(): ChildProcess {
  const pyBin = resolvePythonBin();
  const child = spawn(
    pyBin,
    ['-m', 'uvicorn', 'backend.app.main:app', '--host', FASTAPI_HOST, '--port', String(FASTAPI_PORT)],
    {
      cwd: process.cwd(),
      stdio: 'inherit',
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    }
  );
  child.on('exit', (code) => {
    if (code !== null && code !== 0) {
      console.error(`FastAPI backend exited with code ${code}`);
    }
  });
  return child;
}

async function waitForFastApi(timeoutMs = 15000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const ok = await new Promise<boolean>((resolve) => {
        const req = http.get(`http://${FASTAPI_HOST}:${FASTAPI_PORT}/api/health`, (res) => {
          res.resume();
          resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.setTimeout(1000, () => {
          req.destroy();
          resolve(false);
        });
      });
      if (ok) return;
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  console.warn('FastAPI health check timed out; proxy will still forward requests.');
}

function proxyToFastApi(req: express.Request, res: express.Response): void {
  const options: http.RequestOptions = {
    hostname: FASTAPI_HOST,
    port: FASTAPI_PORT,
    path: req.originalUrl,
    method: req.method,
    headers: {
      ...req.headers,
      host: `${FASTAPI_HOST}:${FASTAPI_PORT}`,
    },
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.status(proxyRes.statusCode || 502);
    for (const [key, val] of Object.entries(proxyRes.headers)) {
      if (val !== undefined) {
        res.setHeader(key, val);
      }
    }
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    if (!res.headersSent) {
      res.status(502).json({
        detail: `FastAPI backend unavailable: ${err.message}`,
      });
    }
  });

  req.pipe(proxyReq);
}

async function startServer() {
  const fastApiProc = startFastApiProcess();
  const cleanup = () => {
    if (!fastApiProc.killed) {
      fastApiProc.kill('SIGTERM');
    }
  };
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);
  process.on('exit', cleanup);

  await waitForFastApi();

  const app = express();
  app.use('/api', proxyToFastApi);
  app.use('/charts', proxyToFastApi);

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(
      `Preview server on http://0.0.0.0:${PORT} -> proxying /api & /charts to FastAPI (${FASTAPI_HOST}:${FASTAPI_PORT})`
    );
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
