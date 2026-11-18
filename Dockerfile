FROM node:22-bookworm-slim

# Install Python 3.11, SSL certs, and build tools for any sdist/wheel fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip/setuptools/wheel first, then install CPU PyTorch and Python dependencies
COPY requirements.txt ./
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && (/app/.venv/bin/python -m pip install --no-cache-dir --prefer-binary --index-url https://download.pytorch.org/whl/cpu torch \
        || /app/.venv/bin/python -m pip install --no-cache-dir --prefer-binary torch) \
    && /app/.venv/bin/python -m pip install --no-cache-dir --prefer-binary -r requirements.txt

# Install Node dependencies & build React frontend
COPY package.json bun.lock* ./
RUN npm install

COPY . .
RUN npm run build

ENV NODE_ENV=production
ENV PORT=3000
ENV FASTAPI_PORT=8001
EXPOSE 3000

CMD ["npm", "start"]
