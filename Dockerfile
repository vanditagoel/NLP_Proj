FROM node:22-bookworm-slim

# Install Python 3.11 & build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python ML dependencies into .venv (CPU PyTorch + requirements)
COPY requirements.txt ./
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

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
