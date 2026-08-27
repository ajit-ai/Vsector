FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY vsector ./vsector
COPY proto ./proto
RUN pip install --no-cache-dir -e .
EXPOSE 8080 50051
ENV VSECTOR_ENV=production
CMD ["vsector", "serve", "--host", "0.0.0.0", "--port", "8080"]
