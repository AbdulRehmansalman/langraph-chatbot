# Deployment Guide

## Overview

This guide covers deploying the Enterprise RAG System to production environments.

---

## Prerequisites

### Required
- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- Supabase account (for storage)
- Environment variables configured

### Optional
- Redis (for caching)
- AWS Bedrock access (for production LLM)
- LangSmith account (for observability)

---

## Environment Setup

### 1. Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd mcp-chatbot/ai-chatbot-be

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

**Critical Variables**:
```bash
ENVIRONMENT=production
SECRET_KEY=<generate-with-openssl-rand-hex-32>
DATABASE_URL=postgresql://user:pass@host:5432/db
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=<your-service-key>
SUPABASE_STORAGE_BUCKET=documents
```

### 3. Database Setup

```bash
# Run migrations
python -m alembic upgrade head

# Verify database connection
python -c "from app.database.connection import SessionLocal; SessionLocal().execute('SELECT 1')"
```

---

## Deployment Options

### Option 1: Docker (Recommended)

#### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health/live')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and Run

```bash
# Build image
docker build -t rag-api:latest .

# Run container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  --env-file .env \
  rag-api:latest
```

### Option 2: Docker Compose

#### Create docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s

  postgres:
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_DB: chatbot_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Deploy

```bash
docker-compose up -d
```

### Option 3: Kubernetes

#### Create Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        envFrom:
        - secretRef:
            name: rag-api-secrets
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 5
          failureThreshold: 30
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Deploy to Kubernetes

```bash
# Create secrets
kubectl create secret generic rag-api-secrets \
  --from-env-file=.env

# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods
kubectl get services
```

---

## Production Checklist

### Security
- [ ] All secrets in environment variables
- [ ] SECRET_KEY is 32+ characters
- [ ] Database uses SSL/TLS
- [ ] CORS origins restricted to production domains
- [ ] Debug mode disabled
- [ ] Rate limiting enabled

### Performance
- [ ] Database connection pooling configured
- [ ] Redis caching enabled (optional)
- [ ] LLM provider configured (AWS Bedrock recommended)
- [ ] Vector store optimized

### Monitoring
- [ ] Health check endpoints accessible
- [ ] LangSmith tracing enabled (optional)
- [ ] Structured JSON logging enabled
- [ ] Log aggregation configured

### Reliability
- [ ] Multiple replicas running (3+ recommended)
- [ ] Auto-scaling configured
- [ ] Backup strategy in place
- [ ] Disaster recovery plan documented

---

## Verification

### 1. Health Checks

```bash
# Liveness
curl http://your-domain/health/live

# Readiness
curl http://your-domain/health/ready

# Detailed health
curl http://your-domain/health
```

### 2. API Endpoints

```bash
# Root
curl http://your-domain/

# Chat (requires auth)
curl -X POST http://your-domain/api/chat/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### 3. Streaming

```bash
# Streaming chat
curl -X POST http://your-domain/api/chat/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "stream_timeout": 30}'
```

---

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale api=3

# Kubernetes
kubectl scale deployment rag-api --replicas=5
```

### Vertical Scaling

Update resource limits in deployment configuration.

---

## Rollback

### Docker

```bash
# Stop current version
docker stop rag-api

# Start previous version
docker run -d --name rag-api rag-api:previous
```

### Kubernetes

```bash
# Rollback to previous version
kubectl rollout undo deployment/rag-api

# Rollback to specific revision
kubectl rollout undo deployment/rag-api --to-revision=2
```

---

## Troubleshooting

### Application Won't Start

1. Check logs: `docker logs rag-api` or `kubectl logs <pod-name>`
2. Verify environment variables
3. Check database connectivity
4. Verify all required secrets are set

### Health Checks Failing

1. Check `/health/ready` endpoint
2. Verify database connection
3. Check LLM provider configuration
4. Review application logs

### Performance Issues

1. Check resource usage (CPU, memory)
2. Review database query performance
3. Enable caching (Redis)
4. Scale horizontally

---

## Support

For issues or questions:
1. Check application logs
2. Review health check endpoints
3. Consult operational runbook
4. Contact DevOps team
