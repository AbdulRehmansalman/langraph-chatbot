# Docker Setup Guide

This guide explains how to build and run the AI Chatbot Backend using Docker.

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)

---

## Quick Start

```bash
# 1. Build the API image
docker build -t ai-chatbot-api:latest .

# 2. Start Redis (required for caching)
docker-compose up -d redis

# 3. Run the API container
docker run -d \
  --name ai-chatbot-api \
  -p 8000:8000 \
  --env-file .env \
  --network ai-chatbot-be_default \
  ai-chatbot-api:latest

# 4. Verify
curl http://localhost:8000/
```

---

## Building the Docker Image

The project uses a multi-stage Dockerfile optimized for smaller image size (~1.5-2GB).

### Build Commands

```bash
# Standard build
docker build -t ai-chatbot-api:latest .

# Build with no cache (clean build)
docker build --no-cache -t ai-chatbot-api:latest .

# Build with build arguments
docker build --build-arg SOME_ARG=value -t ai-chatbot-api:latest .
```

### What the Dockerfile Does

1. **Stage 1 (Builder):** Installs build dependencies and Python packages
2. **Stage 2 (Production):** Creates a slim image with only runtime dependencies
3. Pre-downloads the embedding model (`all-mpnet-base-v2`)
4. Runs as non-root user for security

---

## Running with Docker Compose

### Start Services

```bash
# Start Redis only
docker-compose up -d redis

# Start Redis with debug UI (development)
docker-compose --profile debug up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Services Included

| Service | Port | Description |
|---------|------|-------------|
| redis | 6379 | Session caching |
| redis-commander | 8081 | Redis debug UI (dev only) |

---

## Running the API Container

### Option 1: With Environment File

```bash
docker run -d \
  --name ai-chatbot-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/uploads:/app/uploads \
  ai-chatbot-api:latest
```

### Option 2: With Inline Environment Variables

```bash
docker run -d \
  --name ai-chatbot-api \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
  -e LLM_PROVIDER="ollama" \
  ai-chatbot-api:latest
```

### Option 3: Connect to Docker Network (for Redis)

```bash
# First, find the network name
docker network ls

# Run with network connection
docker run -d \
  --name ai-chatbot-api \
  -p 8000:8000 \
  --env-file .env \
  --network ai-chatbot-be_default \
  -e REDIS_URL="redis://redis:6379" \
  ai-chatbot-api:latest
```

---

## Environment Variables

Create a `.env` file:

```env
# Database (Supabase PostgreSQL)
DATABASE_URL=postgresql://user:password@host:5432/dbname

# LLM Provider
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b

# JWT Authentication
JWT_SECRET_KEY=your-secret-key

# Redis (optional)
REDIS_URL=redis://redis:6379

# Supabase (for storage)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
```

**Important:** Use `host.docker.internal` to access services on the host machine from inside Docker.

---

## Full Docker Compose Setup

Add the API service to `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-chatbot-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./uploads:/app/uploads
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: chatbot-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: chatbot-redis-commander
    environment:
      - REDIS_HOSTS=local:redis:6379
    ports:
      - "8081:8081"
    depends_on:
      - redis
    profiles:
      - debug
    restart: unless-stopped

volumes:
  redis_data:
    driver: local
```

Then run:

```bash
# Build and start everything
docker-compose up -d --build

# View logs
docker-compose logs -f api
```

---

## Common Commands

| Command | Description |
|---------|-------------|
| `docker build -t ai-chatbot-api .` | Build the image |
| `docker-compose up -d` | Start all services |
| `docker-compose down` | Stop all services |
| `docker-compose logs -f api` | Follow API logs |
| `docker-compose restart api` | Restart API service |
| `docker exec -it ai-chatbot-api bash` | Shell into container |
| `docker system prune -a` | Clean up unused images |

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs ai-chatbot-api

# Check if port is in use
lsof -i :8000
```

### Database connection issues

- Use `host.docker.internal` instead of `localhost` for host database
- Ensure database allows connections from Docker network

### Ollama connection issues

```bash
# If Ollama runs on host machine:
OLLAMA_BASE_URL=http://host.docker.internal:11434

# If Ollama runs in same Docker network:
OLLAMA_BASE_URL=http://ollama:11434
```

### Redis connection issues

```bash
# Test Redis connection
docker exec -it chatbot-redis redis-cli ping
# Should return: PONG
```

### Image too large

The multi-stage build should produce ~1.5-2GB image. If larger:

```bash
# Check image size
docker images ai-chatbot-api

# Rebuild with no cache
docker build --no-cache -t ai-chatbot-api .
```

---

## Production Checklist

- [ ] Use secrets management (not `.env` files)
- [ ] Set resource limits in docker-compose
- [ ] Enable logging to external service
- [ ] Set up health check monitoring
- [ ] Configure reverse proxy (nginx/traefik)
- [ ] Enable TLS/SSL
