# Operational Runbook

## Overview

This runbook provides operational procedures for the Enterprise RAG System.

---

## Daily Operations

### Morning Checks

```bash
# 1. Check application health
curl https://api.yourdomain.com/health

# 2. Check all services are running
kubectl get pods  # Kubernetes
docker ps         # Docker

# 3. Review overnight logs for errors
kubectl logs -l app=rag-api --since=24h | grep ERROR
```

### Monitoring Dashboards

- **Health**: `https://api.yourdomain.com/health`
- **API Docs**: `https://api.yourdomain.com/docs`
- **LangSmith** (if enabled): `https://smith.langchain.com`

---

## Common Issues

### Issue 1: High Response Latency

**Symptoms**:
- API responses taking >5 seconds
- Timeout errors in logs

**Diagnosis**:
```bash
# Check health endpoint
curl https://api.yourdomain.com/health

# Check database performance
# Review slow query logs

# Check LLM provider status
# Verify AWS Bedrock/OpenAI status page
```

**Resolution**:
1. Scale horizontally: `kubectl scale deployment rag-api --replicas=5`
2. Enable Redis caching
3. Optimize retrieval parameters (reduce `top_k`)
4. Check database indexes

**Prevention**:
- Set up auto-scaling
- Monitor P95 latency
- Configure alerts for >3s response time

---

### Issue 2: Database Connection Errors

**Symptoms**:
- `/health/ready` returns 503
- "Database connection failed" in logs

**Diagnosis**:
```bash
# Check database connectivity
psql $DATABASE_URL -c "SELECT 1"

# Check connection pool
# Review database logs
```

**Resolution**:
1. Restart application: `kubectl rollout restart deployment/rag-api`
2. Check database credentials
3. Verify database is running
4. Check network connectivity

**Prevention**:
- Monitor database connections
- Configure connection pooling
- Set up database failover

---

### Issue 3: Out of Memory (OOM)

**Symptoms**:
- Pods restarting frequently
- OOMKilled in pod status

**Diagnosis**:
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check memory usage
kubectl top pods
```

**Resolution**:
1. Increase memory limits in deployment
2. Review memory leaks in application
3. Optimize vector store queries
4. Reduce batch sizes

**Prevention**:
- Monitor memory usage trends
- Set appropriate resource limits
- Profile application memory

---

### Issue 4: Streaming Timeouts

**Symptoms**:
- Streaming requests timing out
- "TIMEOUT" events in stream

**Diagnosis**:
```bash
# Check streaming endpoint
curl -X POST https://api.yourdomain.com/api/chat/stream \
  -H "Authorization: Bearer <token>" \
  -d '{"message": "test", "stream_timeout": 60}'

# Review streaming logs
kubectl logs -l app=rag-api | grep "Stream timeout"
```

**Resolution**:
1. Increase `stream_timeout` parameter
2. Optimize RAG pipeline (disable HyDE, compression)
3. Use faster LLM model
4. Check network latency

**Prevention**:
- Monitor streaming latency
- Set realistic timeout values
- Use appropriate RAG configuration

---

### Issue 5: Authentication Failures

**Symptoms**:
- 401 Unauthorized errors
- "Invalid token" in logs

**Diagnosis**:
```bash
# Verify token
curl https://api.yourdomain.com/api/auth/verify \
  -H "Authorization: Bearer <token>"

# Check SECRET_KEY configuration
```

**Resolution**:
1. Verify SECRET_KEY is consistent across all instances
2. Check token expiration
3. Regenerate tokens if needed
4. Verify user exists in database

**Prevention**:
- Monitor authentication error rates
- Set appropriate token expiration
- Implement token refresh

---

## Maintenance Procedures

### Planned Downtime

**Before Maintenance**:
1. Notify users of maintenance window
2. Create database backup
3. Tag current deployment version
4. Prepare rollback plan

**During Maintenance**:
```bash
# 1. Scale down to 1 replica
kubectl scale deployment rag-api --replicas=1

# 2. Apply changes
kubectl apply -f deployment.yaml

# 3. Wait for rollout
kubectl rollout status deployment/rag-api

# 4. Scale back up
kubectl scale deployment rag-api --replicas=3
```

**After Maintenance**:
1. Verify health checks pass
2. Test critical endpoints
3. Monitor error rates
4. Notify users maintenance is complete

### Database Migrations

```bash
# 1. Backup database
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql

# 2. Run migrations
kubectl exec -it <pod-name> -- python -m alembic upgrade head

# 3. Verify migration
kubectl exec -it <pod-name> -- python -m alembic current

# 4. Test application
curl https://api.yourdomain.com/health/ready
```

### Rotating Secrets

```bash
# 1. Generate new secret
openssl rand -hex 32

# 2. Update Kubernetes secret
kubectl create secret generic rag-api-secrets-new \
  --from-literal=SECRET_KEY=<new-secret> \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Update deployment to use new secret
kubectl set env deployment/rag-api --from=secret/rag-api-secrets-new

# 4. Verify rollout
kubectl rollout status deployment/rag-api

# 5. Delete old secret
kubectl delete secret rag-api-secrets
```

---

## Monitoring & Alerts

### Key Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| Response time (P95) | >3s | Scale up or optimize |
| Error rate | >1% | Investigate logs |
| Memory usage | >80% | Increase limits |
| CPU usage | >80% | Scale horizontally |
| Database connections | >80% of pool | Increase pool size |

### Alert Configuration

**Critical Alerts** (PagerDuty):
- Application down (all pods unhealthy)
- Database unreachable
- Error rate >5%

**Warning Alerts** (Slack):
- Response time >3s for 5 minutes
- Memory usage >80%
- Disk space <20%

---

## Backup & Recovery

### Database Backups

**Automated** (Daily):
```bash
# Cron job
0 2 * * * pg_dump $DATABASE_URL | gzip > /backups/db_$(date +\%Y\%m\%d).sql.gz
```

**Manual**:
```bash
pg_dump $DATABASE_URL > backup.sql
```

### Restore from Backup

```bash
# 1. Stop application
kubectl scale deployment rag-api --replicas=0

# 2. Restore database
psql $DATABASE_URL < backup.sql

# 3. Start application
kubectl scale deployment rag-api --replicas=3

# 4. Verify
curl https://api.yourdomain.com/health/ready
```

---

## Performance Tuning

### Database Optimization

```sql
-- Add indexes
CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_documents_user_id ON documents(user_id);

-- Analyze tables
ANALYZE chat_history;
ANALYZE documents;
```

### Application Tuning

**High Traffic** (>1000 req/min):
```python
# Increase workers
uvicorn main:app --workers 4

# Enable caching
RAG_USE_CACHE=true

# Reduce retrieval
RETRIEVAL_TOP_K=15
```

**Low Latency** (<500ms):
```python
# Disable expensive features
RAG_USE_HYDE=false
RAG_USE_COMPRESSION=false
RERANKING_METHOD=none
```

---

## Security Incidents

### Suspected Breach

1. **Immediate Actions**:
   - Rotate all secrets
   - Review access logs
   - Disable compromised accounts
   - Notify security team

2. **Investigation**:
   - Review audit logs
   - Check for unusual patterns
   - Identify affected data

3. **Recovery**:
   - Patch vulnerabilities
   - Update security policies
   - Conduct post-mortem

---

## Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | Slack: #oncall | PagerDuty |
| DevOps Lead | devops@company.com | Phone |
| Security Team | security@company.com | Immediate |
| Database Admin | dba@company.com | Email |

---

## Useful Commands

### Kubernetes

```bash
# View logs
kubectl logs -f <pod-name>

# Execute command in pod
kubectl exec -it <pod-name> -- bash

# Port forward
kubectl port-forward <pod-name> 8000:8000

# Describe pod
kubectl describe pod <pod-name>

# Get events
kubectl get events --sort-by='.lastTimestamp'
```

### Docker

```bash
# View logs
docker logs -f rag-api

# Execute command
docker exec -it rag-api bash

# Restart container
docker restart rag-api

# View resource usage
docker stats rag-api
```

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-22 | Initial runbook created | DevOps Team |
