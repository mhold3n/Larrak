# Architecture Refactor Progress

## Phase 1: Split API from Worker ⚠️ IN PROGRESS

### Completed ✅

- [x] Created `Dockerfile.api` for larrak-api service
- [x] Added Redis service to docker-compose.yml
- [x] Added larrak-api service with port 8000 exposed
- [x] Created larrak-internal network
- [x] Connected all services to internal network
- [x] Added redis_data volume
- [x] Validated docker-compose.yml syntax

### Blocked ⚠️

**Docker credential issue**: Build failed with `docker-credential-osxkeychain` not found

**Fix Options:**

1. **Quick fix** - Remove credential helper:

```bash
# Edit ~/.docker/config.json
# Remove or comment out the "credsStore" line
```

1. **Proper fix** - Install credentials helper:

```bash
brew install docker-credential-helper
```

1. **Workaround** - Use Docker Desktop GUI to build

### Next Steps

1. [ ] Fix Docker credentials (see above)
2. [ ] Build larrak-api: `docker compose build larrak-api`
3. [ ] Start services: `docker compose up -d redis larrak-api`
4. [ ] Test Redis: `docker exec larrak-redis redis-cli ping`
5. [ ] Test API: `curl http://localhost:8000/api/health`

## Phase 2: Implement Job Queue (PENDING)

- Update dashboard/api.py with RQ job submission
- Create worker/run_worker.py
- Test end-to-end job flow

## Files Created

- `Dockerfile.api` - API service container definition
- `REFACTOR_PROGRESS.md` - This file
- Updated: `docker-compose.yml` - Added redis, larrak-api, network config
