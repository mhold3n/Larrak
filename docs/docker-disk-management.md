# Docker Disk Management

## Current Status

As of 2026-01-05, the Docker.raw file was **466GB** - significantly oversized for the actual Docker resources in use (~30GB of images, containers, and volumes).

## Root Causes Identified

1. **Unbounded log growth**: All containers were using `json-file` logging driver with no size limits
2. **Build cache accumulation**: 5.49GB of build cache (now cleaned)
3. **Unused images**: 10.18GB of reclaimable image space

## Fixes Implemented

### 1. Log Rotation Configuration

Added logging limits to all services in `docker-compose.yml`:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"    # Maximum 10MB per log file
    max-file: "3"      # Keep 3 log files (30MB max per container)
```

This prevents logs from growing unbounded and is applied to:

- `larrak-dev`
- `outline-api`
- `cem-service`
- `weaviate`
- `openfoam`
- `calculix`

### 2. Build Cache Cleanup

Cleaned 13.5GB of build cache using:

```bash
docker builder prune -af
```

## Ongoing Maintenance

### Regular Cleanup Commands

**View current disk usage:**

```bash
docker system df
docker system df -v  # Detailed view
```

**Clean build cache (safe, run monthly):**

```bash
docker builder prune -af
```

**Remove unused images (careful - only removes unused):**

```bash
docker image prune -a
```

**Full cleanup (aggressive - removes all unused resources):**

```bash
docker system prune -a --volumes
```

### Reclaim Docker.raw Space on macOS

The `Docker.raw` file doesn't automatically shrink even after cleanup. To reclaim space:

1. **Restart Docker Desktop** - may trigger automatic compaction
2. **Manual compact** (if restart doesn't work):
   - Docker Desktop → Settings → Resources → Advanced
   - Or use CLI: `docker run --rm -it --privileged --pid=host debian nsenter -t 1 -m -u -n -i sh -c "fstrim /var/lib/docker"`

### Monitoring

Check Docker.raw size periodically:

```bash
ls -lh ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw
```

If it grows beyond **100GB** despite cleanup, consider:

- Reviewing container logs for excessive output
- Checking for containers generating large amounts of data
- Increasing log rotation frequency (reduce `max-size` or `max-file`)

## Next Steps

1. **Restart Docker containers** to apply new logging configuration:

   ```bash
   cd /Users/maxholden/Documents/GitHub/Larrak
   docker compose down
   docker compose up -d
   ```

2. **Monitor log sizes** after restart to verify rotation is working

3. **Schedule monthly cleanup** using the commands above
