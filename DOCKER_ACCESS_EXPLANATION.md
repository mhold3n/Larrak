# What's Happening: Docker Command Not Found

## The Issue

You're running commands **inside a development container** (devcontainer), but Docker is not installed or accessible within this container.

## Why This Happens

1. **You're in a container**: The environment is a Micromamba-based devcontainer
2. **Docker not installed**: The `docker` binary is not in the PATH
3. **Docker socket not mounted**: `/var/run/docker.sock` is not accessible

## Solutions

### Option 1: Run Docker Commands on Host (Easiest)

**Exit this container** and run docker commands on your **host machine**:

```bash
# On your HOST (not in container):
cd /workspace
docker compose up -d weaviate outline-api
```

### Option 2: Enable Docker-in-Docker (DevContainer)

If you want Docker available inside this container, you need to configure Docker-in-Docker in `.devcontainer/devcontainer.json`:

```json
{
  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  }
}
```

Then rebuild the devcontainer.

### Option 3: Mount Docker Socket (Alternative)

Mount the host's Docker socket (requires host Docker to be running):

```json
{
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind"
  ]
}
```

## Recommended Approach

**Use Option 1** - Run docker commands on your host machine. The services will be accessible from both the host and this container via `localhost` ports.

Once you run `docker compose up -d weaviate outline-api` on your host:
- Dashboard: http://localhost:5001/
- Weaviate: http://localhost:8080
- WebSocket: ws://localhost:8765

These will be accessible from both the host browser AND from within this container.
