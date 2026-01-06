# Verify Services Are Running

## Step 1: Check Container Status (Run on Your Mac)

```bash
docker compose ps
```

You should see:
- `weaviate` - Status: "Up"
- `larrak-outline-api` - Status: "Up"

## Step 2: Check Logs (Run on Your Mac)

```bash
# Check Weaviate logs
docker compose logs weaviate

# Check Dashboard API logs
docker compose logs outline-api
```

Look for:
- Weaviate: "listening on 0.0.0.0:8080"
- Dashboard API: "Starting dashboard API on http://0.0.0.0:5001"

## Step 3: Test from Your Mac Browser

Open these URLs in your Mac browser (not from inside the container):

- **Dashboard**: http://localhost:5001/
- **Weaviate**: http://localhost:8080

## Step 4: Test from Mac Terminal

```bash
# Test Weaviate
curl http://localhost:8080/v1/.well-known/ready

# Test Dashboard API
curl http://localhost:5001/api/modules
```

## Common Issues

### Containers not starting
- Check logs: `docker compose logs`
- Check if ports 8080 and 5001 are already in use
- Verify docker-compose.yml is correct

### Port conflicts
If ports are in use, you can change them in docker-compose.yml:
```yaml
ports:
  - "8081:8080"  # Change host port
  - "5002:5001"  # Change host port
```

### Containers starting but not accessible
- Wait 10-20 seconds for services to fully start
- Check container logs for errors
- Verify firewall isn't blocking ports
