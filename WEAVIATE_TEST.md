# Testing Weaviate Connection

## Test 1: Ready Endpoint
```bash
curl http://localhost:8080/v1/.well-known/ready
```
Expected: `{"ready":true}`

## Test 2: With Verbose Output
```bash
curl -v http://localhost:8080/v1/.well-known/ready
```
This shows HTTP headers and response details.

## Test 3: Schema Endpoint
```bash
curl http://localhost:8080/v1/schema
```
This shows the Weaviate schema (may be empty if not initialized).

## Test 4: Meta Endpoint
```bash
curl http://localhost:8080/v1/meta
```
This shows Weaviate version and configuration.

## If Weaviate is Not Responding

Check container logs:
```bash
docker compose logs weaviate | tail -20
```

Check if container is running:
```bash
docker compose ps weaviate
```

## Dashboard API Connection

The dashboard API connects to Weaviate using:
- URL: `http://weaviate:8080` (from inside the container)
- This works via Docker networking

Even if `localhost:8080` doesn't respond from your Mac, the dashboard API container can still connect to Weaviate using the service name `weaviate`.
