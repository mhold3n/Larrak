# Run This on Your Mac Host

## Step 1: Find Your Project Directory

The docker-compose.yml file is in your project root. You need to navigate there first.

If you're using VS Code with a devcontainer, your project is likely at one of these locations:

```bash
# Try these paths:
cd ~/workspace
# or
cd ~/Projects/your-project-name
# or
cd /path/to/your/project
```

## Step 2: Verify docker-compose.yml exists

```bash
ls -la docker-compose.yml
```

You should see the file. If not, you're in the wrong directory.

## Step 3: Run Docker Compose

Once you're in the correct directory (where docker-compose.yml is):

```bash
docker compose up -d weaviate outline-api
```

This will:
- Pull the Weaviate image
- Build the outline-api image
- Start both containers

## Step 4: Verify Containers Started

```bash
docker compose ps
```

You should see:
- weaviate (running)
- larrak-outline-api (running)

## Step 5: Access Dashboard

Open in browser: http://localhost:5001/
