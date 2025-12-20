# Environment Management Guide

This document explains how Larrak manages conda environment dependencies to ensure consistency across all development machines.

## Overview

Larrak uses a **lock file workflow** to guarantee reproducible environments:

| File                   | Purpose                             |
| ---------------------- | ----------------------------------- |
| `environment.yml`      | Source of truth with version ranges |
| `environment-lock.yml` | Auto-generated exact versions       |
| `conda-lock.yml`       | Cross-platform lock file            |
| `requirements.txt`     | Pip fallback (auto-synced)          |

## Setting Up a New Machine

### Option 1: Use Lock File (Recommended)

For exact reproducibility:

```bash
# Clone the repository
git clone https://github.com/mhold3n/Larrak.git
cd Larrak

# Create environment from lock file
conda env create -f environment-lock.yml
conda activate larrak

# Verify installation
python scripts/check_environment.py
```

### Option 2: Use Source File

For latest compatible versions:

```bash
# Create from source
conda env create -f environment.yml
conda activate larrak
```

## Updating Dependencies

### Adding a New Package

1. Edit `environment.yml` to add the package:

   ```yaml
   dependencies:
     - new-package>=1.0.0
   ```

2. Commit and push to `main` or `develop` branch

3. The GitHub Action automatically:
   - Generates new `environment-lock.yml`
   - Generates new `conda-lock.yml`
   - Updates `requirements.txt`
   - Commits the lock files

### Manual Lock File Generation

If you need to generate lock files locally:

```bash
# Sync requirements.txt
python scripts/sync_requirements.py --generate

# Export environment lock
conda env export --no-builds > environment-lock.yml

# Generate cross-platform locks (requires conda-lock)
pip install conda-lock
conda-lock -f environment.yml -p linux-64 -p win-64 -p osx-64
```

## Checking Sync Status

To verify `requirements.txt` matches `environment.yml`:

```bash
python scripts/sync_requirements.py --check
```

This check runs automatically in CI.

## Troubleshooting

### Pip/Conda Conflicts

If you see warnings about conflicting packages:

```bash
# Check for conflicts
python scripts/check_conda_environment.py

# Reinstall via conda (preferred)
conda install -c conda-forge package-name
```

### Environment Validation Fails

Run the comprehensive check:

```bash
python scripts/check_environment.py --verbose
```

### Lock File Out of Date

If CI fails because lock files are stale:

1. Manually trigger the lock workflow:

   - Go to Actions → "Lock Conda Environment" → "Run workflow"

2. Or push a change to `environment.yml` to trigger auto-update

## GitHub Workflows

| Workflow                   | Trigger                   | Purpose                       |
| -------------------------- | ------------------------- | ----------------------------- |
| `lock-environment.yml`     | Push to `environment.yml` | Generate lock files           |
| `python-package-conda.yml` | Push/PR to main/develop   | Test environment + sync check |
