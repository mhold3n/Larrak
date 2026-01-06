# Log Directory Structure

This directory contains all Larrak logging output, organized by type and retention policy.

## Directory Structure

```
logs/
├── structured/          # Level-based JSON logs
│   ├── errors/          # ERROR+ logs (retained indefinitely)
│   ├── warnings/        # WARNING logs (7 days retention)
│   └── info/            # INFO logs (24 hours retention)
├── sessions/            # Session-scoped logs
│   └── {run_id}/        # Per-session directory
│       ├── metadata.json    # Session metadata
│       ├── full.log         # All log levels for this session
│       └── debug.log        # DEBUG logs (rotating 100MB)
├── metrics/             # Extracted metrics in JSONL format
│   ├── optimization_metrics.jsonl
│   └── cem_validation_metrics.jsonl
└── archives/            # Compressed session archives (30+ days)
    └── {run_id}_YYYYMMDD.tar.gz
```

## Retention Policies

- **ERROR logs**: Kept indefinitely
- **WARNING logs**: 7 days
- **INFO logs**: 24 hours
- **Session logs**: 30 days (then archived)
- **Archives**: Deleted when total log dir exceeds size limit (default 10GB)

## Cleanup

Run automated cleanup:

```bash
python -m campro.utils.log_cleanup
```

Dry-run (see what would be deleted):

```bash
python -m campro.utils.log_cleanup --dry-run
```

Set custom size limit:

```bash
python -m campro.utils.log_cleanup --max-size-gb 20
```

## Usage

Logs are automatically written when running optimization or orchestration sessions:

```python
from campro.logging import configure_root_logging, session_logging

# Configure at app startup
configure_root_logging(level=logging.INFO)

# Use session context for long-running tasks
with session_logging("my_run_id", params={"batch_size": 10}):
    # All logs during this context go to logs/sessions/my_run_id/
    logger.info("Starting optimization")
```
