"""Automated log cleanup script for managing disk space.

This script enforces retention policies:
- ERROR logs: Kept indefinitely
- WARNING logs: 7 days
- INFO logs: 24 hours
- Session logs: 30 days (compressed)
- Max total size: configurable (default 10GB)

Usage:
    python -m campro.utils.log_cleanup [--dry-run] [--max-size-gb N]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def get_log_base_dir() -> Path:
    """Get base directory for logs."""
    import os

    log_dir = os.getenv("LARRAK_LOG_DIR")
    if log_dir:
        return Path(log_dir)

    # Default to project root / logs
    project_root = Path(__file__).parent.parent.parent
    return project_root / "logs"


def cleanup_old_warnings(base_dir: Path, days: int = 7, dry_run: bool = False) -> int:
    """Remove warning logs older than specified days.

    Args:
        base_dir: Base log directory
        days: Number of days to retain
        dry_run: If True, only report what would be deleted

    Returns:
        Number of files deleted/would be deleted
    """
    warnings_dir = base_dir / "structured" / "warnings"
    if not warnings_dir.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted_count = 0

    for log_file in warnings_dir.glob("*.jsonl"):
        # Parse date from filename (YYYY-MM-DD.jsonl)
        try:
            file_date_str = log_file.stem
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            if file_date < cutoff:
                if dry_run:
                    logger.info("Would delete: %s", log_file)
                else:
                    log_file.unlink()
                    logger.info("Deleted: %s", log_file)
                deleted_count += 1
        except ValueError:
            logger.warning("Skipping file with unexpected name format: %s", log_file)

    return deleted_count


def cleanup_old_info(base_dir: Path, hours: int = 24, dry_run: bool = False) -> int:
    """Remove info logs older than specified hours.

    Args:
        base_dir: Base log directory
        hours: Number of hours to retain
        dry_run: If True, only report what would be deleted

    Returns:
        Number of files deleted/would be deleted
    """
    info_dir = base_dir / "structured" / "info"
    if not info_dir.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    deleted_count = 0

    for log_file in info_dir.glob("*.jsonl"):
        # Parse date from filename
        try:
            file_date_str = log_file.stem
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            if file_date < cutoff:
                if dry_run:
                    logger.info("Would delete: %s", log_file)
                else:
                    log_file.unlink()
                    logger.info("Deleted: %s", log_file)
                deleted_count += 1
        except ValueError:
            logger.warning("Skipping file with unexpected name format: %s", log_file)

    return deleted_count


def archive_old_sessions(base_dir: Path, days: int = 30, dry_run: bool = False) -> tuple[int, int]:
    """Archive old session logs to compressed tar.gz.

    Args:
        base_dir: Base log directory
        days: Number of days before archiving
        dry_run: If True, only report what would be archived

    Returns:
        Tuple of (sessions_archived, sessions_deleted)
    """
    sessions_dir = base_dir / "sessions"
    if not sessions_dir.exists():
        return (0, 0)

    archive_dir = base_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    archived_count = 0
    deleted_count = 0

    for session_path in sessions_dir.iterdir():
        if not session_path.is_dir():
            continue

        # Check metadata for session end time
        metadata_file = session_path / "metadata.json"
        if not metadata_file.exists():
            # Use directory modification time as fallback
            mtime = datetime.fromtimestamp(session_path.stat().st_mtime, tz=timezone.utc)
        else:
            try:
                import json

                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                end_time_str = metadata.get("end_time")
                if end_time_str:
                    from datetime import datetime

                    mtime = datetime.fromisoformat(end_time_str)
                else:
                    mtime = datetime.fromtimestamp(session_path.stat().st_mtime, tz=timezone.utc)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("Error reading metadata for %s: %s", session_path, e)
                continue

        if mtime < cutoff:
            archive_name = f"{session_path.name}_{mtime.strftime('%Y%m%d')}.tar.gz"
            archive_path = archive_dir / archive_name

            if dry_run:
                logger.info("Would archive: %s -> %s", session_path, archive_path)
                archived_count += 1
            else:
                try:
                    # Create compressed archive
                    with tarfile.open(archive_path, "w:gz") as tar:
                        tar.add(session_path, arcname=session_path.name)

                    # Remove original directory
                    shutil.rmtree(session_path)

                    logger.info("Archived: %s -> %s", session_path, archive_path)
                    archived_count += 1
                except (OSError, tarfile.TarError) as e:
                    logger.error("Failed to archive %s: %s", session_path, e)

    return (archived_count, deleted_count)


def get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for path in directory.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except (PermissionError, FileNotFoundError):
        pass
    return total


def enforce_max_size(base_dir: Path, max_size_gb: float, dry_run: bool = False) -> int:
    """Enforce maximum log directory size by removing oldest archives.

    Args:
        base_dir: Base log directory
        max_size_gb: Maximum size in GB
        dry_run: If True, only report what would be deleted

    Returns:
        Number of archives deleted
    """
    current_size = get_directory_size(base_dir)
    max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

    if current_size <= max_size_bytes:
        return 0

    # Remove oldest archives first
    archive_dir = base_dir / "archives"
    if not archive_dir.exists():
        return 0

    archives = sorted(archive_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)
    deleted_count = 0

    for archive in archives:
        if current_size <= max_size_bytes:
            break

        file_size = archive.stat().st_size
        if dry_run:
            logger.info("Would delete archive: %s (%.2f MB)", archive, file_size / (1024 * 1024))
        else:
            archive.unlink()
            logger.info("Deleted archive: %s (%.2f MB)", archive, file_size / (1024 * 1024))
            current_size -= file_size

        deleted_count += 1

    return deleted_count


def main() -> None:
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Clean up old log files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting"
    )
    parser.add_argument(
        "--max-size-gb", type=float, default=10.0, help="Maximum log directory size in GB"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    base_dir = get_log_base_dir()

    if not base_dir.exists():
        logger.warning("Log directory does not exist: %s", base_dir)
        return

    logger.info("Starting log cleanup for: %s", base_dir)
    if args.dry_run:
        logger.info("DRY RUN - no files will be deleted")

    # Get initial size
    initial_size = get_directory_size(base_dir)
    logger.info("Current log directory size: %.2f GB", initial_size / (1024**3))

    # Run cleanup tasks
    warnings_deleted = cleanup_old_warnings(base_dir, days=7, dry_run=args.dry_run)
    logger.info("Warnings cleaned: %d files", warnings_deleted)

    info_deleted = cleanup_old_info(base_dir, hours=24, dry_run=args.dry_run)
    logger.info("Info logs cleaned: %d files", info_deleted)

    sessions_archived, _ = archive_old_sessions(base_dir, days=30, dry_run=args.dry_run)
    logger.info("Sessions archived: %d", sessions_archived)

    archives_deleted = enforce_max_size(base_dir, args.max_size_gb, dry_run=args.dry_run)
    logger.info("Archives deleted to enforce size limit: %d", archives_deleted)

    # Get final size
    final_size = get_directory_size(base_dir)
    space_freed = initial_size - final_size
    logger.info("Final log directory size: %.2f GB", final_size / (1024**3))
    logger.info("Space freed: %.2f MB", space_freed / (1024 * 1024))

    if args.dry_run:
        logger.info("This was a dry run. Rerun without --dry-run to actually delete files.")


if __name__ == "__main__":
    main()
