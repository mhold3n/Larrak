import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class ArchiveManager:
    """
    Manages archival of old run artifacts to a PERMANENT_ARCHIVE directory.
    Keeps the last N runs/files active, moves older ones to archive while
    preserving directory structure.
    """

    def __init__(self, root_dir: Path, archive_root: Path, retention_count: int = 5):
        self.root_dir = root_dir.resolve()
        self.archive_root = archive_root.resolve()
        self.retention_count = retention_count

    def archive_directory(self, target_relative_path: str, pattern: str = "*"):
        """
        Scans a specific directory relative to root, sorts items by modification time,
        and moves items exceeding retention_count to the archive.

        target_relative_path: Directory to clean (e.g. "logs")
        pattern: Glob pattern for items to consider (files or dirs)
        """
        target_dir = self.root_dir / target_relative_path
        if not target_dir.exists():
            log.warning(
                f"Target directory {target_dir} does not exist. Skipping archive."
            )
            return

        # Find all matching items
        items = list(target_dir.glob(pattern))

        # Sort by modification time (descending: newest first)
        items.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Identify items to archive
        to_archive = items[self.retention_count :]

        if not to_archive:
            log.info(
                f"No items to archive in {target_relative_path} (Count: {len(items)} <= {self.retention_count})"
            )
            return

        log.info(f"Archiving {len(to_archive)} items from {target_relative_path}...")

        for item in to_archive:
            try:
                # Calculate destination path
                rel_path = item.relative_to(self.root_dir)
                dest_path = self.archive_root / rel_path

                # Create parent directories in archive
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Move item
                shutil.move(str(item), str(dest_path))
                log.debug(f"Archived: {item.name} -> {dest_path}")

            except Exception as e:
                log.error(f"Failed to archive {item.name}: {e}")

    def run_default_cleanup(self):
        """Standard cleanup routine for Larrak."""
        # Clean logs
        self.archive_directory("logs", "*.log")
        self.archive_directory("logs", "*.txt")  # verification logs

        # Clean plots? User said "Logs not pertaining to last 5 runs".
        # Assuming plots are also ephemeral outputs.
        # But plots are in subdirectories by shape.
        # Let's start with logs as explicitly requested.
