import os
import sys
import threading
import uuid
from typing import Any, List, Optional

from provenance.db import db
from provenance.spec import RunEndEvent, RunStartEvent

# Thread-local storage for current run context
_local = threading.local()

def get_current_run_id() -> Optional[str]:
    return getattr(_local, "run_id", None) or os.environ.get("LARRAK_RUN_ID")


def _set_current_run_id(run_id: Optional[str]) -> None:
    _local.run_id = run_id

class RunContext:
    def __init__(
        self,
        module_id: str,
        args: Optional[List[str]] = None,
        capture_output: bool = False
    ):
        self.module_id = module_id
        self.args = args or sys.argv[1:]
        self.run_id = str(uuid.uuid4())
        self.capture_output = capture_output
        self.env = {
            "user": os.environ.get("USERNAME", "unknown"),
            "host": os.environ.get("COMPUTERNAME", "unknown"),
            "cwd": os.getcwd()
        }

    def start(self):
        _set_current_run_id(self.run_id)

        # 1. DB Start
        db.start_run(self.run_id, self.module_id, self.args, self.env)

        # 2. Log Start Event
        event = RunStartEvent(
            run_id=self.run_id,
            module_id=self.module_id,
            args=self.args
        )
        db.log_event(event)

        print(
            f"[Provenance] Started Run {self.run_id} for Module {self.module_id}"
        )
        return self.run_id

    def end(self, status: str = "SUCCESS") -> None:
        # Log End Event
        event = RunEndEvent(
            run_id=self.run_id,
            status=status
        )
        db.log_event(event)

        # DB End
        db.end_run(self.run_id, status)

        print(f"[Provenance] Ended Run {self.run_id} with status {status}")
        _set_current_run_id(None)

    def __enter__(self) -> "RunContext":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "FAILURE" if exc_type else "SUCCESS"
        self.end(status)


# Helper for easy usage
def run_context(module_id: str) -> RunContext:
    return RunContext(module_id)

