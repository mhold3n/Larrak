import datetime
import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# --- Strings & IDs ---
# Using strict types to avoid confusion
ModuleID = str
RunID = str
ArtifactID = str
OriginID = str

# --- Enums ---

class EntityType(enum.Enum):
    ORIGIN_FILE = "origin_file"
    GENERATED_FILE = "generated_file"
    MODULE = "module"
    RUN = "run"
    DATASET = "dataset"

class FileRole(enum.Enum):
    CONFIG = "config"
    INPUT = "input"
    OUTPUT = "output"
    CACHE = "cache"
    REPORT = "report"
    MODEL = "model"
    PLOT = "plot"
    BINARY = "binary"
    LOG = "log"
    INDEX = "index"
    DASHBOARD = "dashboard"
    SOURCE = "source"
    UNKNOWN = "unknown"

class TagKind(enum.Enum):
    ORIGIN = "origin"
    GENERATED = "generated"

class ParamType(enum.Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    PATH = "path"

# --- Data Classes for Inventory (Static) ---

@dataclass
class ParamSpec:
    name: str
    type: ParamType
    default: Any
    description: str
    options: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    required: bool = True

@dataclass
class OriginFile:
    path: str
    origin_id: OriginID
    role: FileRole
    description: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModuleSpec:
    """Declared metadata for a module (from registry)."""
    module_id: ModuleID
    entrypoint: str  # Path to script or command
    description: str
    owner: str  # Team or person
    expected_inputs: List[Dict[str, Any]] = field(default_factory=list)
    expected_outputs: List[Dict[str, Any]] = field(default_factory=list)
    params: List[ParamSpec] = field(default_factory=list)

# --- Data Classes for Runtime (Dynamic) ---

@dataclass
class RunContext:
    run_id: RunID
    module_id: ModuleID
    start_time: datetime.datetime
    args: List[str]
    env: Dict[str, str]
    # Mutable state during run
    tags: Dict[str, Any] = field(default_factory=dict)
    end_time: Optional[datetime.datetime] = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILURE

@dataclass
class Artifact:
    """A specific instance of a file produced by a run."""
    artifact_id: ArtifactID
    path: str
    content_hash: Optional[str]
    run_id: RunID
    producer_module_id: ModuleID
    role: FileRole
    size_bytes: int
    creation_time: Optional[datetime.datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Events for the Stream ---

@dataclass
class Event:
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    run_id: Optional[RunID] = None

@dataclass
class RunStartEvent(Event):
    module_id: str = ""
    args: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunEndEvent(Event):
    status: str = "SUCCESS"
    duration_ms: float = 0.0

@dataclass
class FileEvent(Event):
    path: str = ""
    op: str = "read" # read, write, open, delete
    size: int = 0
    file_hash: Optional[str] = None

@dataclass
class CheckpointEvent(Event):
    name: str = ""
    expected: Any = None
    observed: Any = None
    passed: bool = True

@dataclass
class RunSummary:
    run_id: RunID
    module_id: ModuleID
    start_time: Optional[Union[datetime.datetime, str]]
    end_time: Optional[Union[datetime.datetime, str]]
    status: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
