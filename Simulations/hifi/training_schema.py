"""Training Data Schema for HiFi Surrogates.

Defines the structure and normalization for training data generated
from CFD/FEA simulations.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class TrainingRecord:
    """
    Single training sample from HiFi simulation.

    Contains normalized inputs and solver outputs for one DOE point.
    """

    # Identifiers
    case_id: int
    run_id: str = ""

    # Geometric inputs (normalized to [0, 1])
    bore: float = 0.0  # Actual range: 70-100 mm
    stroke: float = 0.0  # Actual range: 75-110 mm
    cr: float = 0.0  # Actual range: 10-16

    # Operating inputs (normalized)
    rpm: float = 0.0  # Actual range: 1000-7000
    load: float = 0.0  # Actual range: 0.2-1.0

    # Thermal outputs
    T_crown_max: Optional[float] = None  # [K], range ~400-650
    T_liner_max: Optional[float] = None  # [K], range ~350-500
    htc_mean: Optional[float] = None  # [W/m²K], range ~200-2000

    # Structural outputs
    von_mises_max: Optional[float] = None  # [MPa], range ~50-300
    displacement_max: Optional[float] = None  # [mm], range ~0.01-0.5
    safety_factor: Optional[float] = None  # [], range ~1.0-5.0

    # Flow outputs
    cd_effective: Optional[float] = None  # [], range ~0.4-0.75
    swirl_ratio: Optional[float] = None  # [], range ~0-3
    tumble_ratio: Optional[float] = None  # [], range ~0-3

    # Combustion outputs
    p_max: Optional[float] = None  # [bar], range ~50-120
    imep: Optional[float] = None  # [bar], range ~8-18
    heat_release_rate_max: Optional[float] = None  # [J/deg]

    # Metadata
    solver_success: bool = True
    computation_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class NormalizationParams:
    """
    Min-max normalization parameters for inputs and outputs.
    """

    # Input ranges
    bore_range: tuple[float, float] = (70.0, 100.0)  # mm
    stroke_range: tuple[float, float] = (75.0, 110.0)  # mm
    cr_range: tuple[float, float] = (10.0, 16.0)
    rpm_range: tuple[float, float] = (1000.0, 7000.0)
    load_range: tuple[float, float] = (0.2, 1.0)

    # Output ranges (for bounded outputs)
    T_crown_range: tuple[float, float] = (350.0, 700.0)  # K
    T_liner_range: tuple[float, float] = (300.0, 550.0)  # K
    von_mises_range: tuple[float, float] = (0.0, 400.0)  # MPa
    cd_range: tuple[float, float] = (0.3, 0.8)
    p_max_range: tuple[float, float] = (40.0, 150.0)  # bar

    def normalize_inputs(self, record: TrainingRecord) -> np.ndarray:
        """Convert raw record to normalized input vector."""

        def norm(val, lo, hi):
            return (val - lo) / (hi - lo) if hi > lo else 0.5

        return np.array(
            [
                norm(record.bore, *self.bore_range),
                norm(record.stroke, *self.stroke_range),
                norm(record.cr, *self.cr_range),
                norm(record.rpm, *self.rpm_range),
                norm(record.load, *self.load_range),
            ],
            dtype=np.float32,
        )

    def denormalize_output(self, name: str, normalized_value: float) -> float:
        """Convert normalized output back to physical units."""
        ranges = {
            "T_crown_max": self.T_crown_range,
            "T_liner_max": self.T_liner_range,
            "von_mises_max": self.von_mises_range,
            "cd_effective": self.cd_range,
            "p_max": self.p_max_range,
        }
        if name in ranges:
            lo, hi = ranges[name]
            return normalized_value * (hi - lo) + lo
        return normalized_value

    def save(self, path: str) -> None:
        """Save normalization parameters."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str) -> "NormalizationParams":
        return cls(**json.loads(Path(path).read_text()))


class TrainingDataset:
    """
    Dataset container for training HiFi surrogates.

    Handles:
    - Loading from Parquet/JSON
    - Train/val splits
    - Normalization
    - Conversion to PyTorch tensors
    """

    def __init__(
        self, records: list[TrainingRecord], norm_params: NormalizationParams | None = None
    ):
        self.records = records
        self.norm_params = norm_params or NormalizationParams()

    @classmethod
    def from_parquet(cls, path: str) -> "TrainingDataset":
        """Load from Parquet file."""
        import pandas as pd

        df = pd.read_parquet(path)

        records = []
        for _, row in df.iterrows():
            record = TrainingRecord(
                case_id=row.get("case_id", 0),
                bore=row.get("param_bore_mm", 85),
                stroke=row.get("param_stroke_mm", 90),
                cr=row.get("param_compression_ratio", 12),
                rpm=row.get("param_rpm", 3000),
                load=row.get("param_load_fraction", 1.0),
                T_crown_max=row.get("output_T_crown_max"),
                von_mises_max=row.get("output_von_mises_max"),
                cd_effective=row.get("output_cd_effective"),
                p_max=row.get("output_p_max"),
                solver_success=row.get("success", True),
            )
            records.append(record)

        return cls(records)

    @classmethod
    def from_json(cls, path: str) -> "TrainingDataset":
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())

        records = []
        for item in data:
            params = item.get("params", {})
            outputs = item.get("outputs", {})

            record = TrainingRecord(
                case_id=item.get("case_id", 0),
                bore=params.get("bore_mm", 85),
                stroke=params.get("stroke_mm", 90),
                cr=params.get("compression_ratio", 12),
                rpm=params.get("rpm", 3000),
                load=params.get("load_fraction", 1.0),
                T_crown_max=outputs.get("T_crown_max"),
                von_mises_max=outputs.get("von_mises_max"),
                cd_effective=outputs.get("cd_effective"),
                p_max=outputs.get("p_max"),
                solver_success=item.get("success", True),
            )
            records.append(record)

        return cls(records)

    def get_thermal_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (X, y) for thermal surrogate training."""
        X, y = [], []
        for r in self.records:
            if r.T_crown_max is not None and r.solver_success:
                X.append(self.norm_params.normalize_inputs(r))
                # Normalize T_crown_max to [0, 1]
                lo, hi = self.norm_params.T_crown_range
                y.append((r.T_crown_max - lo) / (hi - lo))

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    def get_structural_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (X, y) for structural surrogate training."""
        X, y = [], []
        for r in self.records:
            if r.von_mises_max is not None and r.solver_success:
                X.append(self.norm_params.normalize_inputs(r))
                lo, hi = self.norm_params.von_mises_range
                y.append((r.von_mises_max - lo) / (hi - lo))

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    def get_flow_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (X, y) for flow coefficient surrogate."""
        X, y = [], []
        for r in self.records:
            if r.cd_effective is not None and r.solver_success:
                X.append(self.norm_params.normalize_inputs(r))
                lo, hi = self.norm_params.cd_range
                y.append((r.cd_effective - lo) / (hi - lo))

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    def split(
        self, train_frac: float = 0.8, seed: int = 42
    ) -> tuple["TrainingDataset", "TrainingDataset"]:
        """Split into train/validation sets."""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.records))
        split_idx = int(len(indices) * train_frac)

        train_records = [self.records[i] for i in indices[:split_idx]]
        val_records = [self.records[i] for i in indices[split_idx:]]

        return TrainingDataset(train_records, self.norm_params), TrainingDataset(
            val_records, self.norm_params
        )
