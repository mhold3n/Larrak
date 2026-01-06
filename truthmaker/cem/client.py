"""
CEM Client: Python interface to the compiled CEM runtime.

The CEM acts as "project manager" - this client queries it for:
- Feasible parameter envelopes (non-differentiable bounds)
- Motion profile validation (regime checks, manufacturability gates)
- Physics-informed initial guesses

The CEM owns NON-DIFFERENTIABLE constraints.
DIFFERENTIABLE constraints stay in the CasADi NLP.

Supports two modes:
- gRPC mode: communicates with compiled C# CEM service
- Mock mode: uses Python fallback for development/testing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import grpc
import numpy as np

# Try to import generated protos; handle failure gracefully for mock mode
try:
    from truthmaker.cem import cem_pb2, cem_pb2_grpc

    GRPC_AVAILABLE = True
except ImportError as e:
    import sys

    print(f"[ERROR] GRPC Import Failed: {e}", file=sys.stderr)
    GRPC_AVAILABLE = False
    cem_pb2 = None  # type: ignore[assignment]
    cem_pb2_grpc = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class ViolationCode(IntEnum):
    """Structured violation codes matching C# enum."""

    NONE = 0

    # Thermodynamic (1xx)
    THERMO_MAX_PRESSURE = 100
    THERMO_MAX_TEMPERATURE = 101
    THERMO_MAX_CROWN_TEMP = 102
    THERMO_LAMBDA_TOO_RICH = 103
    THERMO_LAMBDA_TOO_LEAN = 104
    THERMO_NEGATIVE_MASS = 105
    THERMO_BACKFLOW_DETECTED = 106
    THERMO_KNOCKING_RISK = 107
    THERMO_COMBUSTION_UNSTABLE = 108
    THERMO_HEAT_TRANSFER_LIMIT = 109
    THERMO_MODEL_ASSUMPTION_BROKEN = 110
    THERMO_FMEP_TOO_HIGH = 111
    THERMO_EXHAUST_TEMP_LIMIT = 112
    THERMO_BRAKE_EFFICIENCY_LOW = 113

    # Kinematic (2xx)
    KINEMATIC_MAX_JERK = 200
    KINEMATIC_MAX_ACCELERATION = 201
    KINEMATIC_PERIODICITY_BROKEN = 202
    KINEMATIC_PHASE_MISMATCH = 203
    KINEMATIC_VELOCITY_REVERSAL = 204
    KINEMATIC_CONTACT_FORCE_LIMIT = 205

    # Gear geometry (3xx)
    GEAR_ENVELOPE_STROKE = 300
    GEAR_MIN_RADIUS = 301
    GEAR_MAX_RADIUS = 302
    GEAR_CURVATURE_UNDERCUT = 303
    GEAR_INTERFERENCE = 304
    GEAR_CENTER_DISTANCE_INCONSISTENT = 305
    GEAR_TOPOLOGY_INVALID = 306
    GEAR_CONTACT_RATIO_LOW = 307
    GEAR_TOOTH_THICKNESS_MIN = 308
    GEAR_PROFILE_DEVIATION = 309

    # Manufacturing (4xx)
    AM_MIN_FEATURE_SIZE = 400
    AM_MAX_OVERHANG_ANGLE = 401
    AM_SUPPORT_REQUIRED = 402
    AM_TOOL_ACCESS = 403
    AM_SURFACE_FINISH = 404
    AM_LAYER_ADHESION = 405
    AM_THERMAL_DISTORTION = 406
    AM_POWDER_REMOVAL = 407
    MACHINING_UNDERCUT_IMPOSSIBLE = 410
    MACHINING_TOOL_INTERFERENCE = 411
    MACHINING_SURFACE_CURVATURE = 412
    CASTING_WALL_THICKNESS = 420
    CASTING_DRAFT_ANGLE = 421

    # Model assumptions (5xx)
    ASSUMPTION_REGIME_INVALID = 500
    ASSUMPTION_CORRELATION_OUT_OF_RANGE = 501
    ASSUMPTION_GRID_TOO_COARSE = 502
    ASSUMPTION_STEADY_STATE_VIOLATED = 503
    ASSUMPTION_QUASI_STATIC_VIOLATED = 504

    # Configuration (6xx)
    CONFIG_MISSING_PARAMETER = 600
    CONFIG_INVALID_BOUNDS = 601
    CONFIG_INCONSISTENT_UNITS = 602


class ViolationSeverity(IntEnum):
    """Severity level for violations."""

    INFO = 0
    WARN = 1
    ERROR = 2
    FATAL = 3


class SuggestedActionCode(IntEnum):
    """Suggested corrective actions for automated recovery."""

    NONE = 0
    INCREASE_SMOOTHING = 100
    REDUCE_STROKE = 101
    ADJUST_PHASE = 102
    TIGHTEN_BOUNDS = 200
    RELAX_BOUNDS = 201
    IMPROVE_INITIAL_GUESS = 400
    MANUAL_REVIEW_REQUIRED = 900


class OperatingRegime(IntEnum):
    """Operating regime classification for surrogate training stratification."""

    UNKNOWN = 0
    IDLE = 1  # Low RPM, low load
    CRUISE = 2  # Medium RPM, partial load
    FULL_LOAD = 3  # High RPM/load, near limits


@dataclass
class ConstraintViolation:
    """Structured violation with semantic information."""

    code: ViolationCode
    severity: ViolationSeverity
    message: str
    margin: float | None = None  # Distance to feasibility
    suggested_action: SuggestedActionCode = SuggestedActionCode.NONE
    affected_variables: list[str] | None = None
    metrics: dict[str, float] | None = None


@dataclass
class ValidationReport:
    """Aggregated validation result with all violations."""

    is_valid: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    cem_version: str = "mock-0.1.0"
    config_hash: str = "default"
    regime_id: int = 0  # 0=unknown, 1=idle, 2=cruise, 3=full_load
    geometry_data: dict[str, Any] | None = None

    @property
    def max_severity(self) -> ViolationSeverity:
        if not self.violations:
            return ViolationSeverity.INFO
        return max(v.severity for v in self.violations)

    def get_by_code(self, code: ViolationCode) -> list[ConstraintViolation]:
        return [v for v in self.violations if v.code == code]

    def has_errors(self) -> bool:
        return any(v.severity >= ViolationSeverity.ERROR for v in self.violations)


@dataclass
class OperatingEnvelope:
    """Feasible operating envelope returned by CEM."""

    boost_range: tuple[float, float]
    fuel_range: tuple[float, float]
    motion_bounds: tuple[float, float]
    feasible: bool
    config_hash: str

    @property
    def boost_min(self) -> float:
        return self.boost_range[0]

    @property
    def boost_max(self) -> float:
        return self.boost_range[1]

    @property
    def fuel_min_mg(self) -> float:
        return self.fuel_range[0]

    @property
    def fuel_max_mg(self) -> float:
        return self.fuel_range[1]


@dataclass
class GearInitialGuess:
    """Physics-informed initial guess for gear profiles."""

    Rp: np.ndarray
    Rr: np.ndarray
    C: np.ndarray
    phase_offset: float
    mean_centerline: float


class CEMClient:
    """Client for interacting with the CEM service (or mock)."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        mock: bool = False,
        config: dict | None = None,
    ):
        # Read from environment if not explicitly provided
        import os

        cem_service_url = os.getenv("CEM_SERVICE_URL", "localhost:50051")
        if ":" in cem_service_url:
            default_host, default_port_str = cem_service_url.rsplit(":", 1)
            default_port = int(default_port_str)
        else:
            default_host = cem_service_url
            default_port = 50051

        self.host = host or default_host
        self.port = port or default_port
        self.mock = mock or (not GRPC_AVAILABLE)
        self.config = config or {}
        self._channel: grpc.Channel | None = None
        self._stub: Any = None

        # Config defaults
        self.max_jerk = self.config.get("max_jerk", 500.0)
        self.max_radius = self.config.get("max_radius", 80.0)
        self.min_radius = self.config.get("min_radius", 20.0)

        if not self.mock:
            self._connect()

    def _connect(self) -> None:
        """Establish gRPC connection."""
        try:
            target = f"{self.host}:{self.port}"
            self._channel = grpc.insecure_channel(target)
            self._stub = cem_pb2_grpc.CEMServiceStub(self._channel)  # type: ignore[union-attr]
            # Simple health check
            try:
                response = self._stub.HealthCheck(
                    cem_pb2.HealthCheckRequest(),  # type: ignore[attr-defined,union-attr]  # pylint: disable=no-member
                    timeout=2.0,
                )
                if not response.healthy:
                    log.warning(f"[CEM] Service reported unhealthy: {response.status}")
            except grpc.RpcError as e:
                import sys

                print(f"[ERROR] HealthCheck failed: {e}", file=sys.stderr)
                log.warning(f"[CEM] Failed to connect to service at {target}: {e}")
                log.warning("[CEM] Falling back to mock/local mode.")
                self.mock = True
        except Exception as e:
            import sys

            print(f"[ERROR] Connection setup failed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            log.error(f"[CEM] Connection error: {e}")
            self.mock = True

    def __enter__(self) -> CEMClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._channel:
            self._channel.close()

    def validate_motion(self, x_profile: np.ndarray, theta: np.ndarray) -> ValidationReport:
        """Validate motion profile against constraints."""
        if self.mock:
            return self._validate_motion_mock(x_profile, theta)
        else:
            return self._validate_motion_grpc(x_profile, theta)

    def _validate_motion_mock(self, x_profile: np.ndarray, theta: np.ndarray) -> ValidationReport:
        """Mock validation logic."""
        # Defensive: Generate theta if None (prevents TypeError in np.gradient)
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, len(x_profile))

        # Simple jerk check
        dx = np.gradient(x_profile, theta)
        ddx = np.gradient(dx, theta)
        dddx = np.gradient(ddx, theta)
        max_jerk = np.max(np.abs(dddx))

        violations = []
        if max_jerk > self.max_jerk:
            violations.append(
                ConstraintViolation(
                    code=ViolationCode.KINEMATIC_MAX_JERK,
                    severity=ViolationSeverity.ERROR,
                    message=f"Mock: Jerk {max_jerk:.1f} > {self.max_jerk}",
                    margin=self.max_jerk - max_jerk,
                    suggested_action=SuggestedActionCode.INCREASE_SMOOTHING,
                )
            )

        return ValidationReport(
            is_valid=len(violations) == 0,
            violations=violations,
            cem_version="mock-0.1.0",
            config_hash="mock",
        )

    def _validate_motion_grpc(self, x_profile: np.ndarray, theta: np.ndarray) -> ValidationReport:
        """Live gRPC implementation of motion validation."""
        # This method was missing in the broken file but indented wrongly?
        # Re-implementing with geometry config populator.
        request = cem_pb2.MotionValidationRequest()  # type: ignore[attr-defined,union-attr]  # pylint: disable=no-member
        request.x_profile_mm.extend(x_profile.tolist())
        request.theta_rad.extend(theta.tolist())
        request.max_jerk = self.max_jerk
        request.max_radius = self.max_radius

        # Populate Geometry Config to trigger shape kernel
        geom_config = cem_pb2.GeometryConfig()  # type: ignore[attr-defined,union-attr]  # pylint: disable=no-member
        geom_config.gear_depth_mm = self.config.get("gear_depth_mm", 10.0)
        geom_config.wall_thickness_mm = self.config.get("wall_thickness_mm", 5.0)
        geom_config.voxel_size_mm = self.config.get("voxel_size_mm", 0.5)
        geom_config.min_margin_mm = self.config.get("min_margin_mm", 10.0)
        geom_config.volume_threshold_mm3 = self.config.get("volume_threshold_mm3", 1.0)

        request.geometry_config.CopyFrom(geom_config)

        # Call service
        try:
            response = self._stub.ValidateMotion(request)  # type: ignore[union-attr]
        except grpc.RpcError as e:
            log.error(f"[CEM] gRPC Call Failed: {e}")
            raise

        # Convert proto to Python dataclasses
        violations = []
        for v in response.report.violations:
            violations.append(
                ConstraintViolation(
                    code=ViolationCode(v.code),
                    severity=ViolationSeverity(v.severity),
                    message=v.message,
                    margin=v.margin if v.HasField("margin") else None,
                    suggested_action=SuggestedActionCode(v.suggested_action),
                    affected_variables=list(v.affected_variables) if v.affected_variables else None,
                    metrics=dict(v.metrics) if v.metrics else None,
                )
            )

        # Extract Geometry Data
        geom_data = None
        if response.HasField("geometry_metadata"):
            geom_data = {
                "voxel_file_path": response.geometry_metadata.voxel_file_path,
                "mesh_file_path": response.geometry_metadata.mesh_file_path,
                "volume_mm3": response.geometry_metadata.volume_mm3,
                "surface_area_mm2": response.geometry_metadata.surface_area_mm2,
                "mesh_hash": response.geometry_metadata.mesh_hash,
            }

        return ValidationReport(
            is_valid=response.report.is_valid,
            violations=violations,
            cem_version=response.report.cem_version,
            config_hash=response.report.config_hash,
            geometry_data=geom_data,
        )

    def get_thermo_envelope(
        self, bore: float, stroke: float, cr: float, rpm: float
    ) -> OperatingEnvelope:
        if self.mock:
            return self._get_thermo_envelope_mock(bore, stroke, cr, rpm)
        else:
            return self._get_thermo_envelope_grpc(bore, stroke, cr, rpm)

    def _get_thermo_envelope_mock(
        self, bore: float, stroke: float, cr: float, rpm: float
    ) -> OperatingEnvelope:
        boost_min = self.config.get("boost_min", 0.5)
        boost_max = self.config.get("boost_max", 6.0)

        # Simplified mock logic
        v_disp = np.pi * bore**2 / 4 * stroke
        T_int = 300
        rho_min = boost_min * 1e5 / (287 * T_int)

        fuel_min = 1.0
        fuel_max = 500.0  # Placeholder logic

        return OperatingEnvelope(
            boost_range=(boost_min, boost_max),
            fuel_range=(fuel_min, fuel_max),
            motion_bounds=(0.0, stroke),
            feasible=True,
            config_hash="mock",
        )

    def _get_thermo_envelope_grpc(
        self, bore: float, stroke: float, cr: float, rpm: float
    ) -> OperatingEnvelope:
        """Live gRPC implementation of envelope generation."""
        # pylint: disable=no-member
        geometry = cem_pb2.EngineGeometry(  # type: ignore[attr-defined,union-attr]
            bore_m=bore, stroke_m=stroke, compression_ratio=cr
        )
        request = cem_pb2.ThermoEnvelopeRequest(  # type: ignore[attr-defined,union-attr]
            geometry=geometry, rpm=rpm
        )

        response = self._stub.GetThermoEnvelope(request)  # type: ignore[union-attr]

        return OperatingEnvelope(
            boost_range=(response.boost_min, response.boost_max),
            fuel_range=(response.fuel_min_mg, response.fuel_max_mg),
            motion_bounds=(response.motion_min_m, response.motion_max_m),
            feasible=response.feasible,
            config_hash=response.config_hash,
        )

    def get_gear_initial_guess(
        self, x_target: np.ndarray, theta: np.ndarray | None = None
    ) -> GearInitialGuess:
        if self.mock:
            return self._get_gear_initial_guess_mock(x_target, theta)
        else:
            return self._get_gear_initial_guess_grpc(x_target, theta)

    def _get_gear_initial_guess_mock(
        self, x_target: np.ndarray, theta: np.ndarray | None = None
    ) -> GearInitialGuess:
        n = len(x_target)
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, n)

        stroke = np.max(x_target) - np.min(x_target)
        x_mean = np.mean(x_target)
        rp_mean = stroke / 2 + 10
        c_mean = np.min(x_target) + rp_mean

        return GearInitialGuess(
            Rp=np.full(n, rp_mean),
            Rr=np.full(n, c_mean + rp_mean),
            C=np.full(n, c_mean),
            phase_offset=0.0,
            mean_centerline=float(x_mean),
        )

    def _get_gear_initial_guess_grpc(
        self, x_target: np.ndarray, theta: np.ndarray | None = None
    ) -> GearInitialGuess:
        n = len(x_target)
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, n)

        request = cem_pb2.GearInitialGuessRequest()  # type: ignore[attr-defined,union-attr]  # pylint: disable=no-member
        request.x_target_mm.extend(x_target.tolist())
        request.theta_rad.extend(theta.tolist())

        response = self._stub.GetGearInitialGuess(request)  # type: ignore[union-attr]

        return GearInitialGuess(
            Rp=np.array(list(response.rp)),
            Rr=np.array(list(response.rr)),
            C=np.array(list(response.c)),
            phase_offset=response.phase_offset_rad,
            mean_centerline=response.mean_centerline_mm,
        )

    # =========================================================================
    # Adaptive Rule Management
    # =========================================================================

    def _init_adaptive_rules(self) -> None:
        """Initialize adaptive rules and state store (lazy)."""
        if hasattr(self, "_adaptive_rules"):
            return

        from truthmaker.cem.rules.adaptation import (
            MaxContactStressAdaptive,
            MaxCrownTemperatureAdaptive,
        )
        from truthmaker.cem.state_store import CEMStateStore

        # Create state store (uses default path)
        state_path = self.config.get("cem_state_path", "cem_state.json")
        self._state_store = CEMStateStore(state_path)

        # Initialize adaptive rules with defaults or config overrides
        self._adaptive_rules = [
            MaxCrownTemperatureAdaptive(
                limit_k=self.config.get("max_crown_temp_k", 573.0),
                learning_rate=self.config.get("adaptation_learning_rate", 0.05),
                min_observations=self.config.get("adaptation_min_obs", 10),
            ),
            MaxContactStressAdaptive(
                limit_mpa=self.config.get("max_contact_stress_mpa", 1500.0),
                learning_rate=self.config.get("adaptation_learning_rate", 0.05),
                min_observations=self.config.get("adaptation_min_obs", 10),
            ),
        ]

        # Load persisted state
        for rule in self._adaptive_rules:
            state = self._state_store.load_rule_state(rule.name)
            if state:
                rule.load_state(state)
                log.info(f"Loaded adaptive state for {rule.name}: limit={rule.limit:.2f}")

    def adapt_rules(
        self,
        truth_data: list[tuple[dict[str, Any], float]],
        run_id: str | None = None,
    ) -> Any:
        """
        Adapt CEM rule parameters based on HiFi simulation results.

        Called by the orchestrator after each batch of expensive simulations.
        Learns from the margin between predicted and actual constraint values.

        Args:
            truth_data: List of (candidate_params, hifi_objective) tuples
            run_id: Optional provenance run ID for tracking

        Returns:
            AdaptationReport summarizing which rules adapted
        """
        from truthmaker.cem.rules.adaptation import AdaptationReport

        self._init_adaptive_rules()

        adapted = []

        for candidate, objective in truth_data:
            # Each candidate dict may contain HiFi result metrics
            for rule in self._adaptive_rules:
                # Get predicted value from candidate (surrogate's prediction)
                predicted = candidate.get(f"predicted_{rule.name}", objective)

                # Adapt rule based on HiFi result
                delta = rule.adapt(candidate, predicted)

                if abs(delta) > 1e-6:
                    adapted.append((rule.name, delta))

                    # Log adaptation to state store (and Weaviate)
                    self._state_store.log_adaptation(
                        rule_name=rule.name,
                        rule_category=rule.category.value,
                        limit_before=rule.limit - delta,
                        limit_after=rule.limit,
                        delta=delta,
                        direction="tighten" if delta < 0 else "relax",
                        trigger_margin=float(np.mean(rule._state.margin_history[-10:])),
                        n_observations=rule._state.n_observations,
                        run_id=run_id,
                    )

        # Persist all rule states
        states = {rule.name: rule.get_state() for rule in self._adaptive_rules}
        self._state_store.save_all_states(states)

        return AdaptationReport(adapted_rules=adapted, run_id=run_id)

    def get_adaptive_statistics(self) -> dict[str, Any]:
        """Get current adaptive rule statistics for dashboard display."""
        self._init_adaptive_rules()
        return self._state_store.get_statistics()

    @property
    def cem_version(self) -> str:
        """CEM version string for metadata."""
        return "adaptive-1.0.0" if hasattr(self, "_adaptive_rules") else "mock-0.1.0"


def get_cem_client(
    host: str | None = None,
    port: int | None = None,
    mock: bool = True,
    config: dict | None = None,
) -> CEMClient:
    """Factory function for CEM client."""
    return CEMClient(host, port, mock, config)
