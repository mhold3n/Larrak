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

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, List
import numpy as np


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
    IDLE = 1       # Low RPM, low load
    CRUISE = 2     # Medium RPM, partial load  
    FULL_LOAD = 3  # High RPM/load, near limits


@dataclass
class ConstraintViolation:
    """Structured violation with semantic information."""
    code: ViolationCode
    severity: ViolationSeverity
    message: str
    margin: Optional[float] = None  # Distance to feasibility
    suggested_action: SuggestedActionCode = SuggestedActionCode.NONE
    affected_variables: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class ValidationReport:
    """Aggregated validation result with all violations."""
    is_valid: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    cem_version: str = "mock-0.1.0"
    config_hash: str = "default"
    regime_id: int = 0  # 0=unknown, 1=idle, 2=cruise, 3=full_load
    
    @property
    def max_severity(self) -> ViolationSeverity:
        if not self.violations:
            return ViolationSeverity.INFO
        return max(v.severity for v in self.violations)
    
    def get_by_code(self, code: ViolationCode) -> List[ConstraintViolation]:
        return [v for v in self.violations if v.code == code]
    
    def has_errors(self) -> bool:
        return any(v.severity >= ViolationSeverity.ERROR for v in self.violations)


@dataclass
class OperatingEnvelope:
    """Feasible bounds returned by CEM for Phase 1 NLP."""
    boost_range: tuple[float, float]
    fuel_range: tuple[float, float]
    motion_bounds: tuple[float, float]  # x_min, x_max [m]
    feasible: bool
    config_hash: str = "default"


@dataclass
class GearInitialGuess:
    """Physics-informed initialization for Phase 3 NLP."""
    Rp: np.ndarray
    Rr: np.ndarray
    C: np.ndarray
    phase_offset: float = 0.0
    mean_centerline: float = 0.0


class CEMClient:
    """
    Client for the Larrak CEM runtime.
    
    Usage:
        with CEMClient() as cem:
            # Pre-validate motion before expensive NLP
            report = cem.validate_motion(x_profile)
            if not report.is_valid:
                for v in report.violations:
                    print(f"[{v.code.name}] {v.message}")
                    print(f"  Suggested: {v.suggested_action.name}")
                    if v.margin:
                        print(f"  Margin: {v.margin:.2f}")
            
            # Get physics-informed initial guess
            guess = cem.get_gear_initial_guess(x_target)
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 50051,
        mock: bool = True,  # Default to mock until .NET is installed
        config: Optional[dict] = None
    ):
        self.address = f"{host}:{port}"
        self.mock = mock
        self._channel = None
        self._stub = None
        
        # Configuration (from config or defaults)
        self.config = config or {}
        self.max_jerk = self.config.get("max_jerk", 500.0)
        self.max_radius = self.config.get("max_radius", 80.0)
        self.min_radius = self.config.get("min_radius", 20.0)
        
        # Version tracking for reproducibility
        self.cem_version = "mock-0.1.0" if mock else self._get_cem_version()
        self.config_hash = self._compute_config_hash()
    
    def __enter__(self):
        if not self.mock:
            try:
                import grpc
                from truthmaker.cem import cem_pb2_grpc
                self._channel = grpc.insecure_channel(self.address)
                self._stub = cem_pb2_grpc.CEMServiceStub(self._channel)
                
                # Verify connection by getting version
                try:
                    from truthmaker.cem import cem_pb2
                    response = self._stub.GetVersion(cem_pb2.VersionRequest())
                    self.cem_version = response.cem_version
                    print(f"[CEM] Connected to CEM service v{self.cem_version}")
                except Exception as e:
                    print(f"[CEM] Failed to connect: {e}, falling back to mock mode")
                    self.mock = True
                    self._channel.close()
                    self._channel = None
                    self._stub = None
            except ImportError as e:
                print(f"[CEM] gRPC not available ({e}), falling back to mock mode")
                self.mock = True
        return self
    
    def __exit__(self, *args):
        if self._channel:
            self._channel.close()
    
    def validate_motion(
        self, 
        x_profile: np.ndarray, 
        theta: Optional[np.ndarray] = None
    ) -> ValidationReport:
        """
        Validate a target motion profile against CEM constraints.
        
        This should be called BEFORE Phase 3 NLP to avoid wasted computation.
        Returns structured violations with margins and suggested actions.
        """
        n = len(x_profile)
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, n)
        
        if self.mock:
            return self._validate_motion_mock(x_profile, theta)
        else:
            return self._validate_motion_grpc(x_profile, theta)
    
    def _validate_motion_mock(
        self, 
        x_profile: np.ndarray, 
        theta: np.ndarray
    ) -> ValidationReport:
        """Mock implementation of motion validation."""
        violations = []
        
        # Calculate derivatives with periodic boundary handling
        v = np.gradient(x_profile, theta, edge_order=2)
        a = np.gradient(v, theta, edge_order=2)
        jerk = np.gradient(a, theta, edge_order=2)
        
        max_jerk = np.max(np.abs(jerk))
        jerk_margin = self.max_jerk - max_jerk
        
        # Check jerk (NVH constraint)
        if max_jerk > self.max_jerk:
            violations.append(ConstraintViolation(
                code=ViolationCode.KINEMATIC_MAX_JERK,
                severity=ViolationSeverity.ERROR,
                message=f"Jerk {max_jerk:.1f} mm/rad³ exceeds NVH limit {self.max_jerk:.1f}",
                margin=jerk_margin,
                suggested_action=SuggestedActionCode.INCREASE_SMOOTHING,
                affected_variables=["x_profile"],
                metrics={"max_jerk": max_jerk, "limit": self.max_jerk, "margin": jerk_margin}
            ))
        elif jerk_margin < self.max_jerk * 0.1:
            violations.append(ConstraintViolation(
                code=ViolationCode.KINEMATIC_MAX_JERK,
                severity=ViolationSeverity.WARN,
                message=f"Jerk {max_jerk:.1f} mm/rad³ is within 10% of limit",
                margin=jerk_margin
            ))
        
        # Check stroke vs gear envelope
        stroke = np.max(x_profile) - np.min(x_profile)
        min_rp_required = stroke / 2
        envelope_margin = self.max_radius - min_rp_required
        
        if min_rp_required > self.max_radius:
            violations.append(ConstraintViolation(
                code=ViolationCode.GEAR_ENVELOPE_STROKE,
                severity=ViolationSeverity.ERROR,
                message=f"Stroke {stroke:.1f} mm requires Rp > {min_rp_required:.1f} mm (max: {self.max_radius})",
                margin=envelope_margin,
                suggested_action=SuggestedActionCode.REDUCE_STROKE,
                affected_variables=["stroke", "Rp"],
                metrics={"stroke": stroke, "min_rp_required": min_rp_required, "margin": envelope_margin}
            ))
        
        # Check periodicity
        periodicity_error = abs(x_profile[0] - x_profile[-1])
        if periodicity_error > 0.1:  # 0.1mm tolerance
            violations.append(ConstraintViolation(
                code=ViolationCode.KINEMATIC_PERIODICITY_BROKEN,
                severity=ViolationSeverity.WARN,
                message=f"Motion not periodic: |x[0] - x[end]| = {periodicity_error:.3f} mm",
                margin=-periodicity_error,
                suggested_action=SuggestedActionCode.ADJUST_PHASE
            ))
        
        is_valid = not any(v.severity >= ViolationSeverity.ERROR for v in violations)
        
        return ValidationReport(
            is_valid=is_valid,
            violations=violations,
            cem_version=self.cem_version,
            config_hash=self.config_hash
        )
    
    def _validate_motion_grpc(
        self, 
        x_profile: np.ndarray, 
        theta: np.ndarray
    ) -> ValidationReport:
        """Live gRPC implementation of motion validation."""
        from truthmaker.cem import cem_pb2
        
        # Build request
        request = cem_pb2.MotionValidationRequest()
        request.x_profile_mm.extend(x_profile.tolist())
        request.theta_rad.extend(theta.tolist())
        request.max_jerk = self.max_jerk
        request.max_radius = self.max_radius
        
        # Call service
        response = self._stub.ValidateMotion(request)
        
        # Convert proto to Python dataclasses
        violations = []
        for v in response.report.violations:
            violations.append(ConstraintViolation(
                code=ViolationCode(v.code),
                severity=ViolationSeverity(v.severity),
                message=v.message,
                margin=v.margin if v.HasField('margin') else None,
                suggested_action=SuggestedActionCode(v.suggested_action),
                affected_variables=list(v.affected_variables) if v.affected_variables else None,
                metrics=dict(v.metrics) if v.metrics else None
            ))
        
        return ValidationReport(
            is_valid=response.report.is_valid,
            violations=violations,
            cem_version=response.report.cem_version,
            config_hash=response.report.config_hash
        )
    
    def get_thermo_envelope(
        self, 
        bore: float, 
        stroke: float, 
        cr: float, 
        rpm: float
    ) -> OperatingEnvelope:
        """
        Query CEM for feasible thermodynamic operating envelope.
        
        Returns conservative bounds that Phase 1 NLP should respect.
        These bounds come from configuration, not hardcoded constants.
        """
        if self.mock:
            return self._get_thermo_envelope_mock(bore, stroke, cr, rpm)
        else:
            return self._get_thermo_envelope_grpc(bore, stroke, cr, rpm)
    
    def _get_thermo_envelope_mock(
        self, 
        bore: float, 
        stroke: float, 
        cr: float, 
        rpm: float
    ) -> OperatingEnvelope:
        """Mock implementation of envelope generation."""
        # These should come from config, not hardcoded
        boost_min = self.config.get("boost_min", 0.5)
        boost_max = self.config.get("boost_max", 6.0)  # Conservative vs 8.0
        
        # Lambda-based fuel limits
        v_disp = np.pi * bore**2 / 4 * stroke
        T_int = 300  # Base intake temp
        
        # Conservative fuel range based on lambda limits
        lambda_min = self.config.get("lambda_min", 0.7)
        lambda_max = self.config.get("lambda_max", 1.5)
        afr_stoich = 14.7
        
        rho_min = boost_min * 1e5 / (287 * T_int)
        rho_max = boost_max * 1e5 / (287 * T_int)
        
        m_air_min = rho_min * v_disp * 0.85  # Volumetric efficiency
        m_air_max = rho_max * v_disp * 0.95
        
        fuel_min = m_air_min * 1e6 / (afr_stoich * lambda_max)  # mg
        fuel_max = m_air_max * 1e6 / (afr_stoich * lambda_min)  # mg
        
        return OperatingEnvelope(
            boost_range=(boost_min, boost_max),
            fuel_range=(max(1.0, fuel_min), min(500.0, fuel_max)),
            motion_bounds=(0.0, stroke),
            feasible=True,
            config_hash=self.config_hash
        )
    
    def _get_thermo_envelope_grpc(
        self,
        bore: float,
        stroke: float,
        cr: float,
        rpm: float
    ) -> OperatingEnvelope:
        """Live gRPC implementation of envelope generation."""
        from truthmaker.cem import cem_pb2
        
        # Build request
        geometry = cem_pb2.EngineGeometry(
            bore_m=bore,
            stroke_m=stroke,
            compression_ratio=cr
        )
        request = cem_pb2.ThermoEnvelopeRequest(
            geometry=geometry,
            rpm=rpm
        )
        
        # Call service
        response = self._stub.GetThermoEnvelope(request)
        
        return OperatingEnvelope(
            boost_range=(response.boost_min, response.boost_max),
            fuel_range=(response.fuel_min_mg, response.fuel_max_mg),
            motion_bounds=(response.motion_min_m, response.motion_max_m),
            feasible=response.feasible,
            config_hash=response.config_hash
        )
    
    def get_gear_initial_guess(
        self, 
        x_target: np.ndarray,
        theta: Optional[np.ndarray] = None
    ) -> GearInitialGuess:
        """
        Get physics-informed initial guess from CEM.
        
        Much better than naive constant initialization.
        Includes phase alignment and centerline calculation.
        """
        if self.mock:
            return self._get_gear_initial_guess_mock(x_target, theta)
        else:
            return self._get_gear_initial_guess_grpc(x_target, theta)
    
    def _get_gear_initial_guess_mock(
        self, 
        x_target: np.ndarray,
        theta: Optional[np.ndarray] = None
    ) -> GearInitialGuess:
        """Mock implementation of initial guess generation."""
        n = len(x_target)
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, n)
        
        stroke = np.max(x_target) - np.min(x_target)
        x_min = np.min(x_target)
        x_mean = np.mean(x_target)
        
        # Initial guess: Add margin to minimum required
        rp_mean = stroke / 2 + 10  # 10mm margin
        c_mean = x_min + rp_mean
        
        # Find phase: where is x at minimum?
        phase_offset = theta[np.argmin(x_target)]
        
        # Start with constant profiles
        Rp_init = np.full(n, rp_mean)
        C_init = np.full(n, c_mean)
        Rr_init = C_init + Rp_init  # Conjugacy: C = Rr - Rp for internal
        
        return GearInitialGuess(
            Rp=Rp_init,
            Rr=Rr_init,
            C=C_init,
            phase_offset=phase_offset,
            mean_centerline=x_mean
        )
    
    def _get_gear_initial_guess_grpc(
        self, 
        x_target: np.ndarray,
        theta: Optional[np.ndarray] = None
    ) -> GearInitialGuess:
        """Live gRPC implementation of initial guess generation."""
        from truthmaker.cem import cem_pb2
        
        n = len(x_target)
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, n)
        
        # Build request
        request = cem_pb2.GearInitialGuessRequest()
        request.x_target_mm.extend(x_target.tolist())
        request.theta_rad.extend(theta.tolist())
        
        # Call service
        response = self._stub.GetGearInitialGuess(request)
        
        return GearInitialGuess(
            Rp=np.array(list(response.rp)),
            Rr=np.array(list(response.rr)),
            C=np.array(list(response.c)),
            phase_offset=response.phase_offset_rad,
            mean_centerline=response.mean_centerline_mm
        )
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for reproducibility."""
        import hashlib
        config_str = str(sorted(self.config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_cem_version(self) -> str:
        """Query CEM service for version."""
        if self._stub:
            try:
                from truthmaker.cem import cem_pb2
                response = self._stub.GetVersion(cem_pb2.VersionRequest())
                return response.cem_version
            except Exception:
                pass
        return "cem-0.1.0"
    
    def extract_training_metadata(self, report: ValidationReport) -> Dict[str, any]:
        """
        Extract margins, constraint codes from ValidationReport for training logs.
        
        Returns dict with:
            - margins: {violation_code_name: margin_value}
            - constraint_codes: [violation_code_values]
            - is_valid: bool
            - regime_id: int
            - cem_version: str
        """
        return {
            'margins': {
                v.code.name: v.margin 
                for v in report.violations 
                if v.margin is not None
            },
            'constraint_codes': [v.code.value for v in report.violations],
            'is_valid': report.is_valid,
            'regime_id': report.regime_id,
            'cem_version': report.cem_version,
        }
    
    @staticmethod
    def classify_regime(
        rpm: float, 
        boost_bar: float,
        rpm_idle: float = 800.0,
        rpm_full: float = 4000.0,
        boost_full: float = 3.0
    ) -> OperatingRegime:
        """
        Classify operating regime based on RPM and boost pressure.
        
        Args:
            rpm: Engine speed [RPM]
            boost_bar: Intake manifold pressure [bar]
            rpm_idle: Threshold below which engine is in IDLE regime
            rpm_full: Threshold above which engine approaches FULL_LOAD
            boost_full: Boost pressure threshold for FULL_LOAD
            
        Returns:
            OperatingRegime enum value
        """
        if rpm < rpm_idle:
            return OperatingRegime.IDLE
        elif rpm > rpm_full or boost_bar > boost_full:
            return OperatingRegime.FULL_LOAD
        else:
            return OperatingRegime.CRUISE


def get_cem_client(
    host: str = "localhost", 
    port: int = 50051,
    mock: bool = True,
    config: Optional[dict] = None
) -> CEMClient:
    """Factory function for CEM client."""
    return CEMClient(host, port, mock, config)
