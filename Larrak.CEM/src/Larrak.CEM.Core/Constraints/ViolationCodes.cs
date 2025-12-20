namespace Larrak.CEM.Core.Constraints;

/// <summary>
/// Structured violation codes for semantic failure reporting.
/// This enables automated recovery loops and meaningful dashboards.
/// </summary>
public enum ViolationCode
{
    // No violation
    None = 0,
    
    // === Thermodynamic violations (1xx) ===
    THERMO_MAX_PRESSURE = 100,
    THERMO_MAX_TEMPERATURE = 101,
    THERMO_MAX_CROWN_TEMP = 102,
    THERMO_LAMBDA_TOO_RICH = 103,
    THERMO_LAMBDA_TOO_LEAN = 104,
    THERMO_NEGATIVE_MASS = 105,
    THERMO_BACKFLOW_DETECTED = 106,
    THERMO_KNOCKING_RISK = 107,
    THERMO_COMBUSTION_UNSTABLE = 108,
    THERMO_HEAT_TRANSFER_LIMIT = 109,
    THERMO_MODEL_ASSUMPTION_BROKEN = 110,
    THERMO_FMEP_TOO_HIGH = 111,
    THERMO_EXHAUST_TEMP_LIMIT = 112,
    THERMO_BRAKE_EFFICIENCY_LOW = 113,
    
    // === Kinematic violations (2xx) ===
    KINEMATIC_MAX_JERK = 200,
    KINEMATIC_MAX_ACCELERATION = 201,
    KINEMATIC_PERIODICITY_BROKEN = 202,
    KINEMATIC_PHASE_MISMATCH = 203,
    KINEMATIC_VELOCITY_REVERSAL = 204,
    KINEMATIC_CONTACT_FORCE_LIMIT = 205,
    
    // === Gear geometry violations (3xx) ===
    GEAR_ENVELOPE_STROKE = 300,
    GEAR_MIN_RADIUS = 301,
    GEAR_MAX_RADIUS = 302,
    GEAR_CURVATURE_UNDERCUT = 303,
    GEAR_INTERFERENCE = 304,
    GEAR_CENTER_DISTANCE_INCONSISTENT = 305,
    GEAR_TOPOLOGY_INVALID = 306,
    GEAR_CONTACT_RATIO_LOW = 307,
    GEAR_TOOTH_THICKNESS_MIN = 308,
    GEAR_PROFILE_DEVIATION = 309,
    
    // === Manufacturing violations (4xx) ===
    AM_MIN_FEATURE_SIZE = 400,
    AM_MAX_OVERHANG_ANGLE = 401,
    AM_SUPPORT_REQUIRED = 402,
    AM_TOOL_ACCESS = 403,
    AM_SURFACE_FINISH = 404,
    AM_LAYER_ADHESION = 405,
    AM_THERMAL_DISTORTION = 406,
    AM_POWDER_REMOVAL = 407,
    MACHINING_UNDERCUT_IMPOSSIBLE = 410,
    MACHINING_TOOL_INTERFERENCE = 411,
    MACHINING_SURFACE_CURVATURE = 412,
    CASTING_WALL_THICKNESS = 420,
    CASTING_DRAFT_ANGLE = 421,
    
    // === Model/assumption violations (5xx) ===
    ASSUMPTION_REGIME_INVALID = 500,
    ASSUMPTION_CORRELATION_OUT_OF_RANGE = 501,
    ASSUMPTION_GRID_TOO_COARSE = 502,
    ASSUMPTION_STEADY_STATE_VIOLATED = 503,
    ASSUMPTION_QUASI_STATIC_VIOLATED = 504,
    
    // === Configuration violations (6xx) ===
    CONFIG_MISSING_PARAMETER = 600,
    CONFIG_INVALID_BOUNDS = 601,
    CONFIG_INCONSISTENT_UNITS = 602,
}

/// <summary>
/// Severity level for violations.
/// </summary>
public enum ViolationSeverity
{
    /// <summary>Informational - design is valid but could be improved.</summary>
    INFO = 0,
    
    /// <summary>Warning - design is marginal, may fail under uncertainty.</summary>
    WARN = 1,
    
    /// <summary>Error - design is invalid, must be corrected.</summary>
    ERROR = 2,
    
    /// <summary>Fatal - cannot proceed, fundamental issue.</summary>
    FATAL = 3
}

/// <summary>
/// Suggested corrective actions for automated recovery.
/// </summary>
public enum SuggestedActionCode
{
    NONE = 0,
    
    // Motion profile adjustments
    INCREASE_SMOOTHING = 100,
    REDUCE_STROKE = 101,
    ADJUST_PHASE = 102,
    RESAMPLE_GRID = 103,
    
    // Bound adjustments
    TIGHTEN_BOUNDS = 200,
    RELAX_BOUNDS = 201,
    EXPAND_ENVELOPE = 202,
    
    // Model changes
    CHANGE_FIDELITY_LEVEL = 300,
    USE_ALTERNATIVE_MODEL = 301,
    INCREASE_GRID_RESOLUTION = 302,
    
    // Optimizer hints
    IMPROVE_INITIAL_GUESS = 400,
    INCREASE_REGULARIZATION = 401,
    REDUCE_STEP_SIZE = 402,
    
    // Manual intervention
    MANUAL_REVIEW_REQUIRED = 900,
    CONTACT_DOMAIN_EXPERT = 901,
}

/// <summary>
/// Structured validation result with full diagnostic information.
/// This is the core "semantic infeasibility" type that replaces
/// "IPOPT failed" with actionable information.
/// </summary>
public readonly record struct ConstraintViolation
{
    /// <summary>Violation type code for programmatic handling.</summary>
    public required ViolationCode Code { get; init; }
    
    /// <summary>Severity level.</summary>
    public required ViolationSeverity Severity { get; init; }
    
    /// <summary>Human-readable description.</summary>
    public required string Message { get; init; }
    
    /// <summary>
    /// Distance to feasibility (positive = inside feasible, negative = violating).
    /// Allows automated tuning even for non-differentiable constraints.
    /// </summary>
    public double? Margin { get; init; }
    
    /// <summary>Names/IDs of decision variables affected.</summary>
    public string[]? AffectedVariables { get; init; }
    
    /// <summary>Suggested corrective action.</summary>
    public SuggestedActionCode SuggestedAction { get; init; }
    
    /// <summary>Additional structured metrics.</summary>
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }
    
    /// <summary>Create an error-level violation.</summary>
    public static ConstraintViolation Error(
        ViolationCode code,
        string message,
        double? margin = null,
        SuggestedActionCode action = SuggestedActionCode.NONE,
        string[]? affectedVars = null,
        Dictionary<string, double>? metrics = null) => new()
    {
        Code = code,
        Severity = ViolationSeverity.ERROR,
        Message = message,
        Margin = margin,
        SuggestedAction = action,
        AffectedVariables = affectedVars,
        Metrics = metrics
    };
    
    /// <summary>Create a warning-level violation.</summary>
    public static ConstraintViolation Warn(
        ViolationCode code,
        string message,
        double? margin = null,
        SuggestedActionCode action = SuggestedActionCode.NONE) => new()
    {
        Code = code,
        Severity = ViolationSeverity.WARN,
        Message = message,
        Margin = margin,
        SuggestedAction = action
    };
}

/// <summary>
/// Aggregated validation result containing all violations.
/// </summary>
public readonly record struct ValidationReport
{
    /// <summary>Overall validity (false if any ERROR or FATAL).</summary>
    public bool IsValid { get; init; }
    
    /// <summary>All violations found (may include warnings even if valid).</summary>
    public required IReadOnlyList<ConstraintViolation> Violations { get; init; }
    
    /// <summary>CEM runtime version for reproducibility.</summary>
    public required string CemVersion { get; init; }
    
    /// <summary>Configuration hash for reproducibility.</summary>
    public required string ConfigHash { get; init; }
    
    /// <summary>Timestamp of evaluation.</summary>
    public required DateTimeOffset Timestamp { get; init; }
    
    /// <summary>Highest severity in the report.</summary>
    public ViolationSeverity MaxSeverity => 
        Violations.Count == 0 
            ? ViolationSeverity.INFO 
            : Violations.Max(v => v.Severity);
    
    /// <summary>Get all violations of a specific code.</summary>
    public IEnumerable<ConstraintViolation> GetByCode(ViolationCode code) =>
        Violations.Where(v => v.Code == code);
    
    /// <summary>Create a valid (no violations) report.</summary>
    public static ValidationReport Valid(string cemVersion, string configHash) => new()
    {
        IsValid = true,
        Violations = Array.Empty<ConstraintViolation>(),
        CemVersion = cemVersion,
        ConfigHash = configHash,
        Timestamp = DateTimeOffset.UtcNow
    };
    
    /// <summary>Create a report with violations.</summary>
    public static ValidationReport WithViolations(
        IReadOnlyList<ConstraintViolation> violations,
        string cemVersion,
        string configHash) => new()
    {
        IsValid = !violations.Any(v => v.Severity >= ViolationSeverity.ERROR),
        Violations = violations,
        CemVersion = cemVersion,
        ConfigHash = configHash,
        Timestamp = DateTimeOffset.UtcNow
    };
}
