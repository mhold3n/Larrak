namespace Larrak.CEM.Core.Constraints;

using Larrak.CEM.Core.Domains;

/// <summary>
/// Interface for composable constraints.
/// Constraints can be enabled/disabled via configuration.
/// </summary>
/// <typeparam name="TState">The state type this constraint validates.</typeparam>
public interface IConstraint<in TState>
{
    /// <summary>Unique identifier for this constraint.</summary>
    string Id { get; }
    
    /// <summary>Human-readable name.</summary>
    string Name { get; }
    
    /// <summary>Violation code produced when this constraint fails.</summary>
    ViolationCode ViolationCode { get; }
    
    /// <summary>Whether this constraint is currently enabled.</summary>
    bool IsEnabled { get; }
    
    /// <summary>
    /// Evaluate the constraint.
    /// Returns null if satisfied, ConstraintViolation if violated.
    /// </summary>
    ConstraintViolation? Evaluate(TState state);
}

/// <summary>
/// Registry of all constraints, allowing configuration-based enable/disable.
/// </summary>
public class ConstraintRegistry
{
    private readonly Dictionary<string, object> _constraints = new();
    private readonly HashSet<string> _disabledIds = new();
    
    /// <summary>Register a constraint.</summary>
    public void Register<TState>(IConstraint<TState> constraint)
    {
        _constraints[constraint.Id] = constraint;
    }
    
    /// <summary>Disable a constraint by ID.</summary>
    public void Disable(string constraintId)
    {
        _disabledIds.Add(constraintId);
    }
    
    /// <summary>Enable a previously disabled constraint.</summary>
    public void Enable(string constraintId)
    {
        _disabledIds.Remove(constraintId);
    }
    
    /// <summary>Get all constraints for a state type.</summary>
    public IEnumerable<IConstraint<TState>> GetConstraints<TState>()
    {
        return _constraints.Values
            .OfType<IConstraint<TState>>()
            .Where(c => !_disabledIds.Contains(c.Id));
    }
    
    /// <summary>Evaluate all constraints for a state.</summary>
    public List<ConstraintViolation> EvaluateAll<TState>(TState state)
    {
        var violations = new List<ConstraintViolation>();
        
        foreach (var constraint in GetConstraints<TState>())
        {
            var violation = constraint.Evaluate(state);
            if (violation.HasValue)
            {
                violations.Add(violation.Value);
            }
        }
        
        return violations;
    }
}

// === Concrete constraint implementations ===

/// <summary>
/// Maximum jerk constraint for NVH acceptability.
/// </summary>
public class MaxJerkConstraint : IConstraint<MotionProfile>
{
    private readonly double _maxJerk;
    
    public MaxJerkConstraint(double maxJerk = 500.0)
    {
        _maxJerk = maxJerk;
    }
    
    public string Id => "kinematic.max_jerk";
    public string Name => "Maximum Jerk (NVH)";
    public ViolationCode ViolationCode => ViolationCode.KINEMATIC_MAX_JERK;
    public bool IsEnabled => true;
    
    public ConstraintViolation? Evaluate(MotionProfile profile)
    {
        var maxJerk = profile.MaxAbsoluteJerk;
        var margin = _maxJerk - maxJerk;
        
        if (maxJerk > _maxJerk)
        {
            return ConstraintViolation.Error(
                ViolationCode,
                $"Jerk {maxJerk:F1} mm/rad³ exceeds NVH limit {_maxJerk:F1}",
                margin: margin,
                action: SuggestedActionCode.INCREASE_SMOOTHING,
                metrics: new Dictionary<string, double>
                {
                    ["max_jerk"] = maxJerk,
                    ["limit"] = _maxJerk,
                    ["margin"] = margin
                }
            );
        }
        
        // Warning if close to limit (within 10%)
        if (margin < _maxJerk * 0.1)
        {
            return ConstraintViolation.Warn(
                ViolationCode,
                $"Jerk {maxJerk:F1} mm/rad³ is within 10% of limit",
                margin: margin
            );
        }
        
        return null;
    }
}

/// <summary>
/// Stroke envelope constraint for gear feasibility.
/// </summary>
public class StrokeEnvelopeConstraint : IConstraint<MotionProfile>
{
    private readonly double _maxRadius;
    
    public StrokeEnvelopeConstraint(double maxRadius = 80.0)
    {
        _maxRadius = maxRadius;
    }
    
    public string Id => "gear.stroke_envelope";
    public string Name => "Stroke vs Gear Envelope";
    public ViolationCode ViolationCode => ViolationCode.GEAR_ENVELOPE_STROKE;
    public bool IsEnabled => true;
    
    public ConstraintViolation? Evaluate(MotionProfile profile)
    {
        var stroke = profile.Stroke;
        var minRpRequired = stroke / 2;
        var margin = _maxRadius - minRpRequired;
        
        if (minRpRequired > _maxRadius)
        {
            return ConstraintViolation.Error(
                ViolationCode,
                $"Stroke {stroke:F1} mm requires Rp > {minRpRequired:F1} mm (max: {_maxRadius})",
                margin: margin,
                action: SuggestedActionCode.REDUCE_STROKE,
                affectedVars: new[] { "stroke", "Rp" },
                metrics: new Dictionary<string, double>
                {
                    ["stroke"] = stroke,
                    ["min_rp_required"] = minRpRequired,
                    ["max_radius"] = _maxRadius,
                    ["margin"] = margin
                }
            );
        }
        
        return null;
    }
}

/// <summary>
/// Motion profile data structure for validation.
/// Uses explicit units in property names to avoid confusion.
/// </summary>
public readonly record struct MotionProfile
{
    /// <summary>Piston position array [mm]</summary>
    public required double[] X_mm { get; init; }
    
    /// <summary>Crank angle array [rad]</summary>
    public required double[] Theta_rad { get; init; }
    
    /// <summary>Stroke (max - min) [mm]</summary>
    public double Stroke => X_mm.Max() - X_mm.Min();
    
    /// <summary>Velocity array [mm/rad]</summary>
    public double[] Velocity_mmPerRad => Differentiate(X_mm, Theta_rad);
    
    /// <summary>Acceleration array [mm/rad²]</summary>
    public double[] Acceleration_mmPerRad2 => Differentiate(Velocity_mmPerRad, Theta_rad);
    
    /// <summary>Jerk array [mm/rad³]</summary>
    public double[] Jerk_mmPerRad3 => Differentiate(Acceleration_mmPerRad2, Theta_rad);
    
    /// <summary>Maximum absolute jerk [mm/rad³]</summary>
    public double MaxAbsoluteJerk => Jerk_mmPerRad3.Select(Math.Abs).Max();
    
    /// <summary>Check periodicity: |x[0] - x[end]|</summary>
    public double PeriodicityError => Math.Abs(X_mm[0] - X_mm[^1]);
    
    /// <summary>
    /// Numerical differentiation with periodic boundary handling.
    /// Uses central differences with Savitzky-Golay-like smoothing.
    /// </summary>
    private static double[] Differentiate(double[] y, double[] x)
    {
        var n = y.Length;
        var dy = new double[n];
        
        // Central differences for interior points
        for (int i = 1; i < n - 1; i++)
        {
            dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
        }
        
        // Periodic boundary: use wrap-around
        var dx = x[1] - x[0]; // Assume uniform grid
        dy[0] = (y[1] - y[n - 2]) / (2 * dx);  // y[n-1] ≈ y[0], so use y[n-2]
        dy[n - 1] = dy[0]; // Periodic
        
        return dy;
    }
}
