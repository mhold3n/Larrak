namespace Larrak.CEM.Core.Domains;

/// <summary>
/// Kinematic state of the breathing gear mechanism at a point in the cycle.
/// </summary>
public readonly record struct KinematicState
{
    /// <summary>Ring gear angle (input/crankshaft equivalent) [rad]</summary>
    public required double Theta { get; init; }
    
    /// <summary>Planet gear angle [rad]</summary>
    public required double Psi { get; init; }
    
    /// <summary>Piston position from TDC [m]</summary>
    public required double X { get; init; }
    
    /// <summary>Piston velocity dx/dθ [m/rad]</summary>
    public required double V { get; init; }
    
    /// <summary>Piston acceleration d²x/dθ² [m/rad²]</summary>
    public required double A { get; init; }
    
    /// <summary>
    /// Instantaneous transmission ratio = dψ/dθ.
    /// For standard internal gear: i = Rr/Rp.
    /// </summary>
    public double TransmissionRatio { get; init; }
    
    /// <summary>
    /// Jerk (rate of acceleration change) d³x/dθ³ [m/rad³].
    /// Critical for NVH (Noise, Vibration, Harshness) assessment.
    /// </summary>
    public double Jerk { get; init; }
}

/// <summary>
/// Gear profile geometry at a specific angular position.
/// </summary>
public readonly record struct GearGeometry
{
    /// <summary>Angular position [rad]</summary>
    public required double Theta { get; init; }
    
    /// <summary>Planet gear pitch radius [mm]</summary>
    public required double Rp { get; init; }
    
    /// <summary>Ring gear pitch radius [mm]</summary>
    public required double Rr { get; init; }
    
    /// <summary>Center distance between gear axes [mm]</summary>
    public required double C { get; init; }
    
    /// <summary>Local profile curvature [1/mm]</summary>
    public double Curvature { get; init; }
    
    /// <summary>
    /// Contact stress indicator (Hertzian proxy).
    /// Higher values indicate risk of pitting/wear.
    /// </summary>
    public double ContactStress { get; init; }
    
    /// <summary>
    /// Check geometric consistency: C = Rr - Rp for internal gears.
    /// </summary>
    public ValidationResult ValidateConsistency()
    {
        var expectedC = Rr - Rp;
        var error = Math.Abs(C - expectedC);
        
        if (error > 0.1) // 0.1mm tolerance
            return ValidationResult.Fail($"Center distance C={C:F2} inconsistent with Rr-Rp={expectedC:F2}");
        
        return ValidationResult.Pass();
    }
}

/// <summary>
/// Physical and manufacturing limits for gear geometry.
/// </summary>
public readonly record struct GearLimits
{
    /// <summary>Minimum pitch radius [mm]</summary>
    public double MinRadius { get; init; }
    
    /// <summary>Maximum pitch radius [mm]</summary>
    public double MaxRadius { get; init; }
    
    /// <summary>Minimum tooth width for AM manufacturability [mm]</summary>
    public double MinToothWidth { get; init; }
    
    /// <summary>Maximum profile curvature before undercutting [1/mm]</summary>
    public double MaxCurvature { get; init; }
    
    /// <summary>Maximum jerk for NVH acceptability [mm/rad³]</summary>
    public double MaxJerk { get; init; }
    
    /// <summary>Minimum center distance [mm]</summary>
    public double MinCenterDistance { get; init; }
    
    /// <summary>
    /// Default limits based on typical gear constraints.
    /// </summary>
    public static GearLimits Default => new()
    {
        MinRadius = 20.0,           // 20mm minimum gear size
        MaxRadius = 80.0,           // 80mm maximum (packaging)
        MinToothWidth = 1.5,        // AM minimum feature size
        MaxCurvature = 0.5,         // Undercutting limit
        MaxJerk = 500.0,            // NVH limit
        MinCenterDistance = 10.0    // Mechanical clearance
    };
}
