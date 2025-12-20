namespace Larrak.CEM.Core.Domains;

/// <summary>
/// Result of a validation check with optional failure reason.
/// </summary>
public readonly record struct ValidationResult
{
    /// <summary>Whether the validation passed.</summary>
    public bool IsValid { get; init; }
    
    /// <summary>Human-readable failure reason (null if valid).</summary>
    public string? Reason { get; init; }
    
    /// <summary>Suggested corrective action (null if valid or unknown).</summary>
    public string? SuggestedAction { get; init; }
    
    /// <summary>Create a passing validation result.</summary>
    public static ValidationResult Pass() => new() { IsValid = true };
    
    /// <summary>Create a failing validation result with reason.</summary>
    public static ValidationResult Fail(string reason, string? suggestedAction = null) => 
        new() { IsValid = false, Reason = reason, SuggestedAction = suggestedAction };
    
    /// <summary>Combine multiple results (fails if any fail).</summary>
    public static ValidationResult Combine(params ValidationResult[] results)
    {
        foreach (var result in results)
        {
            if (!result.IsValid)
                return result;
        }
        return Pass();
    }
}

/// <summary>
/// Engine geometry specification (fixed parameters).
/// </summary>
public readonly record struct EngineGeometry
{
    /// <summary>Cylinder bore diameter [m]</summary>
    public required double Bore { get; init; }
    
    /// <summary>Piston stroke [m]</summary>
    public required double Stroke { get; init; }
    
    /// <summary>Geometric compression ratio</summary>
    public required double CompressionRatio { get; init; }
    
    /// <summary>Connecting rod length [m]</summary>
    public double ConrodLength { get; init; }
    
    /// <summary>Displacement volume [m³]</summary>
    public double DisplacementVolume => Math.PI * Bore * Bore / 4 * Stroke;
    
    /// <summary>Clearance volume [m³]</summary>
    public double ClearanceVolume => DisplacementVolume / (CompressionRatio - 1);
    
    /// <summary>Piston area [m²]</summary>
    public double PistonArea => Math.PI * Bore * Bore / 4;
}

/// <summary>
/// Operating point for a single engine cycle evaluation.
/// </summary>
public readonly record struct OperatingPoint
{
    /// <summary>Engine speed [RPM]</summary>
    public required double Rpm { get; init; }
    
    /// <summary>Intake manifold pressure [bar, absolute]</summary>
    public required double BoostBar { get; init; }
    
    /// <summary>Fuel mass per cycle [mg]</summary>
    public required double FuelMg { get; init; }
    
    /// <summary>Angular velocity [rad/s]</summary>
    public double Omega => Rpm * 2 * Math.PI / 60;
    
    /// <summary>Intake pressure [Pa]</summary>
    public double IntakePressure => BoostBar * 1e5;
}
