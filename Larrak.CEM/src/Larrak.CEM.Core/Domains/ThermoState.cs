namespace Larrak.CEM.Core.Domains;

/// <summary>
/// Thermodynamic state at a point in the engine cycle.
/// Immutable record for thread-safe constraint evaluation.
/// </summary>
public readonly record struct ThermoState
{
    /// <summary>Crank angle [rad]</summary>
    public required double Theta { get; init; }
    
    /// <summary>Cylinder pressure [Pa]</summary>
    public required double Pressure { get; init; }
    
    /// <summary>Gas temperature [K]</summary>
    public required double Temperature { get; init; }
    
    /// <summary>Cylinder mass [kg]</summary>
    public required double Mass { get; init; }
    
    /// <summary>Cylinder volume [m³]</summary>
    public required double Volume { get; init; }
    
    /// <summary>
    /// Ratio of specific heats (temperature-dependent approximation).
    /// Decreases at high temperatures due to dissociation.
    /// </summary>
    public double Gamma => Temperature < 1500 ? 1.35 : 1.28;
    
    /// <summary>
    /// Density from ideal gas law [kg/m³].
    /// </summary>
    public double Density => Mass / Volume;
    
    /// <summary>
    /// Validate state against physical limits.
    /// </summary>
    public ValidationResult Validate(ThermoLimits limits)
    {
        if (Pressure > limits.MaxPressure)
            return ValidationResult.Fail($"Pressure {Pressure / 1e5:F0} bar > {limits.MaxPressure / 1e5:F0} bar limit");
        
        if (Temperature > limits.MaxTemperature)
            return ValidationResult.Fail($"Temperature {Temperature:F0} K > {limits.MaxTemperature:F0} K limit");
        
        if (Pressure < 0)
            return ValidationResult.Fail("Negative pressure is non-physical");
        
        if (Temperature < 0)
            return ValidationResult.Fail("Negative temperature is non-physical");
        
        return ValidationResult.Pass();
    }
}

/// <summary>
/// Physical limits for thermodynamic feasibility.
/// These form the "guardrails" that the optimizer must respect.
/// </summary>
public readonly record struct ThermoLimits
{
    /// <summary>Maximum allowable cylinder pressure [Pa]</summary>
    public double MaxPressure { get; init; }
    
    /// <summary>Maximum gas temperature [K]</summary>
    public double MaxTemperature { get; init; }
    
    /// <summary>Maximum piston crown temperature [K] (material limit)</summary>
    public double MaxCrownTemp { get; init; }
    
    /// <summary>Minimum air-fuel equivalence ratio (rich limit)</summary>
    public double MinLambda { get; init; }
    
    /// <summary>Maximum air-fuel equivalence ratio (lean limit)</summary>
    public double MaxLambda { get; init; }
    
    /// <summary>
    /// Default limits based on typical diesel engine constraints.
    /// </summary>
    public static ThermoLimits Default => new()
    {
        MaxPressure = 250e5,      // 250 bar
        MaxTemperature = 2500,    // 2500 K
        MaxCrownTemp = 550,       // 550 K (aluminum limit)
        MinLambda = 0.7,          // Rich limit
        MaxLambda = 1.5           // Lean limit
    };
}
