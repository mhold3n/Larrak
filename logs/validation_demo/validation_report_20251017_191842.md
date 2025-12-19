
# CasADi Physics Validation Report

## Summary
- **Total Validations**: 100
- **Successful Validations**: 92 (92.0%)
- **Within Tolerance**: 0 (0.0%)
- **Collection Period**: 2025-10-17T19:18:42.685029 to 2025-10-17T19:18:42.750688
- **Duration**: 0.0 days

## Parity Analysis
### Torque Average Differences
- **Mean**: 5.35e-01
- **Std Dev**: 2.99e+00
- **Range**: [-6.94e+00, 6.42e+00]

### Torque Ripple Differences
- **Mean**: -1.48e-02
- **Std Dev**: 3.93e-01
- **Range**: [-8.62e-01, 8.03e-01]

### Side Load Penalty Differences
- **Mean**: -3.93e-02
- **Std Dev**: 7.67e-01
- **Range**: [-1.73e+00, 1.73e+00]

## Performance Analysis
### Evaluation Speedup
- **Mean**: 4.01x
- **Median**: 3.70x
- **Range**: [1.54x, 9.55x]

### Gradient Speedup
- **Mean**: 3.73x
- **Median**: 3.71x
- **Range**: [1.80x, 7.26x]

## Convergence Analysis
- **Python Convergence Rate**: 97.0%
- **CasADi Convergence Rate**: 95.0%

## Problem Type Breakdown
- **thermal_efficiency**: 36 problems, 0.0% success rate
- **litvin**: 44 problems, 0.0% success rate
- **crank_center**: 20 problems, 0.0% success rate

## Decision Criteria Analysis
- **Tolerance Threshold**: 1.00e-04
- **Success Rate**: 0.0%
- **Target Success Rate**: 95.0%

## Recommendation
❌ **DO NOT ENABLE**: CasADi physics does not meet validation criteria.
