
# CasADi Physics Validation Report

## Summary
- **Total Validations**: 100
- **Successful Validations**: 86 (86.0%)
- **Within Tolerance**: 0 (0.0%)
- **Collection Period**: 2025-10-17T19:19:14.882320 to 2025-10-17T19:19:14.948646
- **Duration**: 0.0 days

## Parity Analysis
### Torque Average Differences
- **Mean**: 1.20e-03
- **Std Dev**: 1.43e-02
- **Range**: [-2.77e-02, 3.33e-02]

### Torque Ripple Differences
- **Mean**: 5.72e-05
- **Std Dev**: 3.68e-03
- **Range**: [-7.02e-03, 8.41e-03]

### Side Load Penalty Differences
- **Mean**: -1.25e-05
- **Std Dev**: 7.70e-03
- **Range**: [-1.61e-02, 1.67e-02]

## Performance Analysis
### Evaluation Speedup
- **Mean**: 3.54x
- **Median**: 3.37x
- **Range**: [1.48x, 8.03x]

### Gradient Speedup
- **Mean**: 3.56x
- **Median**: 3.34x
- **Range**: [1.65x, 7.32x]

## Convergence Analysis
- **Python Convergence Rate**: 91.0%
- **CasADi Convergence Rate**: 95.0%

## Problem Type Breakdown
- **crank_center**: 43 problems, 0.0% success rate
- **thermal_efficiency**: 27 problems, 0.0% success rate
- **litvin**: 30 problems, 0.0% success rate

## Decision Criteria Analysis
- **Tolerance Threshold**: 1.00e-04
- **Success Rate**: 0.0%
- **Target Success Rate**: 95.0%

## Recommendation
❌ **DO NOT ENABLE**: CasADi physics does not meet validation criteria.
