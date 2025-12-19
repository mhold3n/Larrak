
# CasADi Physics Validation Report

## Summary
- **Total Validations**: 100
- **Successful Validations**: 87 (87.0%)
- **Within Tolerance**: 97 (97.0%)
- **Collection Period**: 2025-10-17T19:19:42.822475 to 2025-10-17T19:19:42.897678
- **Duration**: 0.0 days

## Parity Analysis
### Torque Average Differences
- **Mean**: 3.05e-04
- **Std Dev**: 1.50e-02
- **Range**: [-3.27e-02, 3.32e-02]

### Torque Ripple Differences
- **Mean**: 3.04e-04
- **Std Dev**: 3.86e-03
- **Range**: [-7.95e-03, 8.45e-03]

### Side Load Penalty Differences
- **Mean**: -4.47e-04
- **Std Dev**: 7.40e-03
- **Range**: [-1.60e-02, 1.66e-02]

## Performance Analysis
### Evaluation Speedup
- **Mean**: 3.82x
- **Median**: 3.48x
- **Range**: [1.49x, 8.81x]

### Gradient Speedup
- **Mean**: 3.99x
- **Median**: 3.70x
- **Range**: [1.81x, 7.84x]

## Convergence Analysis
- **Python Convergence Rate**: 91.0%
- **CasADi Convergence Rate**: 96.0%

## Problem Type Breakdown
- **litvin**: 32 problems, 96.9% success rate
- **thermal_efficiency**: 45 problems, 97.8% success rate
- **crank_center**: 23 problems, 95.7% success rate

## Decision Criteria Analysis
- **Tolerance Threshold**: 1.00e-04
- **Success Rate**: 97.0%
- **Target Success Rate**: 95.0%

## Recommendation
✅ **RECOMMEND ENABLING**: CasADi physics meets all validation criteria.
