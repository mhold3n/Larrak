# Phase 1 Progress: Fix NaN in NLP Formulation

## Completed
1. ✅ Added diagnostic function `_diagnose_nan_in_jacobian()` to identify NaN locations
2. ✅ Integrated diagnostics into `_evaluate_nlp_at_x0()` to detect NaN during scaling
3. ✅ Enhanced IPOPT solver diagnostics to show constraint type and variable group
4. ✅ Added safeguards to `drho_dt` computation (protected numerator and denominator)
5. ✅ Added safeguards to `drho_log_dt` computation (protected division by rho_c_safe)
6. ✅ Added safeguards to `dT_dt` computation (protected numerator and denominator)
7. ✅ Added safeguards to `dyF_dt` computation (protected division by m_c_safe)

## Current Status
- **NaN Location**: Row 5, Col 20 (consistently detected)
- **Diagnostic Output**: Constraint type and variable group show as "unknown" (metadata not available)
- **NaN Still Present**: Despite safeguards, NaN persists in Jacobian evaluation

## Remaining Work
1. **Identify Root Cause**: Need to determine which specific constraint at row 5 produces NaN
2. **Fix Constraint Expression**: Apply targeted fix to the problematic constraint
3. **Verify Fix**: Test that NaN is eliminated and optimization can start

## Observations
- NaN is detected during CasADi's symbolic differentiation, not numerical evaluation
- Safeguards protect numerical values but may not prevent NaN during differentiation
- Row 5 is very early (likely first collocation residual), suggesting initialization issue
- Col 20 suggests it's an early variable (possibly initial state or first collocation point)

## Next Steps
1. Add constraint indexing to metadata to map row 5 to specific constraint
2. Add variable indexing to metadata to map col 20 to specific variable
3. Review constraint definitions around row 5 for potential NaN sources
4. Consider adding symbolic-level protections (e.g., using `ca.fmax` in all divisions)
