# HSL Solver Guide

This guide provides comprehensive information about the HSL (Harwell Subroutine Library) linear solvers available in the Larrak optimization framework.

## Overview

The Larrak project includes 5 HSL linear solvers that significantly improve optimization performance compared to default linear solvers. These solvers are automatically selected based on problem characteristics and system capabilities.

## Available HSL Solvers

### MA27 - Classic Sparse Symmetric Solver

**Best for:** Small to medium problems
- **Problem size:** < 5,000 variables
- **Characteristics:** Robust, well-tested, reliable
- **Memory usage:** Low to moderate
- **Performance:** Good for small problems, may be slower for large ones

**When to use:**
- Small optimization problems
- When reliability is more important than speed
- As a fallback when other solvers fail
- Problems with < 5,000 variables

**Example usage:**
```python
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.linear_solver': 'ma27'})
```

### MA57 - Modern Sparse Symmetric Solver

**Best for:** Medium to large problems
- **Problem size:** 5,000-50,000 variables
- **Characteristics:** Better performance than MA27, more memory efficient
- **Memory usage:** Moderate
- **Performance:** Significantly faster than MA27 for medium problems

**When to use:**
- Medium to large optimization problems
- When you need better performance than MA27
- Problems with 5,000-50,000 variables
- General-purpose optimization

**Example usage:**
```python
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.linear_solver': 'ma57'})
```

### MA77 - Out-of-Core Solver

**Best for:** Very large problems with limited RAM
- **Problem size:** > 50,000 variables
- **Characteristics:** Uses disk for memory management, can handle very large problems
- **Memory usage:** Low (uses disk storage)
- **Performance:** Good for problems that don't fit in memory

**When to use:**
- Very large optimization problems
- When system has limited RAM
- Problems with > 50,000 variables
- When other solvers run out of memory

**Example usage:**
```python
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.linear_solver': 'ma77'})
```

### MA86 - Parallel Solver (CPU)

**Best for:** Large problems on multi-core systems
- **Problem size:** Large problems (10,000+ variables)
- **Characteristics:** Can use multiple CPU cores for parallel processing
- **Memory usage:** Moderate to high
- **Performance:** Excellent for large problems on multi-core systems

**When to use:**
- Large problems on multi-core systems
- When you have multiple CPU cores available
- Problems that benefit from parallel processing
- When MA57 is too slow but MA77 is overkill

**Example usage:**
```python
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.linear_solver': 'ma86'})
```

### MA97 - Advanced Parallel Solver

**Best for:** Very large problems on multi-core systems
- **Problem size:** Very large problems (50,000+ variables)
- **Characteristics:** Most modern HSL solver, advanced parallel processing
- **Memory usage:** High
- **Performance:** Best performance for very large problems

**When to use:**
- Very large problems on multi-core systems
- When you need the best possible performance
- Problems with > 50,000 variables
- When you have sufficient RAM and CPU cores

**Example usage:**
```python
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.linear_solver': 'ma97'})
```

## Automatic Solver Selection

The Larrak framework automatically selects the optimal solver based on:

1. **Problem size** (number of variables)
2. **Available system resources** (RAM, CPU cores)
3. **Solver availability** (which HSL solvers are installed)
4. **Historical performance** (if available)

### Selection Logic

```python
if problem_size < 5,000:
    # Prefer MA27
    solver = "ma27"
elif problem_size < 50,000:
    # Prefer MA57
    solver = "ma57"
else:
    # Prefer MA97 or MA77 for very large problems
    if multi_core_available:
        solver = "ma97"  # or "ma86"
    else:
        solver = "ma77"
```

## Performance Comparison

| Solver | Problem Size | Memory Usage | Speed | Parallel | Best Use Case |
|--------|-------------|--------------|-------|----------|---------------|
| MA27   | < 5,000     | Low          | Good  | No       | Small problems, reliability |
| MA57   | 5,000-50,000| Moderate     | Better| No       | Medium problems, general use |
| MA77   | > 50,000    | Low (disk)   | Good  | No       | Very large problems, limited RAM |
| MA86   | 10,000+     | Moderate-High| Excellent| Yes    | Large problems, multi-core |
| MA97   | 50,000+     | High         | Best  | Yes      | Very large problems, multi-core |

## Verification

To verify which HSL solvers are available on your system:

```bash
# Run the verification script
python scripts/verify_hsl_installation.py

# Or check programmatically
from campro.optimization.solver_selection import AdaptiveSolverSelector
selector = AdaptiveSolverSelector()
available = selector._get_available_solvers()
print(f"Available solvers: {[s.value for s in available]}")
```

## Troubleshooting

### Common Issues

1. **"Solver not available" error**
   - Ensure HSL binaries are in your PATH
   - Check that the solver is properly installed
   - Verify CasADi can access the HSL libraries

2. **Out of memory errors**
   - Try MA77 for out-of-core processing
   - Reduce problem size if possible
   - Check available system memory

3. **Slow performance**
   - Try a different solver (MA57 vs MA27)
   - Use parallel solvers (MA86/MA97) on multi-core systems
   - Check if the problem size matches solver recommendations

4. **Solver creation fails**
   - Verify all required DLLs are accessible
   - Check PATH environment variable
   - Ensure proper HSL installation

### Debug Information

Enable debug logging to see solver selection:

```python
import logging
logging.getLogger('campro.optimization.solver_selection').setLevel(logging.DEBUG)
```

## Installation

HSL solvers are included in the CoinHSL package. See the main installation guide for setup instructions.

## License

HSL solvers require a commercial license from STFC. The CoinHSL package includes these solvers for licensed users.

## References

- [HSL Documentation](https://www.hsl.rl.ac.uk/)
- [STFC Licensing Portal](https://licences.stfc.ac.uk/product/coin-hsl)
- [Ipopt Documentation](https://coin-or.github.io/Ipopt/)
- [CasADi Documentation](https://web.casadi.org/)







