#!/usr/bin/env python3
"""Run validation tests and save results to file."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scripts.phase1_combustion_validation import main
    import io
    from contextlib import redirect_stdout
    
    print("Running Phase 1 Combustion Validation Tests...")
    print("=" * 60)
    
    # Capture output
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            main()
            test_passed = True
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            test_passed = False
    
    output = f.getvalue()
    print(output)
    
    # Save to file
    with open("validation_test_results.txt", "w") as outfile:
        outfile.write("Phase 1 Combustion Validation Test Results\n")
        outfile.write("=" * 60 + "\n\n")
        outfile.write(output)
    
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ Tests completed - results saved to validation_test_results.txt")
    else:
        print("✗ Tests failed - check validation_test_results.txt for details")
        sys.exit(1)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTrying alternative import method...")
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/phase1_combustion_validation.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)








