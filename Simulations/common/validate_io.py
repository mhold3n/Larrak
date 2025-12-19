"""
Script to validate simulation input/output files against the schema.
Usage: python validate_io.py <file.json> --type [input|output]
"""

import argparse
import json
import sys
from pathlib import Path
from .io_schema import SimulationInput, SimulationOutput

def validate_file(filepath: str, schema_type: str):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        if schema_type == 'input':
            model = SimulationInput(**data)
            print(f"✅ Valid Input: {model.run_id}")
        elif schema_type == 'output':
            model = SimulationOutput(**data)
            print(f"✅ Valid Output: {model.run_id} (Success={model.success})")
        else:
            print("Unknown schema type. Use 'input' or 'output'")
            
    except Exception as e:
        print(f"❌ Validation Failed:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JSON against Larrak Phase 4 Schema")
    parser.add_argument("file", help="Path to JSON file")
    parser.add_argument("--type", choices=['input', 'output'], required=True, help="Type of file")
    args = parser.parse_args()
    
    validate_file(args.file, args.type)
