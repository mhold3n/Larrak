# Larrak CEM - Computational Engineering Model

Cross-platform C# runtime providing constraint validation, envelope generation, and physics-informed initialization for the Python optimizer.

## Quick Start

```bash
# Check prerequisites
python scripts/setup_cem.py check

# Test client (mock mode - works without .NET)
python scripts/setup_cem.py test

# Build CEM (requires .NET SDK)
python scripts/setup_cem.py build
```

## Install .NET 8 SDK

### Windows

```powershell
# Option 1: winget (recommended)
winget install Microsoft.DotNet.SDK.8

# Option 2: Download installer
# https://dotnet.microsoft.com/download/dotnet/8.0
```

### macOS

```bash
# Option 1: Homebrew
brew install dotnet@8

# Option 2: Download installer
# https://dotnet.microsoft.com/download/dotnet/8.0
```

### Linux (Ubuntu/Debian)

```bash
# Add Microsoft package repository
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

# Install SDK
sudo apt-get update && sudo apt-get install -y dotnet-sdk-8.0
```

### Verify Installation

```bash
dotnet --version
# Should output: 8.0.x
```

## Usage

### Mock Mode (No .NET Required)

```python
from larrak.cem_client import CEMClient

with CEMClient(mock=True) as cem:
    report = cem.validate_motion(x_profile)
    if not report.is_valid:
        for v in report.violations:
            print(f"[{v.code.name}] {v.message}")
```

### gRPC Mode (Requires .NET)

```python
from larrak.cem_client import cem_runtime, CEMClient

if cem_runtime.is_built:
    with cem_runtime.start_service() as port:
        with CEMClient(port=port, mock=False) as cem:
            report = cem.validate_motion(x_profile)
```

## Architecture

```
Larrak.CEM/
├── protos/
│   └── cem.proto           # gRPC service definition
├── src/
│   ├── Larrak.CEM.Core/    # Domain types, constraint taxonomy
│   ├── Larrak.CEM.Engine/  # Constraint evaluation engine
│   └── Larrak.CEM.API/     # gRPC service implementation
└── README.md
```

## Build & Run

```bash
cd Larrak.CEM
dotnet restore
dotnet build -c Release

# Run service
dotnet run --project src/Larrak.CEM.API
# Server starts on localhost:50051
```
