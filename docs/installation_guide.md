# Installation Guide

This guide explains how to install Larrak with CasADi and IPOPT so the Thermal-Efficiency (TE) path runs reliably.

We strongly recommend using a CasADi build that includes the IPOPT solver plugin. On macOS (especially osx-64), many prebuilt CasADi packages do not ship the IPOPT plugin. If the plugin is missing, the TE path is unavailable.

## Quick Checks

Run these to verify IPOPT availability in CasADi:

```bash
python -c "import casadi as ca; print(getattr(ca,'nlpsol_plugins',lambda:[])())"   # should list 'ipopt'
python - <<'PY'
import casadi as ca
try:
    ca.nlpsol('probe','ipopt',{'x':ca.SX.sym('x'),'f':0,'g':ca.SX([])})
    print('ipopt_ok')
except Exception as e:
    print('no_ipopt', e)
PY
```

If 'ipopt' is not listed and the probe prints `no_ipopt`, install using one of the methods below.

## Method A: Conda (Linux or macOS with arm64)

Conda-forge provides CasADi + IPOPT plugin on Linux and often on macOS arm64 (Apple Silicon):

```bash
conda create -n larrak python=3.11 -c conda-forge
conda activate larrak
conda install -c conda-forge casadi ipopt
```

On Apple Silicon, if your base conda is x86_64, use arm64 subdir:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n larrak-te python=3.11
conda activate larrak-te
conda install -c conda-forge casadi ipopt
```

Re-run the quick checks to confirm IPOPT is available.

## Method B: Build CasADi from source with IPOPT (macOS Homebrew)

For macOS (osx-64) where prebuilt CasADi lacks IPOPT, build CasADi from source and enable IPOPT:

1. Install prerequisites with Homebrew:

```bash
brew update
brew install cmake eigen sundials ipopt pkg-config
```

2. Build and install CasADi into your active environment prefix:

```bash
git clone https://github.com/casadi/casadi.git
cd casadi && mkdir build && cd build
cmake \
  -DWITH_IPOPT=ON \
  -DWITH_SUNDIALS=ON \
  -DWITH_LAPACK=ON \
  -DWITH_BLAS=ON \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  ..
make -j
make install
```

3. Verify the plugin is available:

```bash
python -c "import casadi as ca; print(getattr(ca,'nlpsol_plugins',lambda:[])())"
python - <<'PY'
import casadi as ca
try:
    ca.nlpsol('probe','ipopt',{'x':ca.SX.sym('x'),'f':0,'g':ca.SX([])})
    print('ipopt_ok')
except Exception as e:
    print('no_ipopt', e)
PY
```

## Method C: Pip (may lack IPOPT)

Pip wheels often do not include IPOPT on macOS. Use only if A/B unavailable:

```bash
pip install 'casadi>=3.6,<3.7'
```

Then run the quick checks. If IPOPT is missing, use Method B.

## Project Setup

After ensuring CasADi+IPOPT works, set up the project:

```bash
python scripts/setup_environment.py
conda activate larrak
python scripts/check_environment.py
```

If TE is required and IPOPT is unavailable, the GUI will show a clear error message with a link to this guide. Follow Method B on macOS to enable IPOPT.

# Installation Guide

This guide provides comprehensive instructions for installing Larrak and its dependencies across different platforms.

## Quick Start (Recommended)

The fastest way to get started is using conda:

```bash
# Clone the repository
git clone <repository-url>
cd Larrak

# Create and activate environment
python scripts/setup_environment.py
conda activate larrak

# Verify installation
python scripts/check_environment.py
```

## Prerequisites

### Required Software

- **Python 3.9+**: Required for all installations
- **Conda or Miniconda**: Recommended for best compatibility
- **Git**: For cloning the repository

### System Requirements

- **Memory**: At least 4GB RAM (8GB+ recommended for large problems)
- **Storage**: At least 2GB free space for dependencies
- **CPU**: Any modern processor (optimization is CPU-intensive)

## Detailed Installation

### Method 1: Conda Installation (Recommended)

Conda installation provides the most reliable setup with ipopt solver support.

#### Step 1: Install Miniconda or Miniforge

**Linux:**
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or use Miniforge (conda-forge focused)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

**macOS:**
```bash
# Download and install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# For Apple Silicon (M1/M2) Macs
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**Windows:**
1. Download Miniconda installer from: https://docs.conda.io/en/latest/miniconda.html
2. Run the installer and follow the prompts
3. Make sure to check "Add conda to PATH" during installation

#### Step 2: Create Environment

```bash
# Navigate to project directory
cd Larrak

# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate larrak
```

#### Step 3: Verify Installation

```bash
# Run environment check
python scripts/check_environment.py

# Test basic functionality
python -c "import campro; print('Installation successful!')"
```

### Method 2: Pip Installation (Alternative)

Pip installation may not include ipopt solver support. Use only if conda is not available.

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/check_environment.py
```

**Note**: If ipopt is not available, you'll see warnings. Consider using conda instead.

### Method 3: Development Installation

For contributors and developers:

```bash
# Create development environment
conda env create -f environment-dev.yml
conda activate larrak-dev

# Install package in development mode
pip install -e .

# Install pre-commit hooks (optional)
pre-commit install
```

## Platform-Specific Notes

### Linux

**Ubuntu/Debian:**
```bash
# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install build-essential

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**CentOS/RHEL/Fedora:**
```bash
# Install system dependencies (if needed)
sudo yum groupinstall "Development Tools"
# or on newer versions:
sudo dnf groupinstall "Development Tools"

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### macOS

**Intel Macs:**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install conda via Homebrew (alternative)
brew install miniconda
```

**Apple Silicon (M1/M2) Macs:**
```bash
# Use the ARM64 version of Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### Windows

**Using Anaconda Prompt:**
1. Open Anaconda Prompt (installed with Miniconda)
2. Navigate to project directory: `cd C:\path\to\Larrak`
3. Create environment: `conda env create -f environment.yml`
4. Activate environment: `conda activate larrak`

**Using PowerShell:**
```powershell
# Navigate to project
cd C:\path\to\Larrak

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate larrak
```

## Troubleshooting

### Common Issues

#### 1. "ipopt solver is not available"

**Symptoms:**
- Warning message about ipopt not being available
- Optimization functions fail

**Solutions:**
```bash
# Reinstall with conda (recommended)
conda install -c conda-forge casadi ipopt

# Or recreate environment
conda env remove -n larrak
conda env create -f environment.yml
```

#### 2. "CasADi is not installed"

**Symptoms:**
- ImportError when importing casadi
- Environment check fails

**Solutions:**
```bash
# Install CasADi
conda install -c conda-forge casadi

# Or with pip (may not include ipopt)
pip install casadi
```

#### 3. "Python version not supported"

**Symptoms:**
- Error about Python version being too old
- Environment check fails

**Solutions:**
```bash
# Update Python in conda environment
conda install python=3.10

# Or create new environment with specific Python version
conda create -n larrak python=3.10
conda activate larrak
conda env update -f environment.yml
```

#### 4. "Permission denied" errors

**Symptoms:**
- Permission errors during installation
- Cannot write to conda directories

**Solutions:**
```bash
# Fix conda permissions
conda config --set always_yes true
conda clean --all

# Or install to user directory
conda install --user -c conda-forge casadi ipopt
```

#### 5. Slow conda operations

**Symptoms:**
- Very slow package installation
- Timeouts during environment creation

**Solutions:**
```bash
# Use mamba (faster conda alternative)
conda install -c conda-forge mamba
mamba env create -f environment.yml

# Or configure conda for faster downloads
conda config --set channel_priority strict
conda config --set show_channel_urls true
```

### Getting Help

If you encounter issues not covered here:

1. **Check the environment**: Run `python scripts/check_environment.py --verbose`
2. **Check logs**: Look for error messages in the terminal output
3. **Update conda**: `conda update conda`
4. **Clean environment**: `conda clean --all`
5. **Recreate environment**: Remove and recreate the conda environment

### Environment Variables

You can control validation behavior with environment variables:

```bash
# Skip import-time validation (for CI/testing)
export CAMPRO_SKIP_VALIDATION=1

# Enable verbose logging
export CAMPRO_LOG_LEVEL=DEBUG
```

## Verification

After installation, verify everything works:

```bash
# 1. Check environment
python scripts/check_environment.py

# 2. Test basic import
python -c "import campro; print('✓ campro imports successfully')"

# 3. Test CasADi and ipopt
python -c "
import casadi as ca
print(f'✓ CasADi version: {ca.__version__}')
print(f'✓ Available solvers: {ca.nlpsol_plugins()}')
if 'ipopt' in ca.nlpsol_plugins():
    print('✓ ipopt solver is available')
else:
    print('✗ ipopt solver is NOT available')
"

# 4. Run a simple test
python -c "
from CamPro_OptimalMotion import solve_minimum_jerk_motion
result = solve_minimum_jerk_motion(distance=10.0, time_horizon=5.0, max_velocity=5.0)
print('✓ Optimization test passed')
"
```

## Next Steps

Once installation is complete:

1. **Read the documentation**: Check `docs/` for detailed usage guides
2. **Run examples**: Try the scripts in `scripts/` directory
3. **Start with GUI**: Run `python cam_motion_gui.py` for interactive use
4. **Explore the API**: Look at `CamPro_OptimalMotion.py` for programmatic usage

## Uninstallation

To remove Larrak and its environment:

```bash
# Remove conda environment
conda env remove -n larrak

# Remove project directory
rm -rf /path/to/Larrak
```

## Optional HSL Solvers (MA27/MA57)

HSL (Harwell Subroutine Library) solvers are optional but significantly improve optimization performance for large problems. These are not included in GitHub clones due to licensing requirements.

### About HSL Solvers

- **MA27/MA57**: Sparse linear algebra solvers used by IPOPT
- **Performance**: Can provide 2-10x speedup for large optimization problems
- **Licensing**: Commercial license required from STFC
- **Optional**: IPOPT works without HSL but with reduced performance

### Obtaining HSL Solvers

1. **Visit STFC Licensing Portal**: https://licences.stfc.ac.uk/product/coin-hsl
2. **Request License**: Follow the licensing process for HSL software
3. **Download**: Obtain the CoinHSL package after license approval
4. **Install**: Follow the installation instructions provided with the license

### Installation with HSL

After obtaining the HSL license and package:

```bash
# Extract the HSL package to your project directory
# (This will create casadi/, coinhsl-archive-2023.11.17/, or ThirdParty-HSL/)

# Rebuild CasADi with HSL support (if needed)
# See Method B in this guide for building from source

# Verify HSL availability
python scripts/check_environment.py
```

### GitHub Clones and HSL

When you clone this repository from GitHub:

- Large dependency folders (`casadi/`, `coinhsl-archive-2023.11.17/`, `ThirdParty-HSL/`) are excluded
- The environment validator will show HSL solvers as "not available" with a warning
- This is normal and expected - HSL requires separate licensing
- All core functionality works without HSL

### Performance Impact

Without HSL solvers:
- Small to medium problems: Minimal impact
- Large problems (>1000 variables): 2-10x slower
- Memory usage: May be higher for large problems

With HSL solvers:
- All problem sizes: Optimal performance
- Memory usage: More efficient for large problems
- Recommended for production use with large optimization problems

## Support

For additional help:

- **Documentation**: Check `docs/` directory
- **Examples**: Look at `scripts/` directory
- **Issues**: Report problems on the project repository
- **Environment check**: Always run `python scripts/check_environment.py` first
- **HSL Licensing**: Contact STFC at https://licences.stfc.ac.uk/product/coin-hsl
