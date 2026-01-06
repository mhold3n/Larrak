#!/usr/bin/env bash
# Build CoinHSL library from source and package as Linux binary directory
# This is a one-time build script that creates a binary package matching
# the existing macOS/Windows binary structure.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COINHSL_SOURCE="$PROJECT_ROOT/libraries/coinhsl-2024.05.15"
STAGING_DIR="/tmp/coinhsl-install-$$"

# Cleanup function
cleanup() {
    if [ -d "$STAGING_DIR" ]; then
        echo -e "${YELLOW}Cleaning up staging directory...${NC}"
        rm -rf "$STAGING_DIR"
    fi
}
trap cleanup EXIT

# Check if we're on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo -e "${RED}Error: This script is for Linux only.${NC}"
    exit 1
fi

# Detect platform
ARCH=$(uname -m)
OS="linux-gnu"
VERSION="2024.5.15"
TARGET_DIR="$PROJECT_ROOT/libraries/CoinHSL.v${VERSION}.${ARCH}-${OS}-libgfortran5"

echo -e "${GREEN}Building CoinHSL for Linux${NC}"
echo "Target directory: $TARGET_DIR"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check meson
if ! command -v meson &> /dev/null; then
    echo -e "${RED}Error: meson not found. Install with: pip install meson${NC}"
    exit 1
fi
echo "✓ meson: $(meson --version)"

# Check gfortran
if ! command -v gfortran &> /dev/null; then
    echo -e "${RED}Error: gfortran not found. Install with: apt-get install gfortran${NC}"
    exit 1
fi
echo "✓ gfortran: $(gfortran --version | head -1)"

# Check gcc
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc not found. Install with: apt-get install build-essential${NC}"
    exit 1
fi
echo "✓ gcc: $(gcc --version | head -1)"

# Check ninja
if ! command -v ninja &> /dev/null; then
    echo -e "${RED}Error: ninja not found. Install with: apt-get install ninja-build${NC}"
    exit 1
fi
echo "✓ ninja: $(ninja --version)"

# Check for source directory
if [ ! -d "$COINHSL_SOURCE" ]; then
    echo -e "${RED}Error: Source directory not found: $COINHSL_SOURCE${NC}"
    exit 1
fi

# Detect BLAS/LAPACK
BLAS_LIB="openblas"
LAPACK_LIB="openblas"
BLAS_PATH=""
LAPACK_PATH=""

if [ -n "${CONDA_PREFIX:-}" ]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [ -f "$CONDA_LIB/libopenblas.so" ] || [ -f "$CONDA_LIB/libopenblas.a" ]; then
        BLAS_PATH="$CONDA_LIB"
        LAPACK_PATH="$CONDA_LIB"
        echo "✓ Found OpenBLAS in conda environment: $CONDA_LIB"
    fi
fi

if [ -z "$BLAS_PATH" ]; then
    # Try system paths
    for libdir in /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib; do
        if [ -f "$libdir/libopenblas.so" ] || [ -f "$libdir/libopenblas.a" ]; then
            BLAS_PATH="$libdir"
            LAPACK_PATH="$libdir"
            echo "✓ Found OpenBLAS in system: $libdir"
            break
        fi
    done
fi

if [ -z "$BLAS_PATH" ]; then
    echo -e "${YELLOW}Warning: OpenBLAS not found. Meson will try to auto-detect.${NC}"
fi

# Detect METIS (optional)
METIS_OPTIONS=""
if [ -n "${CONDA_PREFIX:-}" ]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [ -f "$CONDA_LIB/libmetis.so" ] || [ -f "$CONDA_LIB/libmetis.a" ]; then
        METIS_OPTIONS="-Dlibmetis=metis -Dlibmetis_path=[$CONDA_LIB]"
        echo "✓ Found METIS in conda environment: $CONDA_LIB"
    fi
fi

if [ -z "$METIS_OPTIONS" ]; then
    # Try system paths
    for libdir in /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib; do
        if [ -f "$libdir/libmetis.so" ] || [ -f "$libdir/libmetis.a" ]; then
            METIS_OPTIONS="-Dlibmetis=metis -Dlibmetis_path=[$libdir]"
            echo "✓ Found METIS in system: $libdir"
            break
        fi
    done
fi

if [ -z "$METIS_OPTIONS" ]; then
    echo -e "${YELLOW}Note: METIS not found. Build will proceed without it (solvers may be slower).${NC}"
fi

echo ""

# Check if target directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Warning: Target directory already exists: $TARGET_DIR${NC}"
    read -p "Do you want to remove it and rebuild? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
    else
        echo "Aborting."
        exit 1
    fi
fi

# Build the library
echo -e "${GREEN}Building CoinHSL library...${NC}"
cd "$COINHSL_SOURCE"

# Clean any existing builddir (but preserve source)
if [ -d "builddir" ]; then
    echo "Cleaning existing builddir..."
    rm -rf builddir
fi

# Configure meson build
MESON_OPTS=(
    "setup"
    "builddir"
    "--buildtype=release"
    "--prefix=$STAGING_DIR"
    "-Dlibblas=$BLAS_LIB"
    "-Dliblapack=$LAPACK_LIB"
)

if [ -n "$BLAS_PATH" ]; then
    MESON_OPTS+=("-Dlibblas_path=[\"$BLAS_PATH\"]")
    MESON_OPTS+=("-Dliblapack_path=[\"$LAPACK_PATH\"]")
fi

if [ -n "$METIS_OPTIONS" ]; then
    # Parse METIS options and add to meson command
    MESON_OPTS+=("-Dlibmetis=metis")
    if [[ "$METIS_OPTIONS" =~ libmetis_path=\[([^\]]+)\] ]]; then
        METIS_PATH="${BASH_REMATCH[1]}"
        MESON_OPTS+=("-Dlibmetis_path=[\"$METIS_PATH\"]")
    fi
fi

echo "Running: meson ${MESON_OPTS[*]}"
meson "${MESON_OPTS[@]}"

# Compile
echo "Compiling..."
meson compile -C builddir

# Install to staging directory
echo "Installing to staging directory..."
meson install -C builddir

# Verify installation (check both lib and lib64)
if [ -f "$STAGING_DIR/lib/libcoinhsl.so" ]; then
    LIB_DIR="$STAGING_DIR/lib"
elif [ -f "$STAGING_DIR/lib64/libcoinhsl.so" ]; then
    LIB_DIR="$STAGING_DIR/lib64"
else
    echo -e "${RED}Error: libcoinhsl.so not found in staging directory${NC}"
    echo "Checked: $STAGING_DIR/lib/libcoinhsl.so"
    echo "Checked: $STAGING_DIR/lib64/libcoinhsl.so"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"
echo ""

# Package into final structure
echo -e "${GREEN}Packaging into binary directory structure...${NC}"

mkdir -p "$TARGET_DIR"

# Copy lib files (from lib or lib64)
if [ -d "$LIB_DIR" ]; then
    mkdir -p "$TARGET_DIR/lib"
    cp -r "$LIB_DIR"/* "$TARGET_DIR/lib/"
    echo "✓ Copied lib files from $LIB_DIR"
fi

# Copy include files
if [ -d "$STAGING_DIR/include" ]; then
    mkdir -p "$TARGET_DIR/include"
    cp -r "$STAGING_DIR/include"/* "$TARGET_DIR/include/"
    echo "✓ Copied include files"
fi

# Copy modules if they exist (check both lib and modules directory)
if [ -d "$STAGING_DIR/modules" ]; then
    mkdir -p "$TARGET_DIR/modules"
    cp -r "$STAGING_DIR/modules"/* "$TARGET_DIR/modules/" 2>/dev/null || true
    echo "✓ Copied Fortran modules"
elif [ -d "$LIB_DIR" ] && find "$LIB_DIR" -name "*.mod" -type f | grep -q .; then
    mkdir -p "$TARGET_DIR/modules"
    find "$LIB_DIR" -name "*.mod" -type f -exec cp {} "$TARGET_DIR/modules/" \;
    echo "✓ Copied Fortran modules"
fi

# Copy share/licenses
if [ -d "$STAGING_DIR/share" ]; then
    mkdir -p "$TARGET_DIR/share"
    cp -r "$STAGING_DIR/share"/* "$TARGET_DIR/share/" 2>/dev/null || true
    echo "✓ Copied share files"
fi

# Copy LICENSE from source if it exists
if [ -f "$COINHSL_SOURCE/LICENCE" ]; then
    mkdir -p "$TARGET_DIR/share/licenses/CoinHSL"
    cp "$COINHSL_SOURCE/LICENCE" "$TARGET_DIR/share/licenses/CoinHSL/"
    echo "✓ Copied LICENSE"
fi

# Verify final structure
echo ""
echo -e "${GREEN}Verifying package structure...${NC}"

REQUIRED_FILES=(
    "$TARGET_DIR/lib/libcoinhsl.so"
    "$TARGET_DIR/include/CoinHslConfig.h"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}Error: Missing required files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo "✓ All required files present"

# Compare structure with existing macOS binary
MACOS_DIR="$PROJECT_ROOT/libraries/CoinHSL.v${VERSION}.x86_64-apple-darwin-libgfortran5"
if [ -d "$MACOS_DIR" ]; then
    echo "Comparing structure with macOS binary..."

    # Check for similar directory structure
    MACOS_DIRS=$(find "$MACOS_DIR" -type d -mindepth 1 | sed "s|^$MACOS_DIR||" | sort)
    LINUX_DIRS=$(find "$TARGET_DIR" -type d -mindepth 1 | sed "s|^$TARGET_DIR||" | sort)

    if [ "$MACOS_DIRS" = "$LINUX_DIRS" ]; then
        echo "✓ Directory structure matches macOS binary"
    else
        echo -e "${YELLOW}Warning: Directory structure differs from macOS binary${NC}"
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Binary package created at:"
echo "  $TARGET_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify the build:"
echo "     python -c \"from campro.environment.hsl_detector import find_coinhsl_directory; print(find_coinhsl_directory())\""
echo ""
echo "  2. Test HSL detection:"
echo "     python -c \"from campro.environment.hsl_detector import get_hsl_library_path; print(get_hsl_library_path())\""
echo ""
echo "  3. Commit the new directory to the repository:"
echo "     git add $TARGET_DIR"
echo "     git commit -m 'Add Linux CoinHSL binaries'"
echo ""
