#!/usr/bin/env bash
# Fix permissions for mounted volumes
# This script attempts to fix permissions, but may need to be run manually with sudo if volumes are root-owned

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Checking volume permissions...${NC}"

# Try to fix permissions (will fail silently if we don't have permission)
# This is a best-effort attempt - if it fails, the user will need to run manually

# Fix GitHub CLI config directory
if [ -d "$HOME/.config/gh" ]; then
    chmod -R u+rwX,go-rwx "$HOME/.config/gh" 2>/dev/null && \
        echo -e "${GREEN}✓ Fixed permissions for ~/.config/gh${NC}" || \
        echo -e "${YELLOW}⚠ Could not fix permissions for ~/.config/gh (may need manual fix)${NC}"
else
    mkdir -p "$HOME/.config/gh"
    chmod -R u+rwX,go-rwx "$HOME/.config/gh" 2>/dev/null && \
        echo -e "${GREEN}✓ Created and fixed permissions for ~/.config/gh${NC}" || \
        echo -e "${YELLOW}⚠ Could not set permissions for ~/.config/gh${NC}"
fi

# Fix git credentials file
touch "$HOME/.git-credentials" 2>/dev/null || true
chmod 600 "$HOME/.git-credentials" 2>/dev/null && \
    echo -e "${GREEN}✓ Fixed permissions for ~/.git-credentials${NC}" || \
    echo -e "${YELLOW}⚠ Could not fix permissions for ~/.git-credentials${NC}"

echo -e "${GREEN}Permission check complete${NC}"
