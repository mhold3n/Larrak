#!/usr/bin/env bash
# Quick check script for GitHub authentication status

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GitHub Authentication Status ===${NC}"
echo ""

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}✗ GitHub CLI (gh) is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GitHub CLI is installed${NC}"
echo ""

# Check authentication status
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✓ GitHub authentication is configured${NC}"
    echo ""
    gh auth status
    echo ""

    # Verify authentication works
    if gh api user &> /dev/null; then
        echo -e "${GREEN}✓ Authentication verified - working correctly${NC}"
        echo ""

        # Show git remote configuration
        if [ -d .git ]; then
            echo "Git remote configuration:"
            git remote -v
            echo ""
        fi

        # Check git credential helper
        CREDENTIAL_HELPER=$(git config --global credential.helper || echo "not set")
        if echo "$CREDENTIAL_HELPER" | grep -q "gh"; then
            echo -e "${GREEN}✓ Git credential helper configured: $CREDENTIAL_HELPER${NC}"
        else
            echo -e "${YELLOW}⚠ Git credential helper not configured for GitHub CLI${NC}"
            echo "  Run: gh auth setup-git"
        fi

        echo ""
        echo -e "${GREEN}All systems ready! You can use git push/pull.${NC}"
        exit 0
    else
        echo -e "${RED}✗ Authentication found but verification failed${NC}"
        echo "  You may need to re-authenticate: gh auth login"
        exit 1
    fi
else
    echo -e "${RED}✗ GitHub authentication is not configured${NC}"
    echo ""
    echo "To set up authentication, run:"
    echo "  bash scripts/setup/setup_git_auth.sh"
    echo ""
    echo "Or manually:"
    echo "  gh auth login"
    exit 1
fi
