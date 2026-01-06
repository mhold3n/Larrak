#!/usr/bin/env bash
# Setup GitHub authentication for devcontainer
# This script prompts users to login on first container startup
# and persists credentials across container restarts

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GitHub Authentication Setup ===${NC}"
echo ""

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed${NC}"
    exit 1
fi

# Check if already authenticated
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✓ GitHub authentication found${NC}"
    gh auth status
    echo ""

    # Verify authentication works
    if gh api user &> /dev/null; then
        echo -e "${GREEN}✓ Authentication verified successfully${NC}"

        # Configure git credential helper if not already set
        if ! git config --global credential.helper | grep -q "gh"; then
            echo "Configuring git to use GitHub CLI for authentication..."
            gh auth setup-git
            echo -e "${GREEN}✓ Git credential helper configured${NC}"
        fi

        echo ""
        echo -e "${GREEN}GitHub authentication is ready!${NC}"
        exit 0
    else
        echo -e "${YELLOW}Warning: Authentication found but verification failed${NC}"
        echo "You may need to re-authenticate."
    fi
fi

# Not authenticated - prompt user
echo -e "${YELLOW}GitHub authentication is not configured.${NC}"
echo ""
echo "To use git operations (push, pull, etc.), you need to authenticate with GitHub."
echo ""

# Check if running in interactive terminal
if [ ! -t 0 ]; then
    echo -e "${YELLOW}Non-interactive terminal detected.${NC}"
    echo "To set up authentication, run this script manually:"
    echo "  bash scripts/setup/setup_git_auth.sh"
    echo ""
    echo "Or run: gh auth login"
    exit 0
fi

# Prompt user
read -p "Would you like to authenticate with GitHub now? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Authentication skipped. You can run this script later:${NC}"
    echo "  bash scripts/setup/setup_git_auth.sh"
    echo ""
    echo "Or manually authenticate with:"
    echo "  gh auth login"
    exit 0
fi

echo ""
echo -e "${BLUE}Starting GitHub authentication...${NC}"
echo ""

# Ensure directories exist with correct permissions
echo "Setting up GitHub CLI configuration directory..."
mkdir -p ~/.config/gh

# Try to fix permissions (entrypoint should have done this, but try again)
if ! chmod -R u+rwX,go-rwx ~/.config/gh 2>/dev/null; then
    echo -e "${YELLOW}Warning: Could not set permissions on ~/.config/gh${NC}"
    echo "This may cause authentication to fail. The entrypoint script should fix this."
    echo "If authentication fails, try running: bash scripts/setup/fix_volume_permissions.sh"
fi

# Run GitHub CLI login (interactive)
if gh auth login; then
    echo ""
    echo -e "${GREEN}✓ Authentication successful!${NC}"

    # Configure git credential helper
    echo "Configuring git to use GitHub CLI for authentication..."
    gh auth setup-git

    # Verify authentication
    if gh api user &> /dev/null; then
        echo -e "${GREEN}✓ Authentication verified${NC}"
        echo ""
        echo -e "${GREEN}GitHub authentication is ready!${NC}"
        echo "You can now use git push/pull without additional authentication."
    else
        echo -e "${RED}Error: Authentication verification failed${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${RED}Authentication failed or was cancelled${NC}"
    echo "You can try again later by running:"
    echo "  bash scripts/setup/setup_git_auth.sh"
    exit 1
fi
