#!/usr/bin/env bash
# Fix permissions for mounted volumes - must run as root
# This script is called by initializeCommand in devcontainer.json

set -euo pipefail

USER_ID=${MAMBA_USER_ID:-57439}
USER_NAME=${MAMBA_USER:-mambauser}

echo "Fixing volume permissions (running as root)..."

# Fix GitHub CLI config directory
if [ -d "/home/${USER_NAME}/.config/gh" ]; then
    chown -R ${USER_ID}:${USER_ID} /home/${USER_NAME}/.config/gh
    chmod -R u+rwX,go-rwx /home/${USER_NAME}/.config/gh
    echo "✓ Fixed permissions for ~/.config/gh"
else
    mkdir -p /home/${USER_NAME}/.config/gh
    chown -R ${USER_ID}:${USER_ID} /home/${USER_NAME}/.config/gh
    chmod -R u+rwX,go-rwx /home/${USER_NAME}/.config/gh
    echo "✓ Created and fixed permissions for ~/.config/gh"
fi

# Fix git credentials file
touch /home/${USER_NAME}/.git-credentials
chown ${USER_ID}:${USER_ID} /home/${USER_NAME}/.git-credentials
chmod 600 /home/${USER_NAME}/.git-credentials
echo "✓ Fixed permissions for ~/.git-credentials"

# Fix SSH directory
if [ -d "/home/${USER_NAME}/.ssh" ]; then
    chown -R ${USER_ID}:${USER_ID} /home/${USER_NAME}/.ssh
    chmod -R u+rwX,go-rwx /home/${USER_NAME}/.ssh
    echo "✓ Fixed permissions for ~/.ssh"
fi

echo "Permission fix complete!"
