#!/bin/bash
# Entrypoint script to fix volume permissions and start container

# Don't exit on error for permission fixes
set +e

# Get the user ID from environment (set by micromamba base image)
USER_ID=${MAMBA_USER_ID:-57439}
USER_NAME=${MAMBA_USER:-mambauser}

# Fix permissions for mounted volumes
if [ "$(id -u)" = "0" ]; then
    # Running as root - fix permissions
    echo "[entrypoint] Fixing permissions for mounted volumes (running as root)..."

    # Ensure directories exist and have correct ownership
    mkdir -p /home/${USER_NAME}/.config/gh
    chown -R ${USER_ID}:${USER_ID} /home/${USER_NAME}/.config/gh 2>/dev/null
    chmod -R u+rwX,go-rwx /home/${USER_NAME}/.config/gh 2>/dev/null

    # Fix git credentials file
    touch /home/${USER_NAME}/.git-credentials
    chown ${USER_ID}:${USER_ID} /home/${USER_NAME}/.git-credentials 2>/dev/null
    chmod 600 /home/${USER_NAME}/.git-credentials 2>/dev/null

    # Fix SSH directory (read-only mount, but ensure parent exists)
    mkdir -p /home/${USER_NAME}/.ssh
    chown -R ${USER_ID}:${USER_ID} /home/${USER_NAME}/.ssh 2>/dev/null
    chmod -R u+rwX,go-rwx /home/${USER_NAME}/.ssh 2>/dev/null

    echo "[entrypoint] Permissions fixed. Switching to user ${USER_NAME}..."

    # Switch to the user and execute the command
    set -e
    if [ $# -eq 0 ]; then
        # No command provided, use default shell
        exec gosu ${USER_NAME} /bin/bash
    else
        exec gosu ${USER_NAME} "$@"
    fi
else
    # Already running as the user - just ensure directories exist
    echo "[entrypoint] Running as user, ensuring directories exist..."
    mkdir -p /home/${USER_NAME}/.config/gh 2>/dev/null || true
    touch /home/${USER_NAME}/.git-credentials 2>/dev/null || true
    set -e
    if [ $# -eq 0 ]; then
        exec /bin/bash
    else
        exec "$@"
    fi
fi
