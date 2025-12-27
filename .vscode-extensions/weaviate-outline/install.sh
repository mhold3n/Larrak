#!/bin/bash
# Installation script for Weaviate Outline VS Code Extension

set -e

echo "Installing Weaviate Outline VS Code Extension..."
echo "================================================"

EXTENSION_DIR="$PWD/.vscode-extensions/weaviate-outline"
VS_CODE_EXTENSIONS_DIR="$HOME/.vscode/extensions"

# Detect if using Cursor instead of VS Code
if [ -d "$HOME/.cursor/extensions" ]; then
    VS_CODE_EXTENSIONS_DIR="$HOME/.cursor-tutor/extensions"
    echo "Detected Cursor IDE"
elif [ -d "$HOME/.vscode/extensions" ]; then
    echo "Detected VS Code"
else
    echo "Error: Could not find VS Code or Cursor extensions directory"
    exit 1
fi

echo "Extension directory: $EXTENSION_DIR"
echo "Target directory: $VS_CODE_EXTENSIONS_DIR"

# Check if extension is compiled
if [ ! -d "$EXTENSION_DIR/out" ]; then
    echo "Error: Extension not compiled. Run 'npm run compile' first"
    exit 1
fi

# Create symlink
EXTENSION_NAME="larrak.weaviate-outline-0.1.0"
TARGET_PATH="$VS_CODE_EXTENSIONS_DIR/$EXTENSION_NAME"

if [ -L "$TARGET_PATH" ] || [ -d "$TARGET_PATH" ]; then
    echo "Removing existing installation..."
    rm -rf "$TARGET_PATH"
fi

echo "Creating symlink..."
ln -s "$EXTENSION_DIR" "$TARGET_PATH"

echo ""
echo "✅ Extension installed successfully!"
echo ""
echo "Next steps:"
echo "1. Restart VS Code/Cursor"
echo "2. Start the Weaviate API: python dashboard/api.py --port 5001"
echo "3. Open a Python file"
echo "4. Check the Outline view (should populate automatically)"
echo ""
echo "Configuration:"
echo "  - Set API URL: Settings → Extensions → Weaviate Outline → Api Url"
echo "  - Default: http://localhost:5001"
echo ""
