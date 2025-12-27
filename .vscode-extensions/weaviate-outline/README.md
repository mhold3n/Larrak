# Weaviate Code Outline Extension

VS Code extension that provides code outline using Weaviate-backed symbol tracking.

## Features

- Displays hierarchical code structure in the Outline view
- Shows classes, functions, methods with decorators and type annotations
- Real-time navigation to symbols
- Supports async functions and return types

## Installation

1. Install dependencies:

```bash
cd .vscode-extensions/weaviate-outline
npm install
```

1. Compile the extension:

```bash
npm run compile
```

1. Install the extension:
   - Open VS Code/Cursor
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Extensions: Install from VSIX" (if packaged) or use Developer mode

## Configuration

- `weaviateOutline.apiUrl`: URL of the Weaviate outline API server (default: `http://localhost:5001`)
- `weaviateOutline.enabled`: Enable/disable the extension (default: `true`)

## Requirements

- Weaviate API server must be running: `python dashboard/api.py --port 5001`
- Python files must be indexed: `python truthmaker/ingestion/code_scanner.py`

## Usage

1. Start the Weaviate API server
2. Open a Python file that has been indexed
3. The Outline view will automatically populate with symbols
4. Click on any symbol to navigate to its definition

## Development

Install in development mode:

1. Open this folder in VS Code
2. Press F5 to launch Extension Development Host
3. Test in the new window
