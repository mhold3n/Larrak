import * as vscode from 'vscode';
import axios from 'axios';

interface OutlineSymbol {
    name: string;
    kind: string;
    line_start: number;
    line_end: number;
    signature: string;
    docstring?: string;
    decorators?: string[];
    is_async?: boolean;
    return_type?: string;
    children?: OutlineSymbol[];
}

interface OutlineResponse {
    file: string;
    symbols: OutlineSymbol[];
    count: number;
}

class WeaviateOutlineProvider implements vscode.DocumentSymbolProvider {
    private apiUrl: string;

    constructor() {
        this.apiUrl = this.getApiUrl();

        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('weaviateOutline.apiUrl')) {
                this.apiUrl = this.getApiUrl();
            }
        });
    }

    private getApiUrl(): string {
        const config = vscode.workspace.getConfiguration('weaviateOutline');
        return config.get<string>('apiUrl', 'http://localhost:5001');
    }

    private getRelativePath(document: vscode.TextDocument): string {
        const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
        if (!workspaceFolder) {
            return document.fileName;
        }
        return document.uri.fsPath.substring(workspaceFolder.uri.fsPath.length + 1);
    }

    private mapKindToSymbolKind(kind: string): vscode.SymbolKind {
        const kindMap: { [key: string]: vscode.SymbolKind } = {
            'class': vscode.SymbolKind.Class,
            'function': vscode.SymbolKind.Function,
            'method': vscode.SymbolKind.Method,
            'variable': vscode.SymbolKind.Variable,
            'property': vscode.SymbolKind.Property,
            'field': vscode.SymbolKind.Field,
        };
        return kindMap[kind] || vscode.SymbolKind.Variable;
    }

    private convertSymbol(symbol: OutlineSymbol): vscode.DocumentSymbol {
        // Create range for the symbol (line numbers are 1-indexed from API, 0-indexed in VS Code)
        const startLine = Math.max(0, symbol.line_start - 1);
        const endLine = Math.max(0, (symbol.line_end || symbol.line_start) - 1);

        const range = new vscode.Range(
            new vscode.Position(startLine, 0),
            new vscode.Position(endLine, 0)
        );

        // Use first line as selection range
        const selectionRange = new vscode.Range(
            new vscode.Position(startLine, 0),
            new vscode.Position(startLine, 0)
        );

        // Build detail string with decorators and return type
        let detail = '';
        if (symbol.decorators && symbol.decorators.length > 0) {
            detail = `@${symbol.decorators.join(' @')} `;
        }
        if (symbol.is_async) {
            detail += 'async ';
        }
        if (symbol.return_type) {
            detail += `→ ${symbol.return_type}`;
        }

        const docSymbol = new vscode.DocumentSymbol(
            symbol.name,
            detail,
            this.mapKindToSymbolKind(symbol.kind),
            range,
            selectionRange
        );

        // Add children recursively
        if (symbol.children && symbol.children.length > 0) {
            docSymbol.children = symbol.children.map(child => this.convertSymbol(child));
        }

        return docSymbol;
    }

    async provideDocumentSymbols(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<vscode.DocumentSymbol[] | undefined> {
        // Check if enabled
        const config = vscode.workspace.getConfiguration('weaviateOutline');
        if (!config.get<boolean>('enabled', true)) {
            return undefined;
        }

        // Only process Python files
        if (document.languageId !== 'python') {
            return undefined;
        }

        const relativePath = this.getRelativePath(document);
        const url = `${this.apiUrl}/api/outline/${encodeURIComponent(relativePath)}`;

        try {
            const response = await axios.get<OutlineResponse>(url, {
                timeout: 5000,
                validateStatus: (status) => status === 200 || status === 404
            });

            if (response.status === 404) {
                // File not indexed - trigger auto-index
                await this.triggerReindex(relativePath);
                return [];
            }

            const data = response.data;

            // Convert symbols to VS Code DocumentSymbol format
            return data.symbols.map(symbol => this.convertSymbol(symbol));

        } catch (error) {
            if (axios.isAxiosError(error)) {
                if (error.code === 'ECONNREFUSED') {
                    vscode.window.showErrorMessage(
                        `Cannot connect to Weaviate API at ${this.apiUrl}. Start services with: ./scripts/start-weaviate-services.sh`,
                        { modal: false }
                    );
                } else if (error.response?.status === 404) {
                    // File not indexed - trigger auto-index
                    await this.triggerReindex(relativePath);
                    return [];
                } else {
                    console.error('Weaviate Outline API error:', error.message);
                }
            } else {
                console.error('Unexpected error:', error);
            }
            return undefined;
        }
    }

    async triggerReindex(filePath: string): Promise<void> {
        try {
            const response = await axios.post(
                `${this.apiUrl}/api/outline/reindex`,
                { file: filePath },
                { timeout: 30000 } // 30 second timeout for indexing
            );

            if (response.status === 200) {
                console.log(`✅ Re-indexed: ${filePath}`);
                // Refresh outline after a short delay
                setTimeout(() => {
                    vscode.commands.executeCommand('vscode.executeDocumentSymbolProvider');
                }, 500);
            }
        } catch (error) {
            console.error(`Failed to trigger re-index for ${filePath}:`, error);
        }
    }
}

class FileWatcher {
    private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
    private provider: WeaviateOutlineProvider;

    constructor(provider: WeaviateOutlineProvider) {
        this.provider = provider;
    }

    async onFileSaved(document: vscode.TextDocument): Promise<void> {
        // Only handle Python files
        if (document.languageId !== 'python') {
            return;
        }

        const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
        if (!workspaceFolder) {
            return;
        }

        const relativePath = document.uri.fsPath.substring(workspaceFolder.uri.fsPath.length + 1);

        // Debounce: cancel existing timer if file is saved again quickly
        const existingTimer = this.debounceTimers.get(relativePath);
        if (existingTimer) {
            clearTimeout(existingTimer);
        }

        // Set new timer to trigger re-index after 1 second of no saves
        const timer = setTimeout(async () => {
            console.log(`📝 File saved, triggering re-index: ${relativePath}`);
            await this.provider.triggerReindex(relativePath);
            this.debounceTimers.delete(relativePath);
        }, 1000);

        this.debounceTimers.set(relativePath, timer);
    }
}

export function activate(context: vscode.ExtensionContext) {
    console.log('Weaviate Outline extension is now active');

    const provider = new WeaviateOutlineProvider();
    const fileWatcher = new FileWatcher(provider);

    // Register document symbol provider
    const symbolProvider = vscode.languages.registerDocumentSymbolProvider(
        { language: 'python', scheme: 'file' },
        provider
    );

    // Register file save listener
    const saveListener = vscode.workspace.onDidSaveTextDocument((document) => {
        fileWatcher.onFileSaved(document);
    });

    context.subscriptions.push(symbolProvider, saveListener);
}

export function deactivate() {
    console.log('Weaviate Outline extension is now deactivated');
}
