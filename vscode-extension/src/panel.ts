import * as vscode from "vscode";
import * as path from "path";
import { client, RunResult } from "./client";

interface PanelMessage {
  type: string;
  [key: string]: unknown;
}

export class AePanelProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _recentExecutions: RunResult[] = [];

  constructor(private readonly _extensionUri: vscode.Uri) {}

  resolveWebviewView(
    view: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = view;

    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, "media")],
    };

    view.webview.html = this._buildHtml(view.webview);

    view.webview.onDidReceiveMessage((msg: PanelMessage) => {
      this._handleMessage(msg);
    });

    // Send initial data once the webview is ready
    view.onDidChangeVisibility(() => {
      if (view.visible) {
        this._refreshStatus();
      }
    });

    this._refreshStatus();
  }

  notifyConnectionChange(connected: boolean): void {
    this._view?.webview.postMessage({ type: "connection", connected });
  }

  notifyExecution(result: RunResult): void {
    this._recentExecutions.unshift(result);
    if (this._recentExecutions.length > 20) {
      this._recentExecutions.pop();
    }
    this._view?.webview.postMessage({ type: "execution", result });
  }

  private async _refreshStatus(): Promise<void> {
    try {
      const [health, stats] = await Promise.all([client.health(), client.stats()]);
      this._view?.webview.postMessage({ type: "status", health, stats });
    } catch {
      this._view?.webview.postMessage({ type: "connection", connected: false });
    }
  }

  private async _handleMessage(msg: PanelMessage): Promise<void> {
    switch (msg.type) {
      case "ready":
        this._refreshStatus();
        this._view?.webview.postMessage({ type: "history", executions: this._recentExecutions });
        break;

      case "run": {
        const folders = vscode.workspace.workspaceFolders;
        const repoPath = (msg.repoPath as string | undefined) ?? folders?.[0]?.uri.fsPath ?? "";
        if (!repoPath) {
          this._view?.webview.postMessage({ type: "error", message: "No workspace folder open." });
          return;
        }
        this._view?.webview.postMessage({ type: "running", prompt: msg.prompt });
        try {
          const result = await client.run({
            repo_path: repoPath,
            prompt: msg.prompt as string,
            workflow_type: (msg.workflowType as string | undefined) ?? "general_qa",
          });
          this.notifyExecution(result);
        } catch (err) {
          this._view?.webview.postMessage({ type: "error", message: String(err) });
        }
        break;
      }

      case "refresh":
        this._refreshStatus();
        break;

      case "openDashboard":
        vscode.commands.executeCommand("ae.openDashboard");
        break;

      case "runCommand":
        vscode.commands.executeCommand(msg.command as string);
        break;
    }
  }

  private _buildHtml(webview: vscode.Webview): string {
    const cssUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, "media", "panel.css")
    );
    const jsUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, "media", "panel.js")
    );
    const nonce = getNonce();

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${cssUri}">
  <title>AI Engineering</title>
</head>
<body>
  <div id="header">
    <span id="status-dot" class="dot disconnected" title="Server status"></span>
    <span id="status-label">Connecting…</span>
    <button id="refresh-btn" title="Refresh status">↺</button>
  </div>

  <div id="stats-bar" class="hidden">
    <span id="stat-total">0</span> runs &nbsp;·&nbsp;
    <span id="stat-rate">—</span> success &nbsp;·&nbsp;
    <span id="stat-active">0</span> active
  </div>

  <section id="quick-actions">
    <h3>Quick Actions</h3>
    <div class="action-grid">
      <button class="action-btn" data-cmd="ae.review">Review Code</button>
      <button class="action-btn" data-cmd="ae.debug">Debug</button>
      <button class="action-btn" data-cmd="ae.generateTests">Gen Tests</button>
      <button class="action-btn" data-cmd="ae.explain">Explain</button>
      <button class="action-btn" data-cmd="ae.refactor">Refactor</button>
      <button class="action-btn" data-cmd="ae.openDashboard">Dashboard ↗</button>
    </div>
  </section>

  <section id="custom-run">
    <h3>Custom Prompt</h3>
    <textarea id="prompt-input" rows="3" placeholder="Describe the task…"></textarea>
    <button id="run-btn">Run</button>
    <div id="run-spinner" class="hidden">Running…</div>
  </section>

  <section id="history-section">
    <h3>Recent Executions</h3>
    <div id="history-list"></div>
  </section>

  <script nonce="${nonce}" src="${jsUri}"></script>
</body>
</html>`;
  }
}

function getNonce(): string {
  let text = "";
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
