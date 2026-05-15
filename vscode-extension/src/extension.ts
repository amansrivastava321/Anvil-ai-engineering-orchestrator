import * as vscode from "vscode";
import * as path from "path";
import { client, RunPayload } from "./client";
import { AePanelProvider } from "./panel";
import { AeCodeLensProvider } from "./annotations";

// ── Status bar ────────────────────────────────────────────────────────────────

let statusBarItem: vscode.StatusBarItem;
let panelProvider: AePanelProvider | undefined;

function updateStatusBar(connected: boolean): void {
  statusBarItem.text = connected ? "$(circuit-board) AE ●" : "$(circuit-board) AE ○";
  statusBarItem.tooltip = connected
    ? "AI Engineering Orchestrator — connected"
    : "AI Engineering Orchestrator — server not running";
  statusBarItem.color = connected ? new vscode.ThemeColor("charts.green") : new vscode.ThemeColor("charts.red");
}

async function checkConnection(): Promise<void> {
  const ok = await client.ping();
  updateStatusBar(ok);
  panelProvider?.notifyConnectionChange(ok);
}

// ── Workspace helpers ─────────────────────────────────────────────────────────

function getRepoPath(uri?: vscode.Uri): string {
  if (uri) {
    return uri.fsPath;
  }
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length > 0) {
    return folders[0].uri.fsPath;
  }
  return "";
}

function currentFileRepoPath(uri?: vscode.Uri): string {
  const filePath = uri?.fsPath ?? vscode.window.activeTextEditor?.document.uri.fsPath;
  if (!filePath) {
    return getRepoPath();
  }
  // Walk up to find the workspace root that contains this file
  const folders = vscode.workspace.workspaceFolders ?? [];
  for (const folder of folders) {
    if (filePath.startsWith(folder.uri.fsPath)) {
      return folder.uri.fsPath;
    }
  }
  return path.dirname(filePath);
}

// ── Output channel ────────────────────────────────────────────────────────────

const outputChannel = vscode.window.createOutputChannel("AI Engineering");

function showResult(result: {
  response?: string;
  error?: string;
  model_used?: string;
  duration_ms?: number;
  tokens_used?: number;
}): void {
  outputChannel.clear();
  if (result.error) {
    outputChannel.appendLine(`ERROR: ${result.error}`);
  } else {
    outputChannel.appendLine(result.response ?? "(no response)");
    outputChannel.appendLine("");
    outputChannel.appendLine(
      `— model: ${result.model_used ?? "?"} · ${Math.round(result.duration_ms ?? 0)}ms · ${result.tokens_used ?? 0} tokens`
    );
  }
  outputChannel.show(true);
}

// ── Command factory ───────────────────────────────────────────────────────────

function makeWorkflowCommand(workflowType: string, prompt: string) {
  return async (uri?: vscode.Uri) => {
    const repoPath = currentFileRepoPath(uri);
    if (!repoPath) {
      vscode.window.showErrorMessage("No workspace folder open.");
      return;
    }
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `AI Engineering: ${prompt}…`,
        cancellable: false,
      },
      async () => {
        try {
          const payload: RunPayload = {
            repo_path: repoPath,
            prompt,
            workflow_type: workflowType,
          };
          if (uri && !vscode.workspace.getWorkspaceFolder(uri)?.uri) {
            // Single file context — narrow the prompt
            payload.prompt = `${prompt} — focus on ${path.basename(uri.fsPath)}`;
          }
          const result = await client.run(payload);
          showResult(result);
          panelProvider?.notifyExecution(result);
        } catch (err) {
          vscode.window.showErrorMessage(`AE server error: ${String(err)}`);
        }
      }
    );
  };
}

// ── Custom run command ────────────────────────────────────────────────────────

async function runCustomPrompt(uri?: vscode.Uri): Promise<void> {
  const repoPath = currentFileRepoPath(uri);
  if (!repoPath) {
    vscode.window.showErrorMessage("No workspace folder open.");
    return;
  }
  const prompt = await vscode.window.showInputBox({
    prompt: "Describe the task for the AI Engineering Orchestrator",
    placeHolder: "e.g. explain the payment retry logic",
  });
  if (!prompt) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "AI Engineering: running…",
      cancellable: false,
    },
    async () => {
      try {
        const result = await client.run({ repo_path: repoPath, prompt, workflow_type: "general_qa" });
        showResult(result);
        panelProvider?.notifyExecution(result);
      } catch (err) {
        vscode.window.showErrorMessage(`AE server error: ${String(err)}`);
      }
    }
  );
}

// ── Index repo command ────────────────────────────────────────────────────────

async function indexRepository(uri?: vscode.Uri): Promise<void> {
  const repoPath = getRepoPath(uri);
  if (!repoPath) {
    vscode.window.showErrorMessage("No workspace folder to index.");
    return;
  }
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "AE: indexing repository with bge-m3…",
      cancellable: false,
    },
    async () => {
      try {
        const result = await client.indexRepo(repoPath);
        vscode.window.showInformationMessage(
          `Indexed ${result.total_functions} functions in ${path.basename(repoPath)}.`
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Index failed: ${String(err)}`);
      }
    }
  );
}

// ── Activate ──────────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "ae.panel.focus";
  statusBarItem.text = "$(circuit-board) AE";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // Side panel
  panelProvider = new AePanelProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("ae.panel", panelProvider)
  );

  // CodeLens
  const codeLensProvider = new AeCodeLensProvider();
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(
      ["python", "typescript", "javascript", "go", "rust", "java"],
      codeLensProvider
    )
  );

  // Commands
  context.subscriptions.push(
    vscode.commands.registerCommand("ae.review",
      makeWorkflowCommand("code_review", "Review code for issues, bugs, and improvements")),
    vscode.commands.registerCommand("ae.debug",
      makeWorkflowCommand("debug_analysis", "Find and fix bugs and errors")),
    vscode.commands.registerCommand("ae.generateTests",
      makeWorkflowCommand("test_generation", "Generate comprehensive tests with edge cases")),
    vscode.commands.registerCommand("ae.explain",
      makeWorkflowCommand("documentation", "Explain the architecture and purpose of this code")),
    vscode.commands.registerCommand("ae.refactor",
      makeWorkflowCommand("code_refactoring", "Refactor and improve code quality")),
    vscode.commands.registerCommand("ae.run", runCustomPrompt),
    vscode.commands.registerCommand("ae.indexRepo", indexRepository),
    vscode.commands.registerCommand("ae.openDashboard", () => {
      const url = (vscode.workspace.getConfiguration("ae").get<string>("serverUrl") ?? "http://localhost:8008") + "/dashboard";
      vscode.env.openExternal(vscode.Uri.parse(url));
    }),
  );

  // Check connection on startup and every 30 s
  if (vscode.workspace.getConfiguration("ae").get<boolean>("autoConnect")) {
    checkConnection();
    const timer = setInterval(checkConnection, 30_000);
    context.subscriptions.push({ dispose: () => clearInterval(timer) });
  }
}

export function deactivate(): void {}
