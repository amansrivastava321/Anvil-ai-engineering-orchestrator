import * as vscode from "vscode";

/**
 * Provides CodeLens annotations above every top-level function/class definition,
 * offering quick "Review" and "Generate Test" actions inline in the editor.
 */
export class AeCodeLensProvider implements vscode.CodeLensProvider {
  private readonly _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;

  // Patterns per language that match the opening line of a function or class.
  private static readonly _PATTERNS: Record<string, RegExp> = {
    python:     /^(async\s+)?def\s+\w+|^class\s+\w+/,
    typescript: /^(export\s+)?(async\s+)?function\s+\w+|^(export\s+)?(abstract\s+)?class\s+\w+|^\s*(public|private|protected|static|async).*\(.*\)\s*[:{]/,
    javascript: /^(export\s+)?(async\s+)?function\s+\w+|^(export\s+)?class\s+\w+/,
    go:         /^func\s+/,
    rust:       /^(pub\s+)?(async\s+)?fn\s+\w+|^(pub\s+)?struct\s+\w+|^(pub\s+)?impl\s+/,
    java:       /^\s*(public|private|protected)\s+.*\(.*\)\s*[{]/,
  };

  provideCodeLenses(
    document: vscode.TextDocument,
    _token: vscode.CancellationToken
  ): vscode.CodeLens[] {
    const langId = document.languageId;
    const pattern = AeCodeLensProvider._PATTERNS[langId];
    if (!pattern) {
      return [];
    }

    const lenses: vscode.CodeLens[] = [];
    const text = document.getText();
    const lines = text.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!pattern.test(line.trimEnd())) {
        continue;
      }

      const range = new vscode.Range(i, 0, i, line.length);

      lenses.push(
        new vscode.CodeLens(range, {
          title: "$(search) Review",
          tooltip: "Ask AI to review this function/class",
          command: "ae.review",
          arguments: [document.uri],
        }),
        new vscode.CodeLens(range, {
          title: "$(beaker) Gen Test",
          tooltip: "Ask AI to generate tests for this function/class",
          command: "ae.generateTests",
          arguments: [document.uri],
        }),
        new vscode.CodeLens(range, {
          title: "$(comment) Explain",
          tooltip: "Ask AI to explain this function/class",
          command: "ae.explain",
          arguments: [document.uri],
        })
      );
    }

    return lenses;
  }

  refresh(): void {
    this._onDidChangeCodeLenses.fire();
  }
}
