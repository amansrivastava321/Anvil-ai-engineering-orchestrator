// Runs inside the VSCode webview (sandboxed browser context).
// Communicates with the extension host via acquireVsCodeApi().

(function () {
  "use strict";

  const vscode = acquireVsCodeApi();

  // ── Element refs ──────────────────────────────────────────────────────────

  const statusDot   = document.getElementById("status-dot");
  const statusLabel = document.getElementById("status-label");
  const statsBar    = document.getElementById("stats-bar");
  const statTotal   = document.getElementById("stat-total");
  const statRate    = document.getElementById("stat-rate");
  const statActive  = document.getElementById("stat-active");
  const historyList = document.getElementById("history-list");
  const promptInput = document.getElementById("prompt-input");
  const runBtn      = document.getElementById("run-btn");
  const runSpinner  = document.getElementById("run-spinner");
  const refreshBtn  = document.getElementById("refresh-btn");

  // ── State ─────────────────────────────────────────────────────────────────

  let connected = false;

  // ── Connection display ────────────────────────────────────────────────────

  function setConnected(ok) {
    connected = ok;
    statusDot.className = "dot " + (ok ? "connected" : "disconnected");
    statusLabel.textContent = ok ? "Connected" : "Server not running";
    statsBar.classList.toggle("hidden", !ok);
  }

  // ── Status update ─────────────────────────────────────────────────────────

  function applyStatus(health, stats) {
    setConnected(true);
    if (stats) {
      const total = stats.total_executions ?? 0;
      const succ  = stats.successful_executions ?? 0;
      statTotal.textContent  = String(total);
      statRate.textContent   = total ? Math.round(succ / total * 100) + "%" : "—";
      statActive.textContent = String(stats.active_executions ?? 0);
    }
  }

  // ── History ───────────────────────────────────────────────────────────────

  function addExecItem(result) {
    const ok = !result.error && result.status !== "failed";
    const wf = (result.workflow_type ?? "general_qa").replace(/_/g, " ");
    const ms = result.duration_ms ? Math.round(result.duration_ms) + "ms" : "";
    const snip = (result.response ?? result.error ?? "").slice(0, 120);

    const item = document.createElement("div");
    item.className = "exec-item";
    item.innerHTML = `
      <div class="exec-header">
        <span class="exec-dot ${ok ? "ok" : "err"}"></span>
        <span class="exec-workflow">${escHtml(capitalize(wf))}</span>
        <span class="exec-meta">${escHtml(ms)}</span>
      </div>
      <div class="exec-snippet">${escHtml(snip)}</div>
    `;
    historyList.prepend(item);

    // Keep at most 20 items
    while (historyList.children.length > 20) {
      historyList.removeChild(historyList.lastChild);
    }
  }

  function populateHistory(executions) {
    historyList.innerHTML = "";
    (executions ?? []).forEach(addExecItem);
  }

  // ── Run custom prompt ─────────────────────────────────────────────────────

  function startRun() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    runBtn.disabled = true;
    runSpinner.classList.remove("hidden");

    vscode.postMessage({ type: "run", prompt });
  }

  function endRun() {
    runBtn.disabled = false;
    runSpinner.classList.add("hidden");
  }

  // ── Quick action buttons ──────────────────────────────────────────────────

  document.querySelectorAll(".action-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cmd = btn.getAttribute("data-cmd");
      if (cmd === "ae.openDashboard") {
        vscode.postMessage({ type: "openDashboard" });
      } else {
        vscode.postMessage({ type: "runCommand", command: cmd });
      }
    });
  });

  // ── Refresh button ────────────────────────────────────────────────────────

  refreshBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "refresh" });
  });

  // ── Run button + Enter in textarea ───────────────────────────────────────

  runBtn.addEventListener("click", startRun);
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      startRun();
    }
  });

  // ── Messages from extension host ──────────────────────────────────────────

  window.addEventListener("message", (event) => {
    const msg = event.data;
    switch (msg.type) {
      case "connection":
        setConnected(msg.connected);
        break;
      case "status":
        applyStatus(msg.health, msg.stats);
        break;
      case "execution":
        addExecItem(msg.result);
        endRun();
        break;
      case "history":
        populateHistory(msg.executions);
        break;
      case "running":
        // already handled by startRun()
        break;
      case "error":
        endRun();
        // Show in a subtle way without blocking the webview
        statusLabel.textContent = "Error: " + (msg.message ?? "unknown");
        break;
    }
  });

  // ── Helpers ───────────────────────────────────────────────────────────────

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  vscode.postMessage({ type: "ready" });
})();
