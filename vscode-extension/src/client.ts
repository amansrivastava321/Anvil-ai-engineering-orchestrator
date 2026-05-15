import * as https from "https";
import * as http from "http";
import { URL } from "url";
import * as vscode from "vscode";

export interface RunPayload {
  repo_path: string;
  prompt: string;
  workflow_type: string;
  mode?: "sync" | "streaming";
  preferred_model?: string;
}

export interface RunResult {
  execution_id: string;
  status: string;
  response?: string;
  error?: string;
  model_used?: string;
  workflow_type?: string;
  duration_ms?: number;
  tokens_used?: number;
  files_analyzed?: string[];
  warnings?: string[];
}

export interface SystemHealth {
  ollama?: { status: string; model_count: number };
  graphify?: { available: boolean; repos_analyzed: number };
  skills?: { loaded: boolean; count: number };
}

export interface AgentStats {
  total_executions: number;
  successful_executions: number;
  active_executions: number;
  avg_duration_ms: number;
}

function getServerUrl(): string {
  return (
    vscode.workspace.getConfiguration("ae").get<string>("serverUrl") ??
    "http://localhost:8008"
  );
}

function request<T>(
  method: string,
  path: string,
  body?: unknown
): Promise<T> {
  return new Promise((resolve, reject) => {
    const base = getServerUrl();
    const url = new URL(path, base);
    const isHttps = url.protocol === "https:";
    const mod = isHttps ? https : http;

    const payload = body ? JSON.stringify(body) : undefined;
    const options: http.RequestOptions = {
      method,
      hostname: url.hostname,
      port: url.port || (isHttps ? 443 : 80),
      path: url.pathname + url.search,
      headers: {
        "Content-Type": "application/json",
        ...(payload ? { "Content-Length": Buffer.byteLength(payload) } : {}),
      },
    };

    const req = mod.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data) as T);
        } catch {
          reject(new Error(`Invalid JSON from ${path}: ${data.slice(0, 200)}`));
        }
      });
    });

    req.on("error", reject);
    req.setTimeout(300_000, () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });

    if (payload) {
      req.write(payload);
    }
    req.end();
  });
}

export const client = {
  async ping(): Promise<boolean> {
    try {
      await request<unknown>("GET", "/api/v1/system/status");
      return true;
    } catch {
      return false;
    }
  },

  async health(): Promise<SystemHealth> {
    return request<SystemHealth>("GET", "/api/v1/system/status");
  },

  async stats(): Promise<AgentStats> {
    return request<AgentStats>("GET", "/api/v1/agent/stats");
  },

  async run(payload: RunPayload): Promise<RunResult> {
    return request<RunResult>("POST", "/api/v1/agent/run", payload);
  },

  async indexRepo(repoPath: string): Promise<{ indexed: boolean; total_functions: number }> {
    return request("POST", "/api/v1/agent/index", { repo_path: repoPath });
  },

  async listRepos(): Promise<{ repos: Array<{ path: string; name?: string }> }> {
    return request("GET", "/api/v1/repo/list");
  },
};
