/* eslint-disable @typescript-eslint/no-require-imports */
const { app, BrowserWindow } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const localApi = process.env.MEETASR_LOCAL_API || "http://127.0.0.1:8000";
const remoteApi = process.env.MEETASR_REMOTE_API || process.env.NEXT_PUBLIC_MEETASR_API;
const frontendUrl = process.env.MEETASR_FRONTEND_URL || "http://127.0.0.1:3000";
let localBackend;

async function isHealthy(url) {
  try {
    const response = await fetch(`${url.replace(/\/$/, "")}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    return response.ok;
  } catch {
    return false;
  }
}

function resolveVirtualEnvPython(workspaceRoot) {
  const configured = process.env.MEETASR_PYTHON;
  const candidates = configured
    ? [configured]
    : process.platform === "win32"
      ? [
          path.join(workspaceRoot, ".venv", "Scripts", "python.exe"),
          path.join(workspaceRoot, "venv", "Scripts", "python.exe"),
        ]
      : [
          path.join(workspaceRoot, ".venv", "bin", "python"),
          path.join(workspaceRoot, "venv", "bin", "python"),
        ];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

function startLocalBackend() {
  if (process.env.MEETASR_DESKTOP_LOCAL_BACKEND === "false") return false;

  const workspaceRoot = path.resolve(__dirname, "..", "..");
  const python = resolveVirtualEnvPython(workspaceRoot);
  if (!python) {
    console.warn(
      "Local backend disabled: create a project virtual environment or set MEETASR_PYTHON to its interpreter.",
    );
    return false;
  }
  localBackend = spawn(
    python,
    ["-m", "uvicorn", "meetasr.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
    {
      cwd: workspaceRoot,
      env: { ...process.env, MEETASR_DESKTOP: "1" },
      stdio: "ignore",
      windowsHide: true,
    },
  );
  localBackend.once("error", (error) => {
    console.warn("Unable to start local MeetASR backend:", error.message);
  });
  return true;
}

async function selectApiBase() {
  if (await isHealthy(localApi)) return localApi;
  return remoteApi || localApi;
}

async function selectApiBaseWithStartupWait(localStarted) {
  if (!localStarted) return selectApiBase();
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (await isHealthy(localApi)) return localApi;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return remoteApi || localApi;
}

async function createWindow() {
  const localStarted = startLocalBackend();
  const apiBase = await selectApiBaseWithStartupWait(localStarted);
  const window = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 700,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
      additionalArguments: [`--meetasr-api-base=${apiBase}`],
    },
  });

  await window.loadURL(frontendUrl);
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (localBackend && !localBackend.killed) localBackend.kill();
  if (process.platform !== "darwin") app.quit();
});
app.on("before-quit", () => {
  if (localBackend && !localBackend.killed) localBackend.kill();
});
