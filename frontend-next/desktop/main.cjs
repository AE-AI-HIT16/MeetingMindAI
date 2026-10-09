/* eslint-disable @typescript-eslint/no-require-imports */
"use strict";

// ─── Bước 10: Logging phải setup TRƯỚC mọi thứ khác ──────────────────────────
const log = require("electron-log");
log.transports.file.level = "info";
log.transports.console.level = "debug";
log.transports.file.maxSize = 5 * 1024 * 1024; // 5MB rotate
console.log = log.log.bind(log);
console.error = log.error.bind(log);
console.warn = log.warn.bind(log);

// ─── Bước 10: Catch lỗi process sớm nhất có thể ────────────────────────────
process.on("uncaughtException", (err) => {
  log.error("[main] uncaughtException:", err);
});

const { app, BrowserWindow, dialog, ipcMain, Menu, Tray, nativeImage } =
  require("electron");
const { autoUpdater } = require("electron-updater");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const net = require("node:net");

// ─── Config ──────────────────────────────────────────────────────────────────

const BACKEND_PORT = 8000;
const FRONTEND_PORT = 3000;
const BACKEND_RETRIES = 40;   // 20 s
const FRONTEND_RETRIES = 30;  // 15 s
const localApi = `http://127.0.0.1:${BACKEND_PORT}`;
const remoteApi =
  process.env.MEETASR_REMOTE_API || process.env.NEXT_PUBLIC_MEETASR_API;
const isPackaged = app.isPackaged;

const workspaceRoot = isPackaged
  ? path.join(process.resourcesPath, "app")
  : path.resolve(__dirname, "..", "..");

const SETTINGS_PATH = path.join(app.getPath("userData"), "settings.json");

let backendProcess = null;
let frontendProcess = null;
let mainWindow = null;
let tray = null;

log.info("[main] Starting MeetingMind AI Desktop", {
  version: app.getVersion(),
  isPackaged,
  workspaceRoot,
  logFile: log.transports.file.getFile()?.path,
});

// ─── Settings (GPU preference, theme…) ───────────────────────────────────────

function loadSettings() {
  try {
    if (fs.existsSync(SETTINGS_PATH)) {
      return JSON.parse(fs.readFileSync(SETTINGS_PATH, "utf8"));
    }
  } catch (e) {
    log.warn("[settings] Failed to load:", e.message);
  }
  return {};
}

function saveSettings(data) {
  try {
    const current = loadSettings();
    fs.writeFileSync(SETTINGS_PATH, JSON.stringify({ ...current, ...data }, null, 2));
  } catch (e) {
    log.warn("[settings] Failed to save:", e.message);
  }
}

// ─── 1.4 GPU Detection ───────────────────────────────────────────────────────

async function detectGpu(python) {
  return new Promise((resolve) => {
    const script = `
import json
try:
    import torch
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "device": f"cuda:{i}", "name": p.name,
                         "vram_gb": round(p.total_memory / 1024**3, 1)})
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpus.append({"index": -1, "device": "mps", "name": "Apple Silicon", "vram_gb": None})
    gpus.append({"index": -2, "device": "cpu", "name": "CPU (no GPU)", "vram_gb": None})
    print(json.dumps(gpus))
except Exception as e:
    print(json.dumps([{"index": -2, "device": "cpu", "name": "CPU only", "vram_gb": None}]))
`;
    const proc = spawn(python, ["-c", script], {
      env: process.env,
      stdio: ["ignore", "pipe", "ignore"],
    });
    let output = "";
    proc.stdout.on("data", (d) => { output += d.toString(); });
    proc.on("close", () => {
      try {
        resolve(JSON.parse(output.trim()));
      } catch {
        resolve([{ index: -2, device: "cpu", name: "CPU only", vram_gb: null }]);
      }
    });
    proc.on("error", () => resolve([{ index: -2, device: "cpu", name: "CPU only", vram_gb: null }]));
    setTimeout(() => {
      proc.kill();
      resolve([{ index: -2, device: "cpu", name: "CPU only", vram_gb: null }]);
    }, 12000);
  });
}

// ─── 11: GPU Selection Dialog ────────────────────────────────────────────────

async function promptGpuSelection(gpus) {
  if (gpus.length <= 1) return gpus[0]; // Chỉ có CPU — không cần hỏi

  const buttons = gpus.map((g) =>
    g.vram_gb ? `${g.name} (${g.vram_gb} GB VRAM)` : g.name
  );
  const { response } = await dialog.showMessageBox({
    type: "question",
    title: "Chọn thiết bị xử lý",
    message: "MeetingMind AI phát hiện nhiều thiết bị. Bạn muốn dùng GPU nào?",
    detail: "Lựa chọn sẽ được lưu và dùng lại lần sau.\nCần ≥ 4 GB VRAM cho Qwen3-ASR.",
    buttons,
    defaultId: 0,
    cancelId: gpus.length - 1, // CPU là lựa chọn cuối
  });
  return gpus[response];
}

async function resolveGpuDevice(python) {
  const settings = loadSettings();

  // Dùng preference đã lưu nếu có (trừ khi user giữ Shift để mở lại dialog)
  if (settings.gpuDevice && !process.env.MEETASR_RESET_GPU) {
    log.info("[gpu] Using saved device:", settings.gpuDevice);
    return settings.gpuDevice;
  }

  const gpus = await detectGpu(python);
  log.info("[gpu] Detected:", gpus);

  const chosen = await promptGpuSelection(gpus);
  log.info("[gpu] Selected:", chosen);

  saveSettings({ gpuDevice: chosen.device, gpuName: chosen.name });
  return chosen.device;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function resolveVirtualEnvPython() {
  if (process.env.MEETASR_PYTHON && fs.existsSync(process.env.MEETASR_PYTHON)) {
    return process.env.MEETASR_PYTHON;
  }
  const candidates =
    process.platform === "win32"
      ? [
          path.join(workspaceRoot, ".venv", "Scripts", "python.exe"),
          path.join(workspaceRoot, "venv", "Scripts", "python.exe"),
          path.join(workspaceRoot, "python_env", "python.exe"),
        ]
      : [
          path.join(workspaceRoot, ".venv", "bin", "python"),
          path.join(workspaceRoot, "venv", "bin", "python"),
          path.join(workspaceRoot, "python_env", "bin", "python"),
          path.join(workspaceRoot, "python_env", "python"),
        ];
  return candidates.find((c) => fs.existsSync(c)) || null;
}

async function isPortOpen(host, port) {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    sock.setTimeout(500);
    sock.on("connect", () => { sock.destroy(); resolve(true); });
    sock.on("timeout", () => { sock.destroy(); resolve(false); });
    sock.on("error", () => resolve(false));
    sock.connect(port, host);
  });
}

async function waitForPort(host, port, retries, label) {
  for (let i = 0; i < retries; i++) {
    if (await isPortOpen(host, port)) return true;
    await new Promise((r) => setTimeout(r, 500));
    if (i % 10 === 9)
      log.info(`[desktop] waiting for ${label}… ${i + 1}/${retries}`);
  }
  return false;
}

async function isHealthy(url) {
  for (const ep of ["/v1/health", "/health"]) {
    try {
      const res = await fetch(`${url}${ep}`, {
        signal: AbortSignal.timeout(1500),
      });
      if (res.ok) return true;
    } catch { /* try next */ }
  }
  return false;
}

// ─── 1.2 Spawn FastAPI backend ───────────────────────────────────────────────

async function startBackend() {
  if (process.env.MEETASR_DESKTOP_LOCAL_BACKEND === "false") {
    log.info("[backend] Disabled via env.");
    return false;
  }

  if (await isPortOpen("127.0.0.1", BACKEND_PORT)) {
    log.info("[backend] Already running on port", BACKEND_PORT);
    return true;
  }

  const python = resolveVirtualEnvPython();
  if (!python) {
    dialog.showErrorBox(
      "MeetingMind — Thiếu Python",
      `Không tìm thấy Python virtual environment.\n\nHãy tạo .venv:\n  cd "${workspaceRoot}"\n  python -m venv .venv\n  .venv/bin/pip install -e .[gpu]`,
    );
    return false;
  }

  // GPU selection (bước 11 + 1.4)
  const gpuDevice = await resolveGpuDevice(python);
  log.info("[backend] GPU device:", gpuDevice);

  const configPath = path.join(workspaceRoot, "meeting_config.yaml");

  backendProcess = spawn(
    python,
    ["-m", "uvicorn", "meetasr.api.app:app",
     "--host", "127.0.0.1",
     "--port", String(BACKEND_PORT),
     "--workers", "1"],
    {
      cwd: workspaceRoot,
      env: {
        ...process.env,
        MEETASR_DESKTOP: "1",
        MEETASR_DEVICE: gpuDevice,
        MEETASR_CONFIG: configPath,
        // 12: Xoá RunPod khỏi desktop — không để RUNPOD_URL ảnh hưởng
        RUNPOD_URL: "",
        RUNPOD_ENDPOINT_ID: "",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );

  // Pipe backend logs vào electron-log
  backendProcess.stdout.on("data", (d) =>
    log.info("[backend]", d.toString().trimEnd()),
  );
  backendProcess.stderr.on("data", (d) =>
    log.warn("[backend:err]", d.toString().trimEnd()),
  );

  backendProcess.on("error", (err) =>
    log.error("[backend] spawn error:", err.message),
  );
  backendProcess.on("exit", (code, signal) => {
    if (code !== 0 && code !== null)
      log.error(`[backend] exited code=${code} signal=${signal}`);
  });

  const ok = await waitForPort("127.0.0.1", BACKEND_PORT, BACKEND_RETRIES, "backend");
  if (!ok) {
    log.error("[backend] Did not become ready in time.");
    return false;
  }
  log.info("[backend] Ready on port", BACKEND_PORT);
  return true;
}

// ─── 1.3 Next.js standalone server ───────────────────────────────────────────

function startFrontendServer() {
  if (!isPackaged) return true; // Dev: dùng `pnpm dev` chạy riêng

  const serverScript = path.join(
    workspaceRoot, "frontend-next", ".next", "standalone", "server.js",
  );
  if (!fs.existsSync(serverScript)) {
    log.error("[frontend] standalone/server.js not found — run `pnpm desktop:build`");
    return false;
  }

  frontendProcess = spawn(
    process.execPath, // Node.js của Electron
    [serverScript],
    {
      cwd: path.dirname(serverScript),
      env: {
        ...process.env,
        PORT: String(FRONTEND_PORT),
        HOSTNAME: "127.0.0.1",
        MEETASR_API: localApi,
        NEXT_PUBLIC_MEETASR_API: localApi,
        NEXTAUTH_URL: `http://127.0.0.1:${FRONTEND_PORT}`,
        NEXT_TELEMETRY_DISABLED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );

  frontendProcess.stdout.on("data", (d) =>
    log.info("[frontend]", d.toString().trimEnd()),
  );
  frontendProcess.stderr.on("data", (d) =>
    log.warn("[frontend:err]", d.toString().trimEnd()),
  );
  frontendProcess.on("error", (err) =>
    log.error("[frontend] spawn error:", err.message),
  );
  log.info("[frontend] Standalone server spawned.");
  return true;
}

// ─── System Tray (bước 7 Phase 2) ────────────────────────────────────────────

function createTray() {
  // Dùng icon trống nếu chưa có file icon
  const iconPath = path.join(__dirname, "assets", "tray-icon.png");
  const icon = fs.existsSync(iconPath)
    ? nativeImage.createFromPath(iconPath).resize({ width: 16, height: 16 })
    : nativeImage.createEmpty();

  tray = new Tray(icon);
  tray.setToolTip("MeetingMind AI");

  const contextMenu = Menu.buildFromTemplate([
    {
      label: "Mở MeetingMind",
      click: () => {
        if (mainWindow) { mainWindow.show(); mainWindow.focus(); }
        else createWindow();
      },
    },
    { type: "separator" },
    {
      label: "Đổi GPU…",
      click: async () => {
        const python = resolveVirtualEnvPython();
        if (!python) return;
        saveSettings({ gpuDevice: undefined });
        dialog.showMessageBox({ type: "info", message: "Khởi động lại app để chọn GPU mới.", buttons: ["OK"] });
      },
    },
    {
      label: `Log: ${log.transports.file.getFile()?.path ?? "N/A"}`,
      enabled: false,
    },
    { type: "separator" },
    { label: "Thoát", click: () => app.quit() },
  ]);

  tray.setContextMenu(contextMenu);
  tray.on("double-click", () => {
    if (mainWindow) { mainWindow.show(); mainWindow.focus(); }
  });
}

// ─── Bước 9: Auto-updater ────────────────────────────────────────────────────

function setupAutoUpdater() {
  if (!isPackaged) return; // Không check update trong dev mode

  autoUpdater.logger = log;
  autoUpdater.autoDownload = false; // User chủ động download

  autoUpdater.on("update-available", (info) => {
    log.info("[updater] Update available:", info.version);
    dialog
      .showMessageBox(mainWindow, {
        type: "info",
        title: "Cập nhật mới",
        message: `Phiên bản ${info.version} đã sẵn sàng!`,
        detail: "Bạn có muốn tải và cài đặt ngay không?",
        buttons: ["Tải ngay", "Để sau"],
        defaultId: 0,
      })
      .then(({ response }) => {
        if (response === 0) autoUpdater.downloadUpdate();
      });
  });

  autoUpdater.on("update-downloaded", (info) => {
    log.info("[updater] Downloaded:", info.version);
    dialog
      .showMessageBox(mainWindow, {
        type: "info",
        title: "Sẵn sàng cài đặt",
        message: `Phiên bản ${info.version} đã tải xong.`,
        detail: "Khởi động lại để hoàn tất cập nhật?",
        buttons: ["Khởi động lại", "Để sau"],
        defaultId: 0,
      })
      .then(({ response }) => {
        if (response === 0) autoUpdater.quitAndInstall();
      });
  });

  autoUpdater.on("error", (err) => {
    log.error("[updater] Error:", err.message);
  });

  // Check sau 5 giây để app load xong
  setTimeout(() => {
    autoUpdater.checkForUpdates().catch((e) =>
      log.warn("[updater] checkForUpdates failed:", e.message),
    );
  }, 5000);
}

// ─── Create main window ───────────────────────────────────────────────────────

async function createWindow() {
  const backendOk = await startBackend();
  startFrontendServer();

  if (isPackaged) {
    await waitForPort("127.0.0.1", FRONTEND_PORT, FRONTEND_RETRIES, "frontend");
  }

  const apiBase =
    backendOk && (await isHealthy(localApi))
      ? localApi
      : remoteApi || localApi;

  if (!backendOk && !remoteApi) {
    log.warn("[main] No backend available — app may not work.");
  }

  log.info("[main] API base:", apiBase);

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 700,
    title: "MeetingMind AI",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
      additionalArguments: [`--meetasr-api-base=${apiBase}`],
    },
  });

  // Bước 10: Catch renderer crashes
  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    log.error("[renderer] Process gone:", details);
    dialog.showMessageBox({
      type: "error",
      title: "Lỗi giao diện",
      message: `Giao diện bị crash (${details.reason}).`,
      detail: `Log tại: ${log.transports.file.getFile()?.path}`,
      buttons: ["Tải lại", "Đóng"],
    }).then(({ response }) => {
      if (response === 0 && mainWindow) mainWindow.reload();
    });
  });

  const frontendUrl =
    process.env.MEETASR_FRONTEND_URL ||
    `http://127.0.0.1:${FRONTEND_PORT}`;
  await mainWindow.loadURL(frontendUrl).catch((err) => {
    log.error("[main] Failed to load URL:", err.message);
  });

  mainWindow.on("close", (e) => {
    // Thu vào tray thay vì đóng hẳn
    if (tray && process.platform !== "darwin") {
      e.preventDefault();
      mainWindow.hide();
    }
  });
  mainWindow.on("closed", () => { mainWindow = null; });
}

// ─── IPC handlers ─────────────────────────────────────────────────────────────

ipcMain.handle("get-gpu-info", async () => {
  const python = resolveVirtualEnvPython();
  if (!python) return [];
  return detectGpu(python);
});

ipcMain.handle("get-settings", () => loadSettings());

ipcMain.handle("save-settings", (_e, data) => {
  saveSettings(data);
  return true;
});

ipcMain.handle("get-log-path", () =>
  log.transports.file.getFile()?.path ?? null,
);

// ─── App lifecycle ────────────────────────────────────────────────────────────

app.whenReady().then(() => {
  createTray();
  createWindow();
  setupAutoUpdater();
});

function cleanup() {
  log.info("[main] Cleanup...");
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill("SIGTERM");
  }
  if (frontendProcess && !frontendProcess.killed) {
    frontendProcess.kill("SIGTERM");
  }
}

app.on("window-all-closed", () => {
  // macOS: app sống trong tray nếu có tray, còn không thì quit
  if (process.platform !== "darwin" && !tray) {
    cleanup();
    app.quit();
  }
});

app.on("before-quit", () => {
  if (tray) { tray.destroy(); tray = null; }
  cleanup();
});

app.on("activate", () => {
  if (mainWindow === null) createWindow();
  else mainWindow.show();
});
