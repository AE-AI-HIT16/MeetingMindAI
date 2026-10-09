/* eslint-disable @typescript-eslint/no-require-imports */
const { contextBridge, ipcRenderer } = require("electron");

const apiArgument = process.argv.find((a) =>
  a.startsWith("--meetasr-api-base="),
);

contextBridge.exposeInMainWorld("meetasrDesktop", {
  isDesktop: true,
  apiBase: apiArgument?.slice("--meetasr-api-base=".length),

  // GPU info
  getGpuInfo: () => ipcRenderer.invoke("get-gpu-info"),

  // Settings (persisted in userData/settings.json)
  getSettings: () => ipcRenderer.invoke("get-settings"),
  saveSettings: (data) => ipcRenderer.invoke("save-settings", data),

  // Log file path
  getLogPath: () => ipcRenderer.invoke("get-log-path"),
});
