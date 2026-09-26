/* eslint-disable @typescript-eslint/no-require-imports */
const { contextBridge } = require("electron");

const apiArgument = process.argv.find((argument) =>
  argument.startsWith("--meetasr-api-base="),
);

contextBridge.exposeInMainWorld("meetasrDesktop", {
  isDesktop: true,
  apiBase: apiArgument?.slice("--meetasr-api-base=".length),
});
