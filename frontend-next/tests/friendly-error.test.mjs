import assert from "node:assert/strict";
import test from "node:test";

import { friendlyError } from "../src/lib/friendlyError.ts";

function apiError(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}

test("API statuses become friendly Vietnamese without jargon", () => {
  assert.equal(friendlyError(apiError(401, "Token đã hết hạn")).title, "Phiên đăng nhập đã hết hạn");
  assert.equal(friendlyError(apiError(404, "Not Found")).title, "Không tìm thấy nội dung");
  const server = friendlyError(apiError(500, "Internal Server Error: Traceback..."));
  assert.equal(server.title, "Máy chủ đang gặp sự cố");
  assert.ok(!/traceback|internal/i.test(server.message));
});

test("user-facing backend details (Vietnamese) are kept, technical ones hidden", () => {
  const limit = friendlyError(apiError(503, "Chức năng tạo tài liệu bằng AI đã đạt giới hạn sử dụng."));
  assert.match(limit.message, /đạt giới hạn/);
  const technical = friendlyError(apiError(400, "Unsupported preset 'x' for format 'docx'"));
  assert.ok(!/Unsupported/.test(technical.message));
});

test("network and browser media errors", () => {
  assert.equal(friendlyError(new TypeError("Failed to fetch")).title, "Không kết nối được tới máy chủ");
  const denied = new Error("Permission denied");
  denied.name = "NotAllowedError";
  assert.equal(friendlyError(denied).title, "Chưa được cấp quyền ghi âm");
});

test("plain messages: Vietnamese kept, English replaced", () => {
  assert.equal(friendlyError("Tệp vượt quá giới hạn 2 GB.").message, "Tệp vượt quá giới hạn 2 GB.");
  assert.ok(!/undefined/.test(friendlyError(new Error("Cannot read properties of undefined")).message));
});
