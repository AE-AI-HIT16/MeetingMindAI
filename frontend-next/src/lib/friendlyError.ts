export interface FriendlyError {
  title: string;
  message: string;
  /** Trying again may help (network blip, server busy, restart…). */
  retryable: boolean;
}

/** Backend `detail` texts written for users are Vietnamese; English ones are technical. */
const VIETNAMESE = /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]/i;

function isApiError(error: unknown): error is Error & { status: number } {
  return error instanceof Error && typeof (error as { status?: unknown }).status === "number";
}

function isUserFacing(text: string | undefined): text is string {
  return Boolean(text && VIETNAMESE.test(text) && text.length < 300);
}

/**
 * Turn any thrown value into a short, friendly Vietnamese explanation.
 * Never returns stack traces, HTTP jargon or English library messages.
 */
export function friendlyError(error: unknown, fallbackTitle = "Đã có lỗi xảy ra"): FriendlyError {
  // APIError (lib/api.ts) — duck-typed so this module has no runtime deps.
  if (isApiError(error)) {
    const detail = isUserFacing(error.message) ? error.message : undefined;
    switch (error.status) {
      case 401:
        return {
          title: "Phiên đăng nhập đã hết hạn",
          message: "Vui lòng đăng nhập lại để tiếp tục.",
          retryable: false,
        };
      case 403:
        return {
          title: "Bạn không có quyền thực hiện thao tác này",
          message: detail ?? "Nội dung này thuộc về một tài khoản khác.",
          retryable: false,
        };
      case 404:
        return {
          title: "Không tìm thấy nội dung",
          message: detail ?? "Có thể nội dung đã bị xóa hoặc đường dẫn không còn đúng.",
          retryable: false,
        };
      case 409:
        return {
          title: "Chưa thể thực hiện lúc này",
          message: detail ?? "Dữ liệu đang được xử lý. Vui lòng đợi một chút rồi thử lại.",
          retryable: true,
        };
      case 413:
        return {
          title: "Tệp quá lớn",
          message: "Vui lòng chọn tệp nhỏ hơn hoặc cắt bớt nội dung.",
          retryable: false,
        };
      case 429:
      case 503:
        return {
          title: "Hệ thống đang bận",
          message: detail ?? "Vui lòng đợi khoảng một phút rồi thử lại.",
          retryable: true,
        };
      default:
        if (error.status >= 500) {
          return {
            title: "Máy chủ đang gặp sự cố",
            message: "Chúng tôi đã ghi nhận lỗi. Vui lòng thử lại sau ít phút.",
            retryable: true,
          };
        }
        return {
          title: fallbackTitle,
          message: detail ?? "Yêu cầu không hợp lệ. Vui lòng kiểm tra lại và thử lại.",
          retryable: false,
        };
    }
  }

  // Browser media errors (getUserMedia / getDisplayMedia).
  const name = error instanceof Error || error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      title: "Chưa được cấp quyền ghi âm",
      message: "Hãy cho phép trình duyệt dùng micro (hoặc chia sẻ tab) rồi thử lại.",
      retryable: true,
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return {
      title: "Không tìm thấy micro",
      message: "Hãy cắm hoặc bật micro, kiểm tra cài đặt âm thanh rồi thử lại.",
      retryable: true,
    };
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return {
      title: "Micro đang bận",
      message: "Một ứng dụng khác có thể đang dùng micro. Hãy đóng ứng dụng đó rồi thử lại.",
      retryable: true,
    };
  }

  const text = error instanceof Error ? error.message : typeof error === "string" ? error : "";
  // fetch() network failures: "Failed to fetch", "NetworkError…", "Load failed", "fetch failed"
  if (error instanceof TypeError || /failed to fetch|networkerror|load failed|fetch failed/i.test(text)) {
    return {
      title: "Không kết nối được tới máy chủ",
      message: "Vui lòng kiểm tra kết nối mạng rồi thử lại.",
      retryable: true,
    };
  }
  if (isUserFacing(text)) {
    return { title: fallbackTitle, message: text, retryable: true };
  }
  return {
    title: fallbackTitle,
    message: "Vui lòng thử lại. Nếu lỗi vẫn tiếp diễn, hãy tải lại trang.",
    retryable: true,
  };
}
