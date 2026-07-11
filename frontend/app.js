(function () {
  const STORAGE_KEY = "meetasr:selectedFile";

  function getPageName() {
    return window.location.pathname.split("/").pop() || "1_dashboard.html";
  }

  function resolveApiUrl(path) {
    const base = (window.API_BASE || "").replace(/\/$/, "");
    return `${base}${path}`;
  }

  function setFeedback(element, message, type = "info") {
    if (!element) return;
    element.textContent = message;
    element.className = `feedback ${type}`;
  }

  function readStoredFile() {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (error) {
      console.error("Cannot parse stored file", error);
      return null;
    }
  }

  function saveSelectedFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const payload = {
          name: file.name,
          type: file.type || "application/octet-stream",
          size: file.size,
          dataUrl: reader.result,
        };
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        resolve(payload);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function createFileFromStoredData(fileMeta) {
    return fetch(fileMeta.dataUrl)
      .then((response) => response.blob())
      .then(
        (blob) =>
          new File([blob], fileMeta.name, {
            type: fileMeta.type || "audio/mpeg",
            lastModified: Date.now(),
          })
      );
  }

  function formatValue(value) {
    if (Array.isArray(value)) {
      return value.join("\n");
    }
    if (typeof value === "object" && value !== null) {
      return JSON.stringify(value, null, 2);
    }
    return String(value ?? "Không có dữ liệu");
  }

  function bindDashboardPage() {
    const addCard = document.getElementById("add-new-card");
    if (addCard) {
      addCard.addEventListener("click", () => {
        window.location.href = "2_upload.html";
      });
    }
  }

  function bindUploadPage() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const statusNode = document.getElementById("upload-status");

    if (!dropzone || !fileInput) return;

    const handleFile = (file) => {
      if (!file) return;
      if (!file.type.startsWith("audio/")) {
        setFeedback(statusNode, "Chỉ hỗ trợ file âm thanh.", "error");
        return;
      }

      setFeedback(statusNode, `Đang chuẩn bị ${file.name}...`, "info");
      saveSelectedFile(file)
        .then(() => {
          window.location.href = "3_workspace.html";
        })
        .catch(() => {
          setFeedback(statusNode, "Không thể đọc file. Vui lòng thử lại.", "error");
        });
    };

    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("is-dragging");
    });

    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragging");
      const [file] = event.dataTransfer.files || [];
      handleFile(file);
    });

    fileInput.addEventListener("change", (event) => {
      const [file] = event.target.files || [];
      handleFile(file);
    });

    const pickButton = document.getElementById("pick-file-button");
    if (pickButton) {
      pickButton.addEventListener("click", () => {
        fileInput.click();
      });
    }
  }

  async function pollMeetingStatus(meetingId, attempt = 0) {
    const response = await fetch(resolveApiUrl(`/v1/meeting/${meetingId}/status`));
    if (!response.ok) {
      throw new Error("Không thể lấy trạng thái xử lý");
    }

    const data = await response.json();
    if (data.status === "completed") {
      return data;
    }

    if (data.status === "failed") {
      throw new Error(data.result?.error || "Xử lý thất bại");
    }

    if (attempt >= 20) {
      throw new Error("Hết thời gian chờ kết quả");
    }

    await new Promise((resolve) => setTimeout(resolve, 1500));
    return pollMeetingStatus(meetingId, attempt + 1);
  }

  function bindWorkspacePage() {
    const fileNameNode = document.getElementById("selected-file-name");
    const transcriptNode = document.getElementById("transcript-content");
    const statusNode = document.getElementById("workspace-status");
    const resultsNode = document.getElementById("analysis-results");
    const analyzeButton = document.getElementById("start-analysis");
    const optionButtons = document.querySelectorAll(".btn-pastel");

    const fileMeta = readStoredFile();
    if (!fileMeta) {
      setFeedback(statusNode, "Chưa có file âm thanh nào được chọn.", "error");
      return;
    }

    if (fileNameNode) {
      fileNameNode.textContent = fileMeta.name;
    }

    setFeedback(statusNode, "Đang chuyển âm thanh thành văn bản...", "info");

    createFileFromStoredData(fileMeta)
      .then((file) => {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", "sensevoice");
        formData.append("language", "auto");
        formData.append("response_format", "json");
        formData.append("speaker_diarization", "false");
        formData.append("timestamp_granularity", "segment");

        return fetch(resolveApiUrl("/v1/audio/transcriptions"), {
          method: "POST",
          body: formData,
        });
      })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("API chuyển đổi giọng nói thất bại");
        }

        const contentType = response.headers.get("content-type") || "";
        let text = "";
        if (contentType.includes("application/json")) {
          const payload = await response.json();
          text = payload.text || payload.transcript || JSON.stringify(payload, null, 2);
        } else {
          text = await response.text();
        }

        if (transcriptNode) {
          transcriptNode.textContent = text;
        }
        setFeedback(statusNode, "Đã nhận được bản transcript.", "success");
      })
      .catch((error) => {
        console.error(error);
        if (transcriptNode) {
          transcriptNode.textContent = "Không thể chuyển đổi âm thanh thành văn bản.";
        }
        setFeedback(statusNode, error.message || "Có lỗi khi gọi API.", "error");
      });

    optionButtons.forEach((button) => {
      button.addEventListener("click", () => {
        button.classList.toggle("is-selected");
      });
    });

    if (analyzeButton) {
      analyzeButton.addEventListener("click", async () => {
        const selectedOptions = Array.from(optionButtons)
          .filter((button) => button.classList.contains("is-selected"))
          .map((button) => button.dataset.option);

        if (!selectedOptions.length) {
          setFeedback(statusNode, "Vui lòng chọn ít nhất một mục phân tích.", "error");
          return;
        }

        if (!fileMeta) {
          setFeedback(statusNode, "Không có file để phân tích.", "error");
          return;
        }

        setFeedback(statusNode, "Đang gửi yêu cầu phân tích...", "info");
        try {
          const file = await createFileFromStoredData(fileMeta);
          const formData = new FormData();
          formData.append("file", file);
          formData.append("language", "vi");
          formData.append("llm_model", "gpt-4o-mini");
          formData.append("asr_model", "sensevoice");
          formData.append("include_transcript", "false");
          formData.append("include_topics", selectedOptions.includes("topic") ? "true" : "false");
          formData.append("include_actions", selectedOptions.includes("actions") ? "true" : "false");
          formData.append("include_decisions", selectedOptions.includes("decisions") ? "true" : "false");

          const response = await fetch(resolveApiUrl("/v1/meeting/summarize"), {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error("API phân tích không phản hồi đúng");
          }

          const payload = await response.json();
          if (!payload.meeting_id) {
            throw new Error("Không nhận được meeting_id");
          }

          setFeedback(statusNode, "Đang chờ kết quả từ server...", "info");
          const resultPayload = await pollMeetingStatus(payload.meeting_id);
          const resultData = resultPayload.result || {};

          const sections = [];
          if (selectedOptions.includes("summary") || selectedOptions.length) {
            sections.push({
              title: "Summary",
              value: resultData.summary || resultData.text || resultData.transcript || "Không có summary",
            });
          }
          if (selectedOptions.includes("topic")) {
            sections.push({
              title: "Topics",
              value: resultData.topics || resultData.key_topics || "Không có chủ đề",
            });
          }
          if (selectedOptions.includes("actions")) {
            sections.push({
              title: "Action Items",
              value: resultData.action_items || resultData.actions || "Không có action items",
            });
          }
          if (selectedOptions.includes("decisions")) {
            sections.push({
              title: "Decisions",
              value: resultData.decisions || "Không có quyết định",
            });
          }

          if (resultsNode) {
            resultsNode.innerHTML = sections.length
              ? sections
                  .map(
                    (section) => `
                      <div class="result-card">
                        <h4>${section.title}</h4>
                        <pre>${formatValue(section.value)}</pre>
                      </div>
                    `
                  )
                  .join("")
              : `<div class="result-card"><pre>${formatValue(resultData)}</pre></div>`;
          }

          setFeedback(statusNode, "Phân tích hoàn tất.", "success");
        } catch (error) {
          console.error(error);
          setFeedback(statusNode, error.message || "Có lỗi khi phân tích.", "error");
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const pageName = getPageName();
    if (pageName === "1_dashboard.html") {
      bindDashboardPage();
    } else if (pageName === "2_upload.html") {
      bindUploadPage();
    } else if (pageName === "3_workspace.html") {
      bindWorkspacePage();
    }
  });
})();
