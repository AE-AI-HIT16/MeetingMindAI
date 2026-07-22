from pathlib import Path

from meetasr.export import create_export_service


markdown = """
# Báo cáo cuộc họp

## Tổng quan

Cuộc họp thảo luận về **tiến độ dự án** và kế hoạch triển khai.

## Công việc cần làm

- Anh Tú hoàn thiện phần export
- Nam kiểm tra frontend
- Minh triển khai API

## Phân công

| Thành viên | Công việc | Trạng thái |
|---|---|---|
| Anh Tú | Export DOCX/PDF | Đang làm |
| Nam | Frontend | Chờ API |
| Minh | API route | Chưa bắt đầu |

> Đây là tài liệu được sinh tự động từ Markdown.

Xem thêm tại [OpenAI](https://openai.com).
"""


def main() -> None:
    service = create_export_service()
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    for export_format in ("docx", "pdf"):
        artifact = service.export(
            markdown=markdown,
            format=export_format,
            title="Báo cáo cuộc họp",
        )

        output_path = output_directory / artifact.filename
        output_path.write_bytes(artifact.content)

        print(f"Đã tạo: {output_path.resolve()}")


if __name__ == "__main__":
    main()