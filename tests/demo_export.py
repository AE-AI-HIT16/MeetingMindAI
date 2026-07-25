import argparse
from pathlib import Path
from typing import cast

from meetasr.export import create_export_service
from meetasr.export.docx_exporter import DocxTemplateName, export_docx
from meetasr.export.pdf_exporter import PdfTemplateName, export_pdf


markdown = (
    Path(__file__).resolve().parent
    / "data"
    / "pdf_blue_modern_demo.md"
).read_text(encoding="utf-8")


def main(
    *,
    docx_template: DocxTemplateName,
    pdf_template: PdfTemplateName,
    author: str,
) -> None:
    service = create_export_service()
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    for export_format in ("docx", "pdf"):
        if export_format == "docx":
            artifact = export_docx(
                markdown=markdown,
                title="Báo cáo cuộc họp",
                template=docx_template,
                context={"author": author} if author else None,
            )
        elif export_format == "pdf":
            artifact = export_pdf(
                markdown=markdown,
                title="Báo cáo cuộc họp",
                template=pdf_template,
                context={"author": author} if author else None,
            )
        else:
            artifact = service.export(
                markdown=markdown,
                format=export_format,
                title="Báo cáo cuộc họp",
            )

        output_path = output_directory / artifact.filename
        output_path.write_bytes(artifact.content)

        print(f"Đã tạo: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo DOCX/PDF mẫu.")
    parser.add_argument(
        "--template",
        choices=("minimal", "modern"),
        default="minimal",
        help="Template chỉ áp dụng cho DOCX (mặc định: minimal).",
    )
    parser.add_argument(
        "--author",
        default="",
        help="Điền {{ author }} nếu template DOCX có sử dụng.",
    )
    parser.add_argument(
        "--pdf-template",
        choices=("minimal", "blue_modern"),
        default="minimal",
        help="Template áp dụng cho PDF (mặc định: minimal).",
    )
    args = parser.parse_args()
    main(
        docx_template=cast(DocxTemplateName, args.template),
        pdf_template=cast(PdfTemplateName, args.pdf_template),
        author=args.author,
    )
