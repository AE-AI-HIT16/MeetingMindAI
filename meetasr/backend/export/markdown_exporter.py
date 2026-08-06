from meetasr.backend.export.service import ExportArtifact


def export_markdown(markdown: str, title: str) -> ExportArtifact:
    """Export Markdown as a UTF-8 encoded .md file."""
    return ExportArtifact(
        content=markdown.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        filename=f"{title}.md"
    )
    