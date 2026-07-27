import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Render persisted/streamed Markdown as safe semantic HTML. */
export function MarkdownLite({ text }: { text: string }) {
  return (
    <div className="markdown-preview">
      <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>
        {text}
      </ReactMarkdown>
    </div>
  );
}
