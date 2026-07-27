import { notFound } from "next/navigation";
import { APIError, getDocument, getSource } from "@/lib/api";
import { ProcessingView } from "@/components/ProcessingView";
import { DocumentView } from "@/components/DocumentView";
import type { DocumentData, Source } from "@/lib/types";

export default async function SourcePage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{
    view?: string;
    jobId?: string;
    documentId?: string;
  }>;
}) {
  const { id } = await params;
  const { view, jobId, documentId } = await searchParams;
  let source: Source;

  try {
    source = await getSource(id);
  } catch (error) {
    if (error instanceof APIError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  if (view === "doc" && documentId) {
    let document: DocumentData;
    try {
      document = await getDocument(documentId);
    } catch (error) {
      if (error instanceof APIError && error.status === 404) notFound();
      throw error;
    }
    if (document.sourceId !== source.id) notFound();
    return <DocumentView source={source} document={document} />;
  }

  return <ProcessingView source={source} jobId={jobId ?? source.jobId} />;
}
