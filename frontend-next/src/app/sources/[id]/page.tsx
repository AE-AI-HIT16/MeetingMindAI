import Link from "next/link";
import { notFound } from "next/navigation";
import { getServerSession } from "next-auth";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { APIError, getDocument, getSource } from "@/lib/api";
import { ProcessingView } from "@/components/ProcessingView";
import { DocumentView } from "@/components/DocumentView";
import type { DocumentData, Source } from "@/lib/types";

// Luon render dong — can session va backend.
export const dynamic = "force-dynamic";

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

  const session = await getServerSession(authOptions);
  const token = session?.accessToken;

  try {
    source = await getSource(id, token);
  } catch (error) {
    if (error instanceof APIError && error.status === 404) {
      notFound();
    }
    if (error instanceof APIError && (error.status === 401 || error.status === 403)) {
      return <AccessProblem status={error.status} />;
    }
    throw error;
  }

  if (view === "doc" && documentId) {
    let document: DocumentData;
    try {
      document = await getDocument(documentId, token);
    } catch (error) {
      if (error instanceof APIError && error.status === 404) notFound();
      if (error instanceof APIError && (error.status === 401 || error.status === 403)) {
        return <AccessProblem status={error.status} />;
      }
      throw error;
    }
    if (document.sourceId !== source.id) notFound();
    return <DocumentView source={source} document={document} />;
  }

  return <ProcessingView source={source} jobId={jobId ?? source.jobId} />;
}

/** Expired login (401) or someone else's file (403): explain instead of crashing. */
function AccessProblem({ status }: { status: number }) {
  const expired = status === 401;
  return (
    <div className="mx-auto flex min-h-[60vh] max-w-md flex-col items-center justify-center px-6 text-center">
      <h1 className="font-display text-xl font-semibold text-ink">
        {expired ? "Phiên đăng nhập đã hết hạn" : "Bạn không có quyền xem tài liệu này"}
      </h1>
      <p className="mt-2 text-sm text-ink-soft">
        {expired
          ? "Vui lòng đăng nhập lại để xem các tài liệu đã lưu của bạn."
          : "Tài liệu thuộc về một tài khoản khác. Hãy đăng nhập đúng tài khoản đã tạo tài liệu."}
      </p>
      <div className="mt-5 flex gap-3">
        <Link href="/login" className="rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white hover:bg-brand-ink">
          Đăng nhập
        </Link>
        <Link href="/" className="rounded-xl border border-line px-4 py-2.5 text-sm font-medium text-ink-soft hover:border-brand">
          Về thư viện
        </Link>
      </div>
    </div>
  );
}
