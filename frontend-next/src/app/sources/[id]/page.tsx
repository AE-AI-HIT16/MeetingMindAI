import { notFound } from "next/navigation";
import { getSource } from "@/lib/mock";
import { ProcessingView } from "@/components/ProcessingView";
import { DocumentView } from "@/components/DocumentView";

export default async function SourcePage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ view?: string }>;
}) {
  const { id } = await params;
  const { view } = await searchParams;
  const source = getSource(id);

  if (!source) notFound();

  // Processing → live view; done (or ?view=doc after finalize) → document viewer.
  const showDoc = source.status === "done" || view === "doc";
  return showDoc ? (
    <DocumentView source={source} />
  ) : (
    <ProcessingView source={source} />
  );
}
