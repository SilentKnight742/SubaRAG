export interface Citation {
  document_id: string;
  text: string;
  source: string | null;
}

export interface QueryResult {
  request_id: string;
  status: "unavailable" | "answered" | "no_answer";
  question: string;
  answer: string | null;
  citations: Citation[];
  message: string | null;
  latency_ms: number;
}

export interface Question {
  question: string;
  language: string | null;
}

export class QueryApiError extends Error {
  readonly retryable: boolean;

  constructor(message: string, retryable: boolean) {
    super(message);
    this.name = "QueryApiError";
    this.retryable = retryable;
  }
}

function isQueryResult(payload: unknown): payload is QueryResult {
  if (typeof payload !== "object" || payload === null) return false;
  const value = payload as Partial<QueryResult>;
  return (
    typeof value.request_id === "string" &&
    ["unavailable", "answered", "no_answer"].includes(value.status ?? "") &&
    typeof value.question === "string" &&
    (value.answer === null || typeof value.answer === "string") &&
    (value.message === null || typeof value.message === "string") &&
    typeof value.latency_ms === "number" &&
    Number.isFinite(value.latency_ms) &&
    Array.isArray(value.citations) &&
    value.citations.every((citation: unknown) => {
      if (typeof citation !== "object" || citation === null) return false;
      const item = citation as Partial<Citation>;
      return (
        typeof item.document_id === "string" &&
        typeof item.text === "string" &&
        (item.source === null || typeof item.source === "string")
      );
    }) &&
    (value.status === "answered"
      ? typeof value.answer === "string" && value.answer.trim().length > 0
      : value.answer === null && value.citations.length === 0)
  );
}

export async function askQuestion(
  question: Question,
  signal?: AbortSignal,
): Promise<QueryResult> {
  const response = await fetch("/api/queries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(question),
    signal,
  });
  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = (payload as { error?: { message?: unknown; retryable?: unknown } } | null)?.error;
    throw new QueryApiError(
      typeof detail?.message === "string"
        ? detail.message
        : "Your question could not be sent. Please try again.",
      typeof detail?.retryable === "boolean" ? detail.retryable : response.status >= 500,
    );
  }

  if (!isQueryResult(payload)) {
    throw new QueryApiError("The answer service returned an invalid response.", false);
  }

  return payload;
}
