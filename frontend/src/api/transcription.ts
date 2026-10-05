export type TranscriptionProvider = "sarvam" | "elevenlabs";
export type ProviderSelection = TranscriptionProvider | "compare";

export interface TranscriptSegment {
  text: string;
  start_seconds: number;
  end_seconds: number;
  confidence: number | null;
  speaker: string | null;
}

export interface TranscriptionResult {
  text: string;
  provider: TranscriptionProvider;
  latency_ms: number;
  duration_seconds: number;
  language: string | null;
  language_confidence: number | null;
  segments: TranscriptSegment[];
  provider_request_id: string | null;
}

interface TranscriptionErrorPayload {
  error?: {
    code?: unknown;
    message?: unknown;
    retryable?: unknown;
  };
}

export class TranscriptionApiError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly status: number;

  constructor(
    message: string,
    {
      code,
      retryable,
      status,
    }: {
      code: string;
      retryable: boolean;
      status: number;
    },
  ) {
    super(message);
    this.name = "TranscriptionApiError";
    this.code = code;
    this.retryable = retryable;
    this.status = status;
  }
}

interface TranscriptionRequest {
  audio: Blob;
  language: string;
  provider?: TranscriptionProvider;
  signal?: AbortSignal;
}

function filenameFor(audio: Blob) {
  const contentType = audio.type.split(";", 1)[0]?.trim().toLowerCase();
  return contentType === "audio/mp4" ? "recording.m4a" : "recording.webm";
}

function isTranscriptionResult(payload: unknown): payload is TranscriptionResult {
  if (typeof payload !== "object" || payload === null) {
    return false;
  }

  const candidate = payload as Partial<TranscriptionResult>;
  return (
    typeof candidate.text === "string" &&
    (candidate.provider === "sarvam" || candidate.provider === "elevenlabs") &&
    typeof candidate.latency_ms === "number" &&
    typeof candidate.duration_seconds === "number" &&
    Array.isArray(candidate.segments)
  );
}

async function responsePayload(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function transcribeAudio({
  audio,
  language,
  provider,
  signal,
}: TranscriptionRequest): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("audio", audio, filenameFor(audio));
  if (language !== "auto") {
    form.append("language", language);
  }
  if (provider) {
    form.append("provider", provider);
  }

  const response = await fetch("/api/transcriptions", {
    method: "POST",
    body: form,
    signal,
  });
  const payload = await responsePayload(response);

  if (!response.ok) {
    const errorPayload = payload as TranscriptionErrorPayload | null;
    const detail = errorPayload?.error;
    throw new TranscriptionApiError(
      typeof detail?.message === "string"
        ? detail.message
        : "Transcription could not be completed.",
      {
        code: typeof detail?.code === "string" ? detail.code : `http_${response.status}`,
        retryable: detail?.retryable === true,
        status: response.status,
      },
    );
  }

  if (!isTranscriptionResult(payload)) {
    throw new TranscriptionApiError("The transcription service returned an invalid response.", {
      code: "invalid_response",
      retryable: false,
      status: response.status,
    });
  }

  return payload;
}
