import { afterEach, describe, expect, it, vi } from "vitest";

import { TranscriptionApiError, transcribeAudio } from "./transcription";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function successPayload() {
  return {
    text: "नमस्ते",
    provider: "sarvam",
    latency_ms: 120,
    duration_seconds: 1.4,
    language: "hi-IN",
    language_confidence: 0.97,
    segments: [],
    provider_request_id: "request-1",
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("transcription API client", () => {
  it("uses a compatible filename and omits auto language", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(successPayload()));
    vi.stubGlobal("fetch", fetchMock);

    await transcribeAudio({
      audio: new Blob(["voice"], { type: "audio/webm;codecs=opus" }),
      language: "auto",
      provider: "sarvam",
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const form = init.body as FormData;
    const file = form.get("audio") as File;

    expect(url).toBe("/api/transcriptions");
    expect(init.method).toBe("POST");
    expect(file.name).toBe("recording.webm");
    expect(file.type).toBe("audio/webm;codecs=opus");
    expect(form.has("language")).toBe(false);
    expect(form.get("provider")).toBe("sarvam");
  });

  it("uses an m4a filename for the recorder's MP4 fallback", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(successPayload()));
    vi.stubGlobal("fetch", fetchMock);

    await transcribeAudio({
      audio: new Blob(["voice"], { type: "audio/mp4" }),
      language: "hi",
      provider: "sarvam",
    });

    const form = fetchMock.mock.calls[0]?.[1]?.body as FormData;
    expect((form.get("audio") as File).name).toBe("recording.m4a");
    expect(form.get("language")).toBe("hi");
  });

  it("preserves normalized backend errors for retry decisions", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: "provider_rate_limited",
            message: "The provider rate limit was reached.",
            retryable: true,
          },
        },
        429,
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const request = transcribeAudio({
      audio: new Blob(["voice"], { type: "audio/webm" }),
      language: "auto",
      provider: "sarvam",
    });

    await expect(request).rejects.toMatchObject<Partial<TranscriptionApiError>>({
      name: "TranscriptionApiError",
      code: "provider_rate_limited",
      message: "The provider rate limit was reached.",
      retryable: true,
      status: 429,
    });
  });

  it("rejects malformed success responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ text: "Incomplete" })));

    const request = transcribeAudio({
      audio: new Blob(["voice"], { type: "audio/webm" }),
      language: "auto",
      provider: "sarvam",
    });

    await expect(request).rejects.toMatchObject<Partial<TranscriptionApiError>>({
      code: "invalid_response",
      retryable: false,
      status: 200,
    });
  });

  it("offers an explicit retry after a temporary non-JSON gateway failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Bad gateway", { status: 502 })));

    await expect(transcribeAudio({
      audio: new Blob(["voice"], { type: "audio/webm" }), language: "auto",
    })).rejects.toMatchObject<Partial<TranscriptionApiError>>({
      code: "http_502", retryable: true, status: 502,
    });
  });
});
