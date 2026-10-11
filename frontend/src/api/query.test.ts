import { afterEach, describe, expect, it, vi } from "vitest";

import { askQuestion } from "./query";

const unavailable = {
  request_id: "query-1", status: "unavailable", question: "What is a monsoon?",
  answer: null, citations: [], message: "Your question is ready. Answers are coming soon.", latency_ms: 0.1,
};

function response(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("query API client", () => {
  it("sends the question and nullable language using the cancellable JSON contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response(unavailable));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();
    const question = { question: unavailable.question, language: null };
    await expect(askQuestion(question, controller.signal)).resolves.toEqual(unavailable);
    expect(fetchMock).toHaveBeenCalledWith("/api/queries", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(question), signal: controller.signal,
    });
  });

  it.each([
    { ...unavailable, answer: "An invented answer" },
    { ...unavailable, citations: [{ document_id: "fake", text: "Invented evidence", source: null }] },
    { ...unavailable, status: "answered", answer: " " },
    { ...unavailable, status: "answered", answer: "An answer", citations: [null] },
  ])("rejects inconsistent or malformed answer responses", async (payload) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(payload)));
    await expect(askQuestion({ question: unavailable.question, language: null })).rejects.toMatchObject({
      name: "QueryApiError", retryable: false, message: "The answer service returned an invalid response.",
    });
  });

  it("preserves nonretryable backend validation errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({
      error: { code: "invalid_request", message: "The question must contain text.", retryable: false },
    }, 422)));
    await expect(askQuestion({ question: "", language: null })).rejects.toMatchObject({
      message: "The question must contain text.", retryable: false,
    });
  });

  it("offers retry for a temporary gateway failure with no JSON body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Bad gateway", { status: 502 })));
    await expect(askQuestion({ question: unavailable.question, language: null })).rejects.toMatchObject({
      name: "QueryApiError", retryable: true,
    });
  });
});
