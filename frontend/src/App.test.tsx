import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { RECORDER_SETTINGS } from "./hooks/useAudioRecorder";

const originalMediaDevices = navigator.mediaDevices;
const fetchMock = vi.fn();
const transcript = {
  text: "What is a monsoon?", provider: "elevenlabs", latency_ms: 241,
  duration_seconds: 1, language: "en", language_confidence: 0.98,
  segments: [], provider_request_id: "transcription-request",
};

function queryResult(question = transcript.text) {
  return {
    request_id: "query-request", status: "unavailable", question, answer: null,
    citations: [], message: "Your question is ready. Answers are coming soon.", latency_ms: 0.5,
  };
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function failedResponse(retryable = true) {
  return jsonResponse({ error: { code: "service_unavailable", message: "Please try again shortly.", retryable } }, 503);
}

function formDataForCall(index: number) {
  return fetchMock.mock.calls[index]?.[1]?.body as FormData;
}

async function defaultResponse(input: RequestInfo | URL, init?: RequestInit) {
  return jsonResponse(input === "/api/transcriptions"
    ? transcript
    : queryResult(JSON.parse(init?.body as string).question));
}

beforeEach(() => {
  fetchMock.mockReset().mockImplementation(defaultResponse);
  vi.stubGlobal("fetch", fetchMock);
});

function installRecorderMock(getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] })) {
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
  class MockMediaRecorder {
    static isTypeSupported() { return true; }
    mimeType: string;
    ondataavailable: ((event: BlobEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    onstop: ((event: Event) => void) | null = null;
    state: RecordingState = "inactive";
    constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
      this.mimeType = options?.mimeType ?? "audio/webm";
    }
    start() { this.state = "recording"; }
    stop() {
      this.state = "inactive";
      this.ondataavailable?.({ data: new Blob(["voice"], { type: this.mimeType }) } as BlobEvent);
      this.onstop?.(new Event("stop"));
    }
  }
  vi.stubGlobal("MediaRecorder", MockMediaRecorder);
}

function installVoiceActivityMock(readRms: () => number) {
  const close = vi.fn().mockResolvedValue(undefined);
  const disconnectAnalyser = vi.fn();
  const disconnectSource = vi.fn();
  class MockAnalyserNode {
    fftSize = 1_024;
    smoothingTimeConstant = 0;
    disconnect = disconnectAnalyser;
    getFloatTimeDomainData(samples: Float32Array) { samples.fill(readRms()); }
  }
  class MockAudioContext {
    state: AudioContextState = "running";
    close = close;
    createAnalyser() { return new MockAnalyserNode(); }
    createMediaStreamSource() { return { connect: vi.fn(), disconnect: disconnectSource }; }
    resume() { return Promise.resolve(); }
  }
  vi.stubGlobal("AudioContext", MockAudioContext);
  return { close, disconnectAnalyser, disconnectSource };
}

async function captureQuestion() {
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));
}

function typeQuestion(question: string) {
  const disclosure = screen.getByText("Or type your question").closest("details");
  disclosure?.setAttribute("open", "");
  fireEvent.change(screen.getByLabelText("Your question"), { target: { value: question } });
  fireEvent.click(screen.getByRole("button", { name: /send question/i }));
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: originalMediaDevices });
});

describe("question flow", () => {
  it("offers a focused voice interface and a typed alternative", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /ask at the speed/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start recording" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Spoken language" })).toHaveValue("auto");
    expect(screen.queryByRole("combobox", { name: "Test provider" })).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Hindi" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Marathi" })).toBeInTheDocument();
    expect(screen.getByText("Your words will appear here.")).toBeInTheDocument();
    expect(screen.getByText(/00:30 maximum/i)).toBeInTheDocument();
    expect(screen.getByText("Or type your question")).toBeInTheDocument();
  });

  it("automatically transcribes with the server default, then submits the recognized question", async () => {
    const stopTrack = vi.fn();
    installRecorderMock(vi.fn().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] }));
    render(<App />);
    await captureQuestion();
    expect(await screen.findByText("Your question is ready. Answers are coming soon.")).toBeInTheDocument();
    expect(screen.getByText(transcript.text)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/transcriptions");
    const form = formDataForCall(0);
    expect(form.has("provider")).toBe(false);
    expect(form.has("language")).toBe(false);
    expect((form.get("audio") as File).type).toBe("audio/webm;codecs=opus");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/queries");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ question: transcript.text, language: "en" });
    expect(document.querySelector("audio")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Sources" })).not.toBeInTheDocument();
    expect(stopTrack).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByText("Ready when you are")).toBeInTheDocument();
    expect(screen.queryByText(transcript.text)).not.toBeInTheDocument();
  });

  it("forwards an explicitly selected language without a provider override", async () => {
    installRecorderMock();
    render(<App />);
    fireEvent.change(screen.getByRole("combobox", { name: "Spoken language" }), { target: { value: "mr" } });
    await captureQuestion();
    await screen.findByText("Question received.");
    expect(formDataForCall(0).get("language")).toBe("mr");
    expect(formDataForCall(0).has("provider")).toBe(false);
  });

  it("sends typed questions without microphone permission or paid transcription", async () => {
    const getUserMedia = vi.fn();
    installRecorderMock(getUserMedia);
    render(<App />);
    typeQuestion("  Why does it rain?  ");
    await screen.findByText("Question received.");
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/queries");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ question: "Why does it rain?", language: null });
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("retries a failed query without repeating successful transcription", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(transcript)).mockResolvedValueOnce(failedResponse());
    installRecorderMock();
    render(<App />);
    await captureQuestion();
    fireEvent.click(await screen.findByRole("button", { name: "Retry question" }));
    await screen.findByText("Question received.");
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(["/api/transcriptions", "/api/queries", "/api/queries"]);
    expect(screen.getByText(transcript.text)).toBeInTheDocument();
  });

  it("retries a failed transcription only after an explicit retry", async () => {
    fetchMock.mockResolvedValueOnce(failedResponse());
    installRecorderMock();
    render(<App />);
    await captureQuestion();
    const retryButton = await screen.findByRole("button", { name: "Retry transcription" });
    expect(fetchMock).toHaveBeenCalledOnce();
    fireEvent.click(retryButton);
    await screen.findByText("Question received.");
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(["/api/transcriptions", "/api/transcriptions", "/api/queries"]);
    expect((formDataForCall(0).get("audio") as File).size).toBe((formDataForCall(1).get("audio") as File).size);
  });

  it("does not retry a deterministic failure or submit empty recognized text", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ...transcript, text: "  " }));
    installRecorderMock();
    render(<App />);
    await captureQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent("No words were recognized.");
    expect(screen.queryByRole("button", { name: /retry transcription/i })).not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("keeps a nonretryable backend error actionable through new input", async () => {
    fetchMock.mockResolvedValueOnce(failedResponse(false));
    render(<App />);
    typeQuestion("Question one");
    expect(await screen.findByRole("alert")).toHaveTextContent("Please try again shortly.");
    expect(screen.queryByRole("button", { name: "Retry question" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send question/i })).toBeEnabled();
  });

  it("cancels a query and ignores a late response", async () => {
    let resolveQuery: ((response: Response) => void) | undefined;
    let signal: AbortSignal | undefined;
    fetchMock.mockImplementationOnce((_input: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((resolve) => {
      signal = init?.signal ?? undefined;
      resolveQuery = resolve;
    }));
    render(<App />);
    typeQuestion("First question");
    expect(screen.getByText("Sending your question…")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(signal?.aborted).toBe(true);
    typeQuestion("Second question");
    await screen.findByText("Question received.");
    await act(async () => { resolveQuery?.(jsonResponse(queryResult("First question"))); });
    expect(screen.getByText("Second question", { selector: "p" })).toBeInTheDocument();
    expect(screen.queryByText("First question", { selector: "p" })).not.toBeInTheDocument();
  });

  it("aborts transcription when recording a new question", async () => {
    let signal: AbortSignal | undefined;
    fetchMock.mockImplementationOnce((_input: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      signal = init?.signal ?? undefined;
      signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    }));
    installRecorderMock();
    render(<App />);
    await captureQuestion();
    await screen.findByText("Transcribing your question…");
    fireEvent.click(screen.getByRole("button", { name: "Record again" }));
    await waitFor(() => expect(signal?.aborted).toBe(true));
    await screen.findByRole("button", { name: "Stop recording" });
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("submits each stage once under React Strict Mode", async () => {
    installRecorderMock();
    render(<StrictMode><App /></StrictMode>);
    await captureQuestion();
    await screen.findByText("Question received.");
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(["/api/transcriptions", "/api/queries"]);
  });

  it("renders answers and evidence only when returned by the answer service", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...queryResult("What is a monsoon?"), status: "answered", answer: "A seasonal change in wind direction.",
      citations: [{ document_id: "doc-1", text: "A monsoon is a seasonal wind.", source: "Weather reference" }],
    }));
    render(<App />);
    typeQuestion("What is a monsoon?");
    expect(await screen.findByText("A seasonal change in wind direction.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Sources" })).toBeInTheDocument();
    expect(screen.getByText("Weather reference")).toBeInTheDocument();
    expect(screen.queryByText("Your question is ready. Answers are coming soon.")).not.toBeInTheDocument();
  });
});

describe("voice capture", () => {
  it("does not open two permission requests from repeated starts", async () => {
    let resolveStream: ((stream: MediaStream) => void) | undefined;
    const getUserMedia = vi.fn().mockImplementation(() => new Promise<MediaStream>((resolve) => { resolveStream = resolve; }));
    installRecorderMock(getUserMedia);
    render(<App />);
    const startButton = screen.getByRole("button", { name: "Start recording" });
    fireEvent.click(startButton);
    fireEvent.click(startButton);
    expect(getUserMedia).toHaveBeenCalledOnce();
    await act(async () => { resolveStream?.({ getTracks: () => [{ stop: vi.fn() }] } as unknown as MediaStream); });
  });

  it("starts recording after React Strict Mode replays effects", async () => {
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] });
    installRecorderMock(getUserMedia);
    render(<StrictMode><App /></StrictMode>);
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    await screen.findByRole("button", { name: "Stop recording" });
    expect(getUserMedia).toHaveBeenCalledOnce();
  });

  it("stops automatically at the 30 second limit", async () => {
    vi.useFakeTimers();
    installRecorderMock();
    render(<App />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Start recording" })); });
    await act(async () => {
      vi.advanceTimersByTime(RECORDER_SETTINGS.maxDurationSeconds * 1000 - RECORDER_SETTINGS.providerSafetyMarginMs - 1);
    });
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();
    await act(async () => { vi.advanceTimersByTime(1); });
    expect(screen.getByText("Audio captured")).toBeInTheDocument();
    expect(screen.getByText(/00:30 limit reached/i)).toBeInTheDocument();
  });

  it("auto-stops after confirmed speech followed by sustained silence", async () => {
    vi.useFakeTimers();
    let rms = 0.03;
    const stopTrack = vi.fn();
    const audioContext = installVoiceActivityMock(() => rms);
    installRecorderMock(vi.fn().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] }));
    render(<App />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Start recording" })); });
    await act(async () => { vi.advanceTimersByTime(400); });
    expect(screen.getByText(/auto-stops after 2.5 seconds of silence/i)).toBeInTheDocument();
    rms = 0;
    await act(async () => { vi.advanceTimersByTime(1500); });
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();
    rms = 0.03;
    await act(async () => { vi.advanceTimersByTime(300); });
    rms = 0;
    await act(async () => { vi.advanceTimersByTime(2400); });
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();
    await act(async () => { vi.advanceTimersByTime(200); });
    expect(screen.getByText("Audio captured")).toBeInTheDocument();
    expect(screen.getByText(/silence detected/i)).toBeInTheDocument();
    expect(stopTrack).toHaveBeenCalledOnce();
    expect(audioContext.disconnectSource).toHaveBeenCalledOnce();
    expect(audioContext.disconnectAnalyser).toHaveBeenCalledOnce();
    expect(audioContext.close).toHaveBeenCalledOnce();
  });

  it("does not submit a noise spike as speech", async () => {
    vi.useFakeTimers();
    let rms = 0.03;
    installVoiceActivityMock(() => rms);
    installRecorderMock();
    render(<App />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Start recording" })); });
    await act(async () => { vi.advanceTimersByTime(100); });
    rms = 0;
    await act(async () => { vi.advanceTimersByTime(RECORDER_SETTINGS.initialSilenceMs); });
    expect(screen.getByText("No speech was detected. Move closer to the microphone and try again.")).toBeInTheDocument();
    expect(screen.getByText("No speech detected")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("explains denied microphone permission while keeping typed input available", async () => {
    installRecorderMock(vi.fn().mockRejectedValue(new DOMException("Permission denied", "NotAllowedError")));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Microphone permission was denied.");
    expect(screen.getByRole("button", { name: "Retry recording" })).toBeInTheDocument();
    typeQuestion("A typed question");
    await screen.findByText("Question received.");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/queries");
  });
});
