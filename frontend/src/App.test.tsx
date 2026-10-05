import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { TranscriptionProvider } from "./api/transcription";
import { RECORDER_SETTINGS } from "./hooks/useAudioRecorder";

const originalMediaDevices = navigator.mediaDevices;
const originalCreateObjectUrl = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevokeObjectUrl = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
const createObjectUrl = vi.fn(() => "blob:subarag-recording");
const revokeObjectUrl = vi.fn();
const fetchMock = vi.fn();

function transcriptionResult(provider: TranscriptionProvider) {
  return {
    text: `Transcript from ${provider === "sarvam" ? "Sarvam" : "ElevenLabs"}.`,
    provider,
    latency_ms: provider === "sarvam" ? 182 : 241,
    duration_seconds: 1,
    language: "hi-IN",
    language_confidence: 0.98,
    segments: [],
    provider_request_id: `${provider}-request-id`,
  };
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function formDataForCall(callIndex: number) {
  return fetchMock.mock.calls[callIndex]?.[1]?.body as FormData;
}

beforeEach(() => {
  createObjectUrl.mockClear();
  revokeObjectUrl.mockClear();
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (_input: RequestInfo | URL, init?: RequestInit) => {
    const provider = (init?.body as FormData).get("provider") as TranscriptionProvider;
    return jsonResponse(transcriptionResult(provider));
  });
  vi.stubGlobal("fetch", fetchMock);
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: createObjectUrl,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: revokeObjectUrl,
  });
});

function installRecorderMock(getUserMedia: ReturnType<typeof vi.fn>) {
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });

  class MockMediaRecorder {
    static isTypeSupported() {
      return true;
    }

    mimeType: string;
    ondataavailable: ((event: BlobEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    onstop: ((event: Event) => void) | null = null;
    state: RecordingState = "inactive";

    constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
      this.mimeType = options?.mimeType ?? "audio/webm";
    }

    start() {
      this.state = "recording";
    }

    stop() {
      this.state = "inactive";
      this.ondataavailable?.({
        data: new Blob(["voice"], { type: this.mimeType }),
      } as BlobEvent);
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

    getFloatTimeDomainData(samples: Float32Array) {
      samples.fill(readRms());
    }
  }

  class MockAudioContext {
    state: AudioContextState = "running";

    close = close;

    createAnalyser() {
      return new MockAnalyserNode();
    }

    createMediaStreamSource() {
      return {
        connect: vi.fn(),
        disconnect: disconnectSource,
      };
    }

    resume() {
      return Promise.resolve();
    }
  }

  vi.stubGlobal("AudioContext", MockAudioContext);

  return { close, disconnectAnalyser, disconnectSource };
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: originalMediaDevices,
  });

  if (originalCreateObjectUrl) {
    Object.defineProperty(URL, "createObjectURL", originalCreateObjectUrl);
  } else {
    Reflect.deleteProperty(URL, "createObjectURL");
  }

  if (originalRevokeObjectUrl) {
    Object.defineProperty(URL, "revokeObjectURL", originalRevokeObjectUrl);
  } else {
    Reflect.deleteProperty(URL, "revokeObjectURL");
  }
});

describe("voice input shell", () => {
  it("presents the focused recording controls and supported languages", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /ask at the speed/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start recording" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Spoken language" })).toHaveValue("auto");
    expect(screen.getByRole("combobox", { name: "Test provider" })).toHaveValue("compare");
    expect(screen.getByRole("option", { name: "Hindi" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Marathi" })).toBeInTheDocument();
    expect(screen.getByText("Your words will appear here.")).toBeInTheDocument();
    expect(screen.getByText(/00:30 maximum/i)).toBeInTheDocument();
  });

  it("captures audio locally and releases the microphone", async () => {
    const stopTrack = vi.fn();
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: stopTrack }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));

    expect(await screen.findByRole("button", { name: "Stop recording" })).toBeInTheDocument();
    expect(screen.getByText("Listening…")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Stop recording" }));

    expect(await screen.findByText("Audio captured")).toBeInTheDocument();
    expect(await screen.findByText("Provider comparison ready.")).toBeInTheDocument();
    expect(screen.getByText("Transcript from Sarvam.")).toBeInTheDocument();
    expect(screen.getByText("Transcript from ElevenLabs.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.getByLabelText("Recorded audio preview")).toHaveAttribute(
      "src",
      "blob:subarag-recording",
    );
    expect(stopTrack).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Discard" }));

    expect(screen.queryByLabelText("Recorded audio preview")).not.toBeInTheDocument();
    expect(screen.getByText("Ready when you are")).toBeInTheDocument();
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:subarag-recording");
  });

  it("sends the same captured audio to both providers automatically", async () => {
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    expect(await screen.findByText("Provider comparison ready.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const sarvamForm = formDataForCall(0);
    const elevenLabsForm = formDataForCall(1);
    const sarvamAudio = sarvamForm.get("audio") as File;
    const elevenLabsAudio = elevenLabsForm.get("audio") as File;

    expect(sarvamForm.get("provider")).toBe("sarvam");
    expect(elevenLabsForm.get("provider")).toBe("elevenlabs");
    expect(sarvamForm.has("language")).toBe(false);
    expect(elevenLabsForm.has("language")).toBe(false);
    expect(sarvamAudio.size).toBe(elevenLabsAudio.size);
    expect(sarvamAudio.type).toBe("audio/webm;codecs=opus");
    expect(elevenLabsAudio.type).toBe("audio/webm;codecs=opus");
  });

  it("forwards an explicit language to one selected provider", async () => {
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.change(screen.getByRole("combobox", { name: "Spoken language" }), {
      target: { value: "mr" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Test provider" }), {
      target: { value: "sarvam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    expect(await screen.findByText("Transcript ready.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(formDataForCall(0).get("provider")).toBe("sarvam");
    expect(formDataForCall(0).get("language")).toBe("mr");
  });

  it("keeps a successful comparison result when the other provider fails", async () => {
    let elevenLabsAttempts = 0;
    fetchMock.mockImplementation(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const provider = (init?.body as FormData).get("provider") as TranscriptionProvider;

      if (provider === "elevenlabs" && elevenLabsAttempts++ === 0) {
        return jsonResponse(
          {
            error: {
              code: "provider_unavailable",
              message: "ElevenLabs is temporarily unavailable.",
              retryable: true,
            },
          },
          503,
        );
      }

      return jsonResponse(transcriptionResult(provider));
    });
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    expect(await screen.findByText("One provider completed.")).toBeInTheDocument();
    expect(screen.getByText("Transcript from Sarvam.")).toBeInTheDocument();
    expect(screen.getByText("ElevenLabs is temporarily unavailable.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry transcription" }));

    expect(await screen.findByText("Provider comparison ready.")).toBeInTheDocument();
    expect(screen.getByText("Transcript from Sarvam.")).toBeInTheDocument();
    expect(screen.getByText("Transcript from ElevenLabs.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(formDataForCall(0).get("provider")).toBe("sarvam");
    expect(formDataForCall(1).get("provider")).toBe("elevenlabs");
    expect(formDataForCall(2).get("provider")).toBe("elevenlabs");
  });

  it("does not offer retry for a deterministic transcription error", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: "unsupported_language",
            message: "The selected language is not supported.",
            retryable: false,
          },
        },
        400,
      ),
    );
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.change(screen.getByRole("combobox", { name: "Test provider" }), {
      target: { value: "sarvam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    expect(await screen.findByText("The selected language is not supported.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry transcription" })).not.toBeInTheDocument();
  });

  it("retries a failed transcription with the same recording", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(
          {
            error: {
              code: "provider_timeout",
              message: "Sarvam timed out.",
              retryable: true,
            },
          },
          504,
        ),
      )
      .mockResolvedValueOnce(jsonResponse(transcriptionResult("sarvam")));
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.change(screen.getByRole("combobox", { name: "Test provider" }), {
      target: { value: "sarvam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    fireEvent.click(await screen.findByRole("button", { name: "Retry transcription" }));

    expect(await screen.findByText("Transcript from Sarvam.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(formDataForCall(0).get("audio")).toBeInstanceOf(File);
    expect(formDataForCall(1).get("audio")).toBeInstanceOf(File);
  });

  it("aborts an in-flight transcription when recording again", async () => {
    let requestSignal: AbortSignal | undefined;
    fetchMock.mockImplementation(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          requestSignal = init?.signal ?? undefined;
          requestSignal?.addEventListener("abort", () => {
            reject(new DOMException("Request aborted", "AbortError"));
          });
        }),
    );
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.change(screen.getByRole("combobox", { name: "Test provider" }), {
      target: { value: "sarvam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));
    expect(await screen.findByText("Transcribing your question…")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Record again" }));

    await waitFor(() => expect(requestSignal?.aborted).toBe(true));
    expect(await screen.findByRole("button", { name: "Stop recording" })).toBeInTheDocument();
  });

  it("submits a completed recording only once under React Strict Mode", async () => {
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );

    fireEvent.change(screen.getByRole("combobox", { name: "Test provider" }), {
      target: { value: "sarvam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    expect(await screen.findByText("Transcript ready.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("does not open two permission requests from repeated starts", async () => {
    let resolveStream: ((stream: MediaStream) => void) | undefined;
    const getUserMedia = vi.fn().mockImplementation(
      () =>
        new Promise<MediaStream>((resolve) => {
          resolveStream = resolve;
        }),
    );
    installRecorderMock(getUserMedia);
    render(<App />);

    const startButton = screen.getByRole("button", { name: "Start recording" });
    fireEvent.click(startButton);
    fireEvent.click(startButton);

    expect(getUserMedia).toHaveBeenCalledOnce();

    await act(async () => {
      resolveStream?.({ getTracks: () => [{ stop: vi.fn() }] } as unknown as MediaStream);
      await Promise.resolve();
    });
  });

  it("starts recording after React Strict Mode replays effects", async () => {
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);

    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));

    expect(await screen.findByRole("button", { name: "Stop recording" })).toBeInTheDocument();
    expect(getUserMedia).toHaveBeenCalledOnce();
  });

  it("stops automatically at the 30 second limit", async () => {
    vi.useFakeTimers();
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installRecorderMock(getUserMedia);
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
      await Promise.resolve();
    });

    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(
        RECORDER_SETTINGS.maxDurationSeconds * 1_000 -
          RECORDER_SETTINGS.providerSafetyMarginMs -
          1,
      );
    });
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1);
    });

    expect(screen.getByText("Audio captured")).toBeInTheDocument();
    expect(screen.getByText(/00:30 limit reached/i)).toBeInTheDocument();
  });

  it("auto-stops after confirmed speech followed by sustained silence", async () => {
    vi.useFakeTimers();
    let rms = 0.03;
    const stopTrack = vi.fn();
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: stopTrack }],
    });
    const audioContext = installVoiceActivityMock(() => rms);
    installRecorderMock(getUserMedia);
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
      await Promise.resolve();
    });

    act(() => {
      vi.advanceTimersByTime(400);
    });

    expect(screen.getByText(/auto-stops after 2.5 seconds of silence/i)).toBeInTheDocument();

    rms = 0;
    act(() => {
      vi.advanceTimersByTime(1_500);
    });
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();

    rms = 0.03;
    act(() => {
      vi.advanceTimersByTime(300);
    });
    rms = 0;
    act(() => {
      vi.advanceTimersByTime(2_400);
    });
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(200);
    });

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
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    installVoiceActivityMock(() => rms);
    installRecorderMock(getUserMedia);
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
      await Promise.resolve();
    });

    act(() => {
      vi.advanceTimersByTime(100);
    });
    rms = 0;
    act(() => {
      vi.advanceTimersByTime(RECORDER_SETTINGS.initialSilenceMs);
    });

    expect(screen.getByText("No speech was detected. Move closer to the microphone and try again."))
      .toBeInTheDocument();
    expect(screen.getByText("No speech detected")).toBeInTheDocument();
    expect(screen.queryByLabelText("Recorded audio preview")).not.toBeInTheDocument();
  });

  it("explains when microphone permission is denied", async () => {
    const getUserMedia = vi
      .fn()
      .mockRejectedValue(new DOMException("Permission denied", "NotAllowedError"));
    installRecorderMock(getUserMedia);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));

    expect(
      await screen.findByText("Microphone permission was denied. Allow access and try again."),
    ).toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry recording" })).toBeInTheDocument();
  });
});
