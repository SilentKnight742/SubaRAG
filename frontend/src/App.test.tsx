import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { RECORDER_SETTINGS } from "./hooks/useAudioRecorder";

const originalMediaDevices = navigator.mediaDevices;
const originalCreateObjectUrl = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevokeObjectUrl = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
const createObjectUrl = vi.fn(() => "blob:subarag-recording");
const revokeObjectUrl = vi.fn();

beforeEach(() => {
  createObjectUrl.mockClear();
  revokeObjectUrl.mockClear();
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
    expect(screen.getByText("Recording ready for transcription.")).toBeInTheDocument();
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
      vi.advanceTimersByTime(30_000);
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
