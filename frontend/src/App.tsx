import { useEffect, useState, type CSSProperties } from "react";

import { useAudioRecorder, type RecorderStatus } from "./hooks/useAudioRecorder";

const languages = [
  { value: "auto", label: "Auto detect" },
  { value: "en", label: "English" },
  { value: "hi", label: "Hindi" },
  { value: "mr", label: "Marathi" },
];

const waveform = [30, 48, 68, 42, 76, 92, 58, 36, 64, 82, 52, 72, 40, 56, 34];

function MicrophoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 15.25a3.75 3.75 0 0 0 3.75-3.75v-5a3.75 3.75 0 1 0-7.5 0v5A3.75 3.75 0 0 0 12 15.25Z" />
      <path d="M5.75 11.25v.25a6.25 6.25 0 0 0 12.5 0v-.25M12 17.75v3.5M8.75 21.25h6.5" />
    </svg>
  );
}

function StopIcon() {
  return <span className="stop-icon" aria-hidden="true" />;
}

function formatTime(seconds: number) {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;

  return `${minutes.toString().padStart(2, "0")}:${remainingSeconds
    .toString()
    .padStart(2, "0")}`;
}

function getTranscriptState(status: RecorderStatus) {
  switch (status) {
    case "requesting":
      return { copy: "Approve microphone access to begin.", state: "Permission" };
    case "recording":
      return { copy: "Capturing your question…", state: "Recording" };
    case "ready":
      return { copy: "Recording ready for transcription.", state: "Audio ready" };
    case "error":
      return { copy: "Your words will appear here.", state: "Needs attention" };
    default:
      return { copy: "Your words will appear here.", state: "Waiting for audio" };
  }
}

export default function App() {
  const {
    capturedAudio,
    discardRecording,
    elapsedSeconds,
    errorMessage,
    hasDetectedSpeech,
    isStopping,
    maxDurationSeconds,
    silenceDetectionAvailable,
    startRecording,
    status,
    stopRecording,
  } = useAudioRecorder();
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const isRecording = status === "recording";
  const isRequesting = status === "requesting";
  const transcript = getTranscriptState(status);

  useEffect(() => {
    if (!capturedAudio) {
      setAudioUrl(null);
      return;
    }

    const nextAudioUrl = URL.createObjectURL(capturedAudio.blob);
    setAudioUrl(nextAudioUrl);

    return () => URL.revokeObjectURL(nextAudioUrl);
  }, [capturedAudio]);

  const recordingDetail =
    silenceDetectionAvailable === false
      ? `Tap to stop · ${formatTime(elapsedSeconds)} / ${formatTime(maxDurationSeconds)}`
      : hasDetectedSpeech
        ? `Auto-stops after 2.5 seconds of silence · ${formatTime(elapsedSeconds)}`
        : `Start speaking · ${formatTime(elapsedSeconds)} / ${formatTime(maxDurationSeconds)}`;
  const readyDetail = (() => {
    const duration = formatTime(capturedAudio?.durationSeconds ?? elapsedSeconds);

    if (capturedAudio?.stopReason === "silence") {
      return `${duration} captured · silence detected`;
    }

    if (capturedAudio?.stopReason === "limit") {
      return `${formatTime(maxDurationSeconds)} limit reached`;
    }

    return `${duration} captured · tap to record again`;
  })();

  const statusCopy = {
    idle: {
      detail: `Auto-stops after silence · ${formatTime(maxDurationSeconds)} maximum`,
      title: "Ready when you are",
    },
    requesting: {
      detail: "Approve access in your browser",
      title: "Requesting microphone",
    },
    recording: {
      detail: recordingDetail,
      title: "Listening…",
    },
    ready: {
      detail: readyDetail,
      title: "Audio captured",
    },
    error: {
      detail: errorMessage ?? "Check your microphone and try again.",
      title: errorMessage?.startsWith("No speech")
        ? "No speech detected"
        : "Microphone unavailable",
    },
  }[status];
  const visibleStatusCopy = isStopping
    ? { detail: "Saving the final audio chunk", title: "Finishing recording…" }
    : statusCopy;
  const recordButtonLabel = isStopping
    ? "Finishing recording"
    : isRecording
      ? "Stop recording"
      : status === "ready"
        ? "Record again"
        : status === "error"
          ? "Retry recording"
          : "Start recording";

  const handleRecordClick = () => {
    if (isRecording) {
      stopRecording();
      return;
    }

    void startRecording();
  };

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <a className="brand" href="/" aria-label="SubaRAG home">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <span>SubaRAG</span>
        </a>
        <div className="build-label">
          <span className="status-dot" aria-hidden="true" />
          Voice input · Step 01
        </div>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">VOICE-FIRST RETRIEVAL</p>
        <h1 id="page-title">
          Ask at the speed
          <span> of thought.</span>
        </h1>
        <p className="hero-copy">
          Speak naturally. SubaRAG will turn your question into a grounded search
          without putting a dashboard between you and the answer.
        </p>
      </section>

      <section className="voice-card" aria-labelledby="voice-card-title">
        <div className="card-heading">
          <div>
            <p className="section-index">01 / CAPTURE</p>
            <h2 id="voice-card-title">What do you want to know?</h2>
          </div>

          <label className="language-control">
            <span>Spoken language</span>
            <select
              defaultValue="auto"
              aria-label="Spoken language"
              disabled={isRecording || isRequesting || isStopping}
            >
              {languages.map((language) => (
                <option key={language.value} value={language.value}>
                  {language.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="recorder-stage" data-state={status}>
          <div className={`waveform${isRecording ? " is-recording" : ""}`} aria-hidden="true">
            {waveform.map((height, index) => (
              <span
                key={`${height}-${index}`}
                style={{
                  "--bar-height": `${height}%`,
                  "--bar-delay": `${index * 45}ms`,
                } as CSSProperties}
              />
            ))}
          </div>

          <button
            className="record-button"
            type="button"
            aria-label={recordButtonLabel}
            aria-pressed={isRecording}
            disabled={isRequesting || isStopping}
            onClick={handleRecordClick}
          >
            <span className="record-ring" aria-hidden="true" />
            <span className="record-core">
              {isRecording ? <StopIcon /> : <MicrophoneIcon />}
            </span>
          </button>

          <div
            className="recorder-status"
            role={status === "error" ? "alert" : "status"}
            aria-live="polite"
          >
            <strong>{visibleStatusCopy.title}</strong>
            <span>{visibleStatusCopy.detail}</span>
          </div>
        </div>

        <div className="transcript-panel" data-state={status} aria-labelledby="transcript-title">
          <div className="transcript-heading">
            <div>
              <p className="section-index">LIVE TRANSCRIPT</p>
              <h3 id="transcript-title">{transcript.copy}</h3>
            </div>
            <span className="transcript-state" data-state={status}>
              {transcript.state}
            </span>
          </div>

          {status === "ready" && audioUrl ? (
            <div className="audio-preview">
              <audio controls preload="metadata" src={audioUrl} aria-label="Recorded audio preview">
                Your browser does not support audio playback.
              </audio>
              <button className="discard-button" type="button" onClick={discardRecording}>
                Discard
              </button>
            </div>
          ) : null}
        </div>
      </section>

      <footer className="page-footer">
        <p>Built for low-latency questions, from microphone to evidence.</p>
        <p>Playback is temporarily available for capture debugging.</p>
      </footer>
    </main>
  );
}
