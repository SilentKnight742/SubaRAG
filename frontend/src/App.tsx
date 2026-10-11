import { useEffect, useState, type CSSProperties, type FormEvent } from "react";

import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { useQuestion } from "./hooks/useQuestion";

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

function formatTime(seconds: number) {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  return `${Math.floor(wholeSeconds / 60).toString().padStart(2, "0")}:${(wholeSeconds % 60)
    .toString().padStart(2, "0")}`;
}

export default function App() {
  const {
    capturedAudio, discardRecording, elapsedSeconds, errorMessage,
    hasDetectedSpeech, isStopping, maxDurationSeconds, silenceDetectionAvailable,
    startRecording, status, stopRecording,
  } = useAudioRecorder();
  const {
    question, result, error, reset, retry, status: questionStatus, submitAudio, submitText,
  } = useQuestion();
  const [language, setLanguage] = useState("auto");
  const [draft, setDraft] = useState("");
  const isRecording = status === "recording";
  const isRequesting = status === "requesting";
  const isWorking = questionStatus === "transcribing" || questionStatus === "querying";
  const captureActive = isRecording || isRequesting || isStopping;

  useEffect(() => {
    if (capturedAudio) submitAudio({ audio: capturedAudio.blob, language });
  }, [capturedAudio, language, submitAudio]);

  const recordingDetail = silenceDetectionAvailable === false
    ? `Tap to stop · ${formatTime(elapsedSeconds)} / ${formatTime(maxDurationSeconds)}`
    : hasDetectedSpeech
      ? `Auto-stops after 2.5 seconds of silence · ${formatTime(elapsedSeconds)}`
      : `Start speaking · ${formatTime(elapsedSeconds)} / ${formatTime(maxDurationSeconds)}`;
  const readyDetail = capturedAudio?.stopReason === "silence"
    ? `${formatTime(capturedAudio.durationSeconds)} captured · silence detected`
    : capturedAudio?.stopReason === "limit"
      ? `${formatTime(maxDurationSeconds)} limit reached`
      : `${formatTime(capturedAudio?.durationSeconds ?? elapsedSeconds)} captured · tap to record again`;
  const statusCopy = {
    idle: { title: "Ready when you are", detail: `Auto-stops after silence · ${formatTime(maxDurationSeconds)} maximum` },
    requesting: { title: "Requesting microphone", detail: "Approve access in your browser" },
    recording: { title: "Listening…", detail: recordingDetail },
    ready: { title: "Audio captured", detail: readyDetail },
    error: {
      title: errorMessage?.startsWith("No speech") ? "No speech detected" : "Microphone unavailable",
      detail: errorMessage ?? "Check your microphone and try again.",
    },
  }[status];
  const visibleStatusCopy = isStopping
    ? { title: "Finishing recording…", detail: "Saving the final audio chunk" }
    : statusCopy;
  const recordButtonLabel = isStopping
    ? "Finishing recording"
    : isRecording ? "Stop recording"
      : status === "ready" ? "Record again"
        : status === "error" ? "Retry recording" : "Start recording";

  const panelTitle = questionStatus === "transcribing" ? "Transcribing your question…"
    : questionStatus === "querying" ? "Sending your question…"
      : questionStatus === "error" ? "Let’s try that again."
        : result?.status === "answered" ? "Here’s what we found."
          : result?.status === "no_answer" ? "No answer found."
            : result ? "Question received." : "Your words will appear here.";
  const panelState = isWorking ? "Working"
    : questionStatus === "error" ? "Needs attention"
      : result?.status === "answered" ? "Answered"
        : result ? "Received" : "Ready";

  const handleRecordClick = () => {
    if (isRecording) {
      stopRecording();
      return;
    }
    reset();
    void startRecording();
  };

  const handleClear = () => {
    reset();
    discardRecording();
  };

  const handleTextSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (captureActive || isWorking || !draft.trim()) return;
    discardRecording();
    reset();
    submitText({ question: draft, language: language === "auto" ? null : language });
  };

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="topbar">
        <a className="brand" href="/" aria-label="SubaRAG home">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span>SubaRAG</span>
        </a>
        <div className="build-label"><span className="status-dot" aria-hidden="true" />Early preview</div>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">VOICE-FIRST RETRIEVAL</p>
        <h1 id="page-title">Ask at the speed<span> of thought.</span></h1>
        <p className="hero-copy">
          A shorter path from a question to understanding. Speak naturally, or type what’s on your mind.
        </p>
      </section>

      <section className="voice-card" aria-labelledby="voice-card-title">
        <div className="card-heading">
          <div>
            <p className="section-index">START WITH A QUESTION</p>
            <h2 id="voice-card-title">What do you want to know?</h2>
          </div>
          <label className="language-control">
            <span>Spoken language</span>
            <select value={language} aria-label="Spoken language" disabled={captureActive || isWorking}
              onChange={(event) => setLanguage(event.target.value)}>
              {languages.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
          </label>
        </div>

        <div className="recorder-stage" data-state={status}>
          <div className={`waveform${isRecording ? " is-recording" : ""}`} aria-hidden="true">
            {waveform.map((height, index) => (
              <span key={`${height}-${index}`} style={{
                "--bar-height": `${height}%`, "--bar-delay": `${index * 45}ms`,
              } as CSSProperties} />
            ))}
          </div>
          <button className="record-button" type="button" aria-label={recordButtonLabel}
            aria-pressed={isRecording} disabled={isRequesting || isStopping} onClick={handleRecordClick}>
            <span className="record-ring" aria-hidden="true" />
            <span className="record-core">
              {isRecording ? <span className="stop-icon" aria-hidden="true" /> : <MicrophoneIcon />}
            </span>
          </button>
          <div className="recorder-status" role={status === "error" ? "alert" : "status"} aria-live="polite">
            <strong>{visibleStatusCopy.title}</strong><span>{visibleStatusCopy.detail}</span>
          </div>
        </div>

        <details className="typed-question">
          <summary>Or type your question</summary>
          <form onSubmit={handleTextSubmit}>
            <label htmlFor="typed-question">Your question</label>
            <textarea id="typed-question" value={draft} rows={2} maxLength={2000}
              placeholder="What would you like to know?" disabled={captureActive || isWorking}
              onChange={(event) => setDraft(event.target.value)} />
            <button className="submit-button" type="submit" disabled={captureActive || isWorking || !draft.trim()}>
              Send question <span aria-hidden="true">↗</span>
            </button>
          </form>
        </details>

        <section className="transcript-panel" data-state={questionStatus} aria-labelledby="transcript-title"
          aria-busy={isWorking} aria-live="polite">
          <div className="transcript-heading">
            <div><p className="section-index">YOUR QUESTION</p><h3 id="transcript-title">{panelTitle}</h3></div>
            <span className="transcript-state" data-state={questionStatus}>{panelState}</span>
          </div>
          {question ? <p className="question-text">{question}</p> : null}
          {isWorking ? (
            <div className="transcription-progress" role="status"><span aria-hidden="true" />
              {questionStatus === "transcribing" ? "Turning your words into text…" : "Getting your question ready…"}
            </div>
          ) : null}
          {result?.status === "unavailable" ? (
            <p className="answer-notice">Your question is ready. Answers are coming soon.</p>
          ) : result?.status === "no_answer" ? (
            <p className="answer-notice">{result.message ?? "We couldn’t find enough evidence to answer that question."}</p>
          ) : result?.status === "answered" ? (
            <div className="answer-result">
              <p>{result.answer}</p>
              {result.citations.length > 0 ? <div className="answer-sources">
                <h4>Sources</h4>
                <ol>{result.citations.map((citation, index) => <li key={`${citation.document_id}-${index}`}>
                  <p>{citation.text}</p>{citation.source ? <small>{citation.source}</small> : null}
                </li>)}</ol>
              </div> : null}
            </div>
          ) : null}
          {error ? <p className="request-error" role="alert">{error.message}</p> : null}
          {questionStatus !== "idle" ? <div className="result-actions">
            {error?.retryable ? <button className="retry-button" type="button" onClick={retry}>
              {error.stage === "audio" ? "Retry transcription" : "Retry question"}
            </button> : null}
            <button className="clear-button" type="button" onClick={handleClear}>{isWorking ? "Cancel" : "Clear"}</button>
          </div> : null}
        </section>
      </section>
      <footer className="page-footer">
        <p>From microphone to evidence.</p><p>Voice input is ready. Knowledge search is being built.</p>
      </footer>
    </main>
  );
}
