import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderStatus = "idle" | "requesting" | "recording" | "ready" | "error";
export type RecordingStopReason = "manual" | "silence" | "limit" | "no-speech";

export interface CapturedAudio {
  blob: Blob;
  durationSeconds: number;
  mimeType: string;
  stopReason: Exclude<RecordingStopReason, "no-speech">;
}

export const RECORDER_SETTINGS = {
  analysisIntervalMs: 100,
  initialSilenceMs: 6_000,
  maxDurationSeconds: 30,
  providerSafetyMarginMs: 500,
  speechConfirmationMs: 200,
  speechContinueRms: 0.008,
  speechStartRms: 0.015,
  trailingSilenceMs: 2_500,
} as const;

type RecorderSettings = typeof RECORDER_SETTINGS;
type AudioContextGlobal = typeof globalThis & {
  webkitAudioContext?: typeof AudioContext;
};

const preferredMimeTypes = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
];

function selectMimeType() {
  if (typeof MediaRecorder.isTypeSupported !== "function") {
    return undefined;
  }

  return preferredMimeTypes.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
}

function describeRecorderError(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Microphone permission was denied. Allow access and try again.";
    }

    if (error.name === "NotFoundError") {
      return "No microphone was found on this device.";
    }

    if (error.name === "NotReadableError" || error.name === "AbortError") {
      return "The microphone is busy or could not be read.";
    }
  }

  return "The microphone could not start. Check your browser settings and try again.";
}

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function calculateRms(samples: Float32Array) {
  let squareSum = 0;

  for (const sample of samples) {
    squareSum += sample * sample;
  }

  return Math.sqrt(squareSum / samples.length);
}

export function useAudioRecorder(overrides: Partial<RecorderSettings> = {}) {
  const analysisIntervalMs =
    overrides.analysisIntervalMs ?? RECORDER_SETTINGS.analysisIntervalMs;
  const initialSilenceMs = overrides.initialSilenceMs ?? RECORDER_SETTINGS.initialSilenceMs;
  const maxDurationSeconds =
    overrides.maxDurationSeconds ?? RECORDER_SETTINGS.maxDurationSeconds;
  const providerSafetyMarginMs =
    overrides.providerSafetyMarginMs ?? RECORDER_SETTINGS.providerSafetyMarginMs;
  const speechConfirmationMs =
    overrides.speechConfirmationMs ?? RECORDER_SETTINGS.speechConfirmationMs;
  const speechContinueRms =
    overrides.speechContinueRms ?? RECORDER_SETTINGS.speechContinueRms;
  const speechStartRms = overrides.speechStartRms ?? RECORDER_SETTINGS.speechStartRms;
  const trailingSilenceMs = overrides.trailingSilenceMs ?? RECORDER_SETTINGS.trailingSilenceMs;

  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [capturedAudio, setCapturedAudio] = useState<CapturedAudio | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [hasDetectedSpeech, setHasDetectedSpeech] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [silenceDetectionAvailable, setSilenceDetectionAvailable] = useState<boolean | null>(
    null,
  );

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  const captureIdRef = useRef(0);
  const requestingRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const stopReasonRef = useRef<RecordingStopReason>("manual");
  const speechDetectedRef = useRef(false);
  const elapsedIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const durationTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const analysisIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mountedRef = useRef(true);

  const clearRecordingTimers = useCallback(() => {
    if (elapsedIntervalRef.current !== null) {
      clearInterval(elapsedIntervalRef.current);
      elapsedIntervalRef.current = null;
    }

    if (durationTimeoutRef.current !== null) {
      clearTimeout(durationTimeoutRef.current);
      durationTimeoutRef.current = null;
    }
  }, []);

  const stopSilenceDetection = useCallback(() => {
    if (analysisIntervalRef.current !== null) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }

    audioSourceRef.current?.disconnect();
    audioSourceRef.current = null;
    analyserRef.current?.disconnect();
    analyserRef.current = null;

    const audioContext = audioContextRef.current;
    audioContextRef.current = null;

    if (audioContext && audioContext.state !== "closed") {
      void audioContext.close().catch(() => undefined);
    }
  }, []);

  const releaseStream = useCallback(() => {
    stopStream(streamRef.current);
    streamRef.current = null;
  }, []);

  const stopRecording = useCallback(
    (reason: RecordingStopReason = "manual") => {
      const recorder = recorderRef.current;

      if (recorder && recorder.state !== "inactive" && !stopRequestedRef.current) {
        stopRequestedRef.current = true;
        stopReasonRef.current = reason;
        clearRecordingTimers();
        stopSilenceDetection();
        setIsStopping(true);
        recorder.stop();
      }
    },
    [clearRecordingTimers, stopSilenceDetection],
  );

  const startSilenceDetection = useCallback(
    (stream: MediaStream, captureId: number) => {
      const AudioContextConstructor =
        globalThis.AudioContext ?? (globalThis as AudioContextGlobal).webkitAudioContext;

      if (!AudioContextConstructor) {
        setSilenceDetectionAvailable(false);
        return;
      }

      try {
        const audioContext = new AudioContextConstructor();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        const samples = new Float32Array(1_024);
        let monitorStartedAt = performance.now();
        let speechCandidateStartedAt: number | null = null;
        let lastSpeechAt = monitorStartedAt;

        analyser.fftSize = samples.length;
        analyser.smoothingTimeConstant = 0.2;
        source.connect(analyser);

        audioContextRef.current = audioContext;
        audioSourceRef.current = source;
        analyserRef.current = analyser;
        setSilenceDetectionAvailable(true);

        if (audioContext.state === "suspended") {
          void audioContext
            .resume()
            .then(() => {
              monitorStartedAt = performance.now();
              lastSpeechAt = monitorStartedAt;
            })
            .catch(() => {
              if (captureId === captureIdRef.current) {
                stopSilenceDetection();
                setSilenceDetectionAvailable(false);
              }
            });
        }

        analysisIntervalRef.current = setInterval(() => {
          if (
            !mountedRef.current ||
            captureId !== captureIdRef.current ||
            stopRequestedRef.current ||
            audioContext.state !== "running"
          ) {
            return;
          }

          const now = performance.now();
          analyser.getFloatTimeDomainData(samples);
          const rms = calculateRms(samples);
          const activeThreshold = speechDetectedRef.current
            ? speechContinueRms
            : speechStartRms;

          if (rms >= activeThreshold) {
            if (speechDetectedRef.current) {
              lastSpeechAt = now;
              return;
            }

            speechCandidateStartedAt ??= now;

            if (now - speechCandidateStartedAt >= speechConfirmationMs) {
              speechDetectedRef.current = true;
              lastSpeechAt = now;
              setHasDetectedSpeech(true);
            }

            return;
          }

          speechCandidateStartedAt = null;

          if (speechDetectedRef.current && now - lastSpeechAt >= trailingSilenceMs) {
            stopRecording("silence");
            return;
          }

          if (!speechDetectedRef.current && now - monitorStartedAt >= initialSilenceMs) {
            stopRecording("no-speech");
          }
        }, analysisIntervalMs);
      } catch {
        stopSilenceDetection();
        setSilenceDetectionAvailable(false);
      }
    },
    [
      analysisIntervalMs,
      initialSilenceMs,
      speechConfirmationMs,
      speechContinueRms,
      speechStartRms,
      stopRecording,
      stopSilenceDetection,
      trailingSilenceMs,
    ],
  );

  const startRecording = useCallback(async () => {
    if (
      requestingRef.current ||
      status === "requesting" ||
      status === "recording" ||
      (recorderRef.current && recorderRef.current.state !== "inactive")
    ) {
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setErrorMessage("Audio recording is not supported in this browser.");
      setStatus("error");
      return;
    }

    clearRecordingTimers();
    stopSilenceDetection();
    releaseStream();
    const captureId = captureIdRef.current + 1;
    captureIdRef.current = captureId;
    requestingRef.current = true;
    stopRequestedRef.current = false;
    stopReasonRef.current = "manual";
    speechDetectedRef.current = false;
    setCapturedAudio(null);
    setElapsedSeconds(0);
    setErrorMessage(null);
    setHasDetectedSpeech(false);
    setIsStopping(false);
    setSilenceDetectionAvailable(null);
    setStatus("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });

      if (!mountedRef.current || captureId !== captureIdRef.current) {
        stopStream(stream);
        return;
      }

      streamRef.current = stream;
      chunksRef.current = [];

      const mimeType = selectMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (captureId === captureIdRef.current && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        if (captureId !== captureIdRef.current) {
          return;
        }

        clearRecordingTimers();
        stopSilenceDetection();
        releaseStream();
        recorder.ondataavailable = null;
        recorder.onstop = null;
        recorderRef.current = null;
        requestingRef.current = false;
        stopRequestedRef.current = false;

        if (mountedRef.current) {
          setIsStopping(false);
          setErrorMessage("Recording stopped because the microphone reported an error.");
          setStatus("error");
        }
      };

      recorder.onstop = () => {
        if (captureId !== captureIdRef.current) {
          return;
        }

        clearRecordingTimers();
        stopSilenceDetection();
        releaseStream();
        recorderRef.current = null;
        requestingRef.current = false;
        stopRequestedRef.current = false;
        recorder.ondataavailable = null;
        recorder.onerror = null;
        recorder.onstop = null;

        if (!mountedRef.current) {
          return;
        }

        const durationSeconds = Math.min(
          maxDurationSeconds,
          Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1_000)),
        );
        const stopReason = stopReasonRef.current;

        if (stopReason === "no-speech") {
          chunksRef.current = [];
          setElapsedSeconds(0);
          setIsStopping(false);
          setErrorMessage("No speech was detected. Move closer to the microphone and try again.");
          setStatus("error");
          return;
        }

        const recordedMimeType = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: recordedMimeType });

        if (blob.size === 0) {
          setIsStopping(false);
          setErrorMessage("No audio was captured. Check the microphone and try again.");
          setStatus("error");
          return;
        }

        setElapsedSeconds(durationSeconds);
        setCapturedAudio({ blob, durationSeconds, mimeType: recordedMimeType, stopReason });
        setIsStopping(false);
        setStatus("ready");
      };

      startedAtRef.current = Date.now();
      recorder.start();
      requestingRef.current = false;
      setStatus("recording");
      startSilenceDetection(stream, captureId);

      elapsedIntervalRef.current = setInterval(() => {
        const nextElapsed = Math.min(
          maxDurationSeconds,
          Math.floor((Date.now() - startedAtRef.current) / 1_000),
        );
        setElapsedSeconds(nextElapsed);
      }, 250);

      durationTimeoutRef.current = setTimeout(
        () => stopRecording("limit"),
        Math.max(0, maxDurationSeconds * 1_000 - providerSafetyMarginMs),
      );
    } catch (error) {
      const recorder = recorderRef.current;
      if (recorder) {
        recorder.ondataavailable = null;
        recorder.onerror = null;
        recorder.onstop = null;
        recorderRef.current = null;
      }

      clearRecordingTimers();
      stopSilenceDetection();
      releaseStream();
      requestingRef.current = false;
      stopRequestedRef.current = false;

      if (mountedRef.current && captureId === captureIdRef.current) {
        setIsStopping(false);
        setErrorMessage(describeRecorderError(error));
        setStatus("error");
      }
    }
  }, [
    clearRecordingTimers,
    maxDurationSeconds,
    providerSafetyMarginMs,
    releaseStream,
    startSilenceDetection,
    status,
    stopRecording,
    stopSilenceDetection,
  ]);

  const discardRecording = useCallback(() => {
    captureIdRef.current += 1;
    requestingRef.current = false;
    stopRequestedRef.current = false;
    speechDetectedRef.current = false;
    clearRecordingTimers();
    stopSilenceDetection();

    const recorder = recorderRef.current;
    if (recorder) {
      recorder.ondataavailable = null;
      recorder.onerror = null;
      recorder.onstop = null;

      if (recorder.state !== "inactive") {
        recorder.stop();
      }

      recorderRef.current = null;
    }

    releaseStream();
    chunksRef.current = [];
    setCapturedAudio(null);
    setElapsedSeconds(0);
    setErrorMessage(null);
    setHasDetectedSpeech(false);
    setIsStopping(false);
    setSilenceDetectionAvailable(null);
    setStatus("idle");
  }, [clearRecordingTimers, releaseStream, stopSilenceDetection]);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      captureIdRef.current += 1;
      requestingRef.current = false;
      stopRequestedRef.current = false;
      speechDetectedRef.current = false;
      clearRecordingTimers();
      stopSilenceDetection();

      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.ondataavailable = null;
        recorder.onerror = null;
        recorder.onstop = null;
        recorder.stop();
      }

      releaseStream();
    };
  }, [clearRecordingTimers, releaseStream, stopSilenceDetection]);

  return {
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
  };
}
