import { useCallback, useEffect, useRef, useState } from "react";

import { askQuestion, QueryApiError, type QueryResult, type Question } from "../api/query";
import { transcribeAudio, TranscriptionApiError } from "../api/transcription";

interface AudioSubmission {
  audio: Blob;
  language: string;
}

type Submission =
  | { kind: "audio"; value: AudioSubmission }
  | { kind: "question"; value: Question };

interface QuestionState {
  status: "idle" | "transcribing" | "querying" | "complete" | "error";
  question: string | null;
  result: QueryResult | null;
  error: { message: string; retryable: boolean; stage: Submission["kind"] } | null;
}

const initialState: QuestionState = {
  status: "idle",
  question: null,
  result: null,
  error: null,
};

export function useQuestion() {
  const [state, setState] = useState<QuestionState>(initialState);
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const lastAudioRef = useRef<Blob | null>(null);
  const retrySubmissionRef = useRef<Submission | null>(null);

  const run = useCallback(async (submission: Submission) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    const requestId = ++requestIdRef.current;
    controllerRef.current = controller;
    retrySubmissionRef.current = submission;
    let currentSubmission = submission;
    const isCurrent = () => !controller.signal.aborted && requestId === requestIdRef.current;

    setState({
      ...initialState,
      status: submission.kind === "audio" ? "transcribing" : "querying",
      question: submission.kind === "question" ? submission.value.question : null,
    });

    try {
      if (currentSubmission.kind === "audio") {
        const transcript = await transcribeAudio({
          ...currentSubmission.value,
          signal: controller.signal,
        });
        if (!isCurrent()) return;
        const question = transcript.text.trim();
        if (!question) {
          throw new TranscriptionApiError("No words were recognized. Try recording again or type your question.", {
            code: "empty_transcript", retryable: false, status: 200,
          });
        }
        currentSubmission = {
          kind: "question",
          value: { question, language: transcript.language ?? null },
        };
        // Once speech has been transcribed, retries only resend the text.
        retrySubmissionRef.current = currentSubmission;
        setState({ ...initialState, status: "querying", question });
      }

      const result = await askQuestion(currentSubmission.value, controller.signal);
      if (!isCurrent()) return;
      controllerRef.current = null;
      setState({ status: "complete", question: currentSubmission.value.question, result, error: null });
    } catch (error) {
      if (!isCurrent()) return;
      controllerRef.current = null;
      const knownError = error instanceof QueryApiError || error instanceof TranscriptionApiError;
      setState((previous) => ({
        ...previous,
        status: "error",
        error: {
          message: knownError ? error.message : "We couldn’t reach the service. Check your connection and try again.",
          retryable: knownError ? error.retryable : true,
          stage: currentSubmission.kind,
        },
      }));
    }
  }, []);

  const submitAudio = useCallback((submission: AudioSubmission) => {
    if (lastAudioRef.current === submission.audio) return;
    lastAudioRef.current = submission.audio;
    void run({ kind: "audio", value: submission });
  }, [run]);

  const submitText = useCallback((question: Question) => {
    const text = question.question.trim();
    if (!text) return;
    void run({ kind: "question", value: { ...question, question: text } });
  }, [run]);

  const retry = useCallback(() => {
    if (retrySubmissionRef.current) void run(retrySubmissionRef.current);
  }, [run]);

  const reset = useCallback(() => {
    ++requestIdRef.current;
    controllerRef.current?.abort();
    controllerRef.current = null;
    retrySubmissionRef.current = null;
    lastAudioRef.current = null;
    setState(initialState);
  }, []);

  useEffect(() => () => {
    ++requestIdRef.current;
    controllerRef.current?.abort();
  }, []);

  return { ...state, submitAudio, submitText, retry, reset };
}
