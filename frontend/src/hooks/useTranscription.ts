import { useCallback, useEffect, useRef, useState } from "react";

import {
  TranscriptionApiError,
  transcribeAudio,
  type ProviderSelection,
  type TranscriptionProvider,
  type TranscriptionResult,
} from "../api/transcription";

export type TranscriptionStatus = "idle" | "uploading" | "success" | "partial" | "error";

export type ProviderOutcome =
  | {
      provider: TranscriptionProvider;
      status: "success";
      result: TranscriptionResult;
    }
  | {
      provider: TranscriptionProvider;
      status: "error";
      code: string;
      message: string;
      retryable: boolean;
    };

interface Submission {
  audio: Blob;
  language: string;
  selection: ProviderSelection;
}

interface TranscriptionState {
  status: TranscriptionStatus;
  outcomes: ProviderOutcome[];
}

const initialState: TranscriptionState = {
  status: "idle",
  outcomes: [],
};

function providerError(provider: TranscriptionProvider, error: unknown): ProviderOutcome {
  if (error instanceof TranscriptionApiError) {
    return {
      provider,
      status: "error",
      code: error.code,
      message: error.message,
      retryable: error.retryable,
    };
  }

  return {
    provider,
    status: "error",
    code: "network_error",
    message: "The transcription service could not be reached.",
    retryable: true,
  };
}

export function useTranscription() {
  const [state, setState] = useState<TranscriptionState>(initialState);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastAudioRef = useRef<Blob | null>(null);
  const lastOutcomesRef = useRef<ProviderOutcome[]>([]);
  const lastSubmissionRef = useRef<Submission | null>(null);
  const requestIdRef = useRef(0);

  const runProviders = useCallback(async (
    submission: Submission,
    providers: TranscriptionProvider[],
    retainedOutcomes: ProviderOutcome[],
  ) => {
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    const requestId = requestIdRef.current + 1;

    abortControllerRef.current = controller;
    requestIdRef.current = requestId;
    setState({ status: "uploading", outcomes: retainedOutcomes });

    const retriedOutcomes = await Promise.all(
      providers.map(async (provider): Promise<ProviderOutcome> => {
        try {
          const result = await transcribeAudio({
            audio: submission.audio,
            language: submission.language,
            provider,
            signal: controller.signal,
          });
          return { provider, status: "success", result };
        } catch (error) {
          return providerError(provider, error);
        }
      }),
    );

    if (controller.signal.aborted || requestId !== requestIdRef.current) {
      return;
    }

    abortControllerRef.current = null;
    const outcomeByProvider = new Map(
      [...retainedOutcomes, ...retriedOutcomes].map((outcome) => [
        outcome.provider,
        outcome,
      ]),
    );
    const providerOrder: TranscriptionProvider[] =
      submission.selection === "compare"
        ? ["sarvam", "elevenlabs"]
        : [submission.selection];
    const outcomes = providerOrder.flatMap((provider) => {
      const outcome = outcomeByProvider.get(provider);
      return outcome ? [outcome] : [];
    });
    const successfulCount = outcomes.filter((outcome) => outcome.status === "success").length;
    lastOutcomesRef.current = outcomes;
    setState({
      outcomes,
      status:
        successfulCount === outcomes.length
          ? "success"
          : successfulCount > 0
            ? "partial"
            : "error",
    });
  }, []);

  const submit = useCallback(async (submission: Submission) => {
    if (lastAudioRef.current === submission.audio) {
      return;
    }

    const providers: TranscriptionProvider[] =
      submission.selection === "compare"
        ? ["sarvam", "elevenlabs"]
        : [submission.selection];

    lastAudioRef.current = submission.audio;
    lastOutcomesRef.current = [];
    lastSubmissionRef.current = submission;
    await runProviders(submission, providers, []);
  }, [runProviders]);

  const retry = useCallback(() => {
    const submission = lastSubmissionRef.current;
    if (!submission) {
      return;
    }

    const retryProviders = lastOutcomesRef.current.flatMap((outcome) =>
      outcome.status === "error" && outcome.retryable ? [outcome.provider] : [],
    );
    if (retryProviders.length === 0) {
      return;
    }

    const retainedOutcomes = lastOutcomesRef.current.filter(
      (outcome) => !retryProviders.includes(outcome.provider),
    );
    void runProviders(submission, retryProviders, retainedOutcomes);
  }, [runProviders]);

  const reset = useCallback(() => {
    requestIdRef.current += 1;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    lastAudioRef.current = null;
    lastOutcomesRef.current = [];
    lastSubmissionRef.current = null;
    setState(initialState);
  }, []);

  useEffect(
    () => () => {
      requestIdRef.current += 1;
      abortControllerRef.current?.abort();
    },
    [],
  );

  return {
    ...state,
    canRetry: state.outcomes.some(
      (outcome) => outcome.status === "error" && outcome.retryable,
    ),
    reset,
    retry,
    submit,
  };
}
