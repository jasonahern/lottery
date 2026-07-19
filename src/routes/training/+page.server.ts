import { fail } from "@sveltejs/kit";

import {
  attachDecisionToRun,
  clearPinnedPredictionRunId,
  defaultPolicyConfig,
  getFeedbackLoopStatus,
  getLatestOpenDecision,
  getPinnedPredictionRunId,
  setPinnedPredictionRunId,
  setPolicyMode,
  type PolicyMode,
  type PolicyTunableConfig,
} from "$lib/server/ml/policy";
import {
  DEFAULT_TRAINING_CONFIG,
  parseHiddenLayersInput,
  resolveHiddenLayers,
  type TrainingPreset,
} from "$lib/server/ml/config";
import {
  createTrainingRun,
  getDatasetStats,
  getLatestRuns,
  getPredictionRunCandidates,
  getRunComparisonSummaries,
  getRunDetail,
  getRunProgress,
  getTrainingRecommendation,
  type RunSummary,
} from "$lib/server/ml/runs";
import { getNextDrawPrediction } from "$lib/server/ml/prediction";
import { startTrainingRun } from "$lib/server/ml/trainer";
import type { Actions } from "./$types";

function coercePositiveInt(
  value: FormDataEntryValue | null,
  fallback: number,
): number {
  if (typeof value !== "string" || value.trim().length === 0) {
    return fallback;
  }

  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    return fallback;
  }

  return parsed;
}

function coercePositiveNumber(
  value: FormDataEntryValue | null,
  fallback: number,
): number {
  if (typeof value !== "string" || value.trim().length === 0) {
    return fallback;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }

  return parsed;
}

function coerceNonNegativeNumber(
  value: FormDataEntryValue | null,
  fallback: number,
): number {
  if (typeof value !== "string" || value.trim().length === 0) {
    return fallback;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return fallback;
  }

  return parsed;
}

export const load = async ({ url }) => {
  const runIdParam = url.searchParams.get("runId");
  const runId = runIdParam ? Number(runIdParam) : null;

  const latestRuns = getLatestRuns(20);
  const activeRun =
    runId && Number.isInteger(runId) && runId > 0
      ? getRunProgress(runId)
      : (latestRuns[0] ?? null);
  const predictionRunCandidates = getPredictionRunCandidates(40);
  const pinnedPredictionRunId = getPinnedPredictionRunId();
  const holdoutResultsRunId = pinnedPredictionRunId ?? activeRun?.id ?? null;
  const holdoutRunDetail = holdoutResultsRunId
    ? getRunDetail(holdoutResultsRunId, { epochs: 300, tests: 50 })
    : null;

  let nextDrawPrediction = null;
  let predictionRunMessage: string | null = null;

  if (pinnedPredictionRunId) {
    nextDrawPrediction = await getNextDrawPrediction(pinnedPredictionRunId);
    if (!nextDrawPrediction) {
      predictionRunMessage =
        "Pinned prediction run is not currently available. Using automatic fallback run.";
      nextDrawPrediction =
        activeRun?.status === "completed"
          ? await getNextDrawPrediction(activeRun.id)
          : await getNextDrawPrediction();
    }
  } else {
    nextDrawPrediction =
      activeRun?.status === "completed"
        ? await getNextDrawPrediction(activeRun.id)
        : await getNextDrawPrediction();
  }

  return {
    defaults: DEFAULT_TRAINING_CONFIG,
    latestRuns,
    activeRun,
    holdoutRunDetail,
    holdoutResultsRunId,
    nextDrawPrediction,
    datasetStats: getDatasetStats(),
    recommendation: getTrainingRecommendation(),
    feedbackStatus: getFeedbackLoopStatus(),
    predictionRunCandidates,
    runComparisonSummaries: getRunComparisonSummaries(20),
    pinnedPredictionRunId,
    predictionRunMessage,
  };
};

export const actions: Actions = {
  setPredictionRun: async ({ request }) => {
    const form = await request.formData();
    const runId = Number(form.get("predictionRunId"));

    if (!Number.isInteger(runId) || runId <= 0) {
      return fail(400, {
        success: false,
        predictionMessage: "Please choose a valid completed run.",
      });
    }

    const candidates = getPredictionRunCandidates(200);
    const exists = candidates.some((candidate) => candidate.id === runId);
    if (!exists) {
      return fail(404, {
        success: false,
        predictionMessage:
          "Selected run is not available for fixed-weight prediction.",
      });
    }

    try {
      setPinnedPredictionRunId(runId);
      return {
        success: true,
        predictionMessage: `Pinned run #${runId} for next-draw predictions.`,
      };
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to pin prediction run.";
      return fail(500, {
        success: false,
        predictionMessage: message,
      });
    }
  },

  clearPredictionRun: async () => {
    try {
      clearPinnedPredictionRunId();
      return {
        success: true,
        predictionMessage:
          "Switched back to automatic run selection for predictions.",
      };
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to clear pinned prediction run.";
      return fail(500, {
        success: false,
        predictionMessage: message,
      });
    }
  },

  setPolicyMode: async ({ request }) => {
    const form = await request.formData();
    const requested = form.get("policyMode");

    if (
      requested !== "off" &&
      requested !== "shadow" &&
      requested !== "active"
    ) {
      return fail(400, {
        success: false,
        policyMessage: "Policy mode must be one of: off, shadow, active.",
      });
    }

    try {
      setPolicyMode(requested as PolicyMode);
      return {
        success: true,
        policyMessage: `Policy mode updated to ${requested}.`,
      };
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to update policy mode.";
      return fail(500, {
        success: false,
        policyMessage: message,
      });
    }
  },

  quickStart: async () => {
    try {
      const run = createTrainingRun();
      const started = startTrainingRun(run.id);

      return {
        success: true,
        message: started
          ? `Quick Start launched run #${run.id} using default settings.`
          : `Quick Start created run #${run.id}. Training is already active for this run.`,
        run,
      };
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Quick Start failed.";
      return fail(400, {
        success: false,
        message,
      });
    }
  },

  createRun: async ({ request }) => {
    const form = await request.formData();

    const preset = form.get("preset");
    const hiddenLayersInput = form.get("hiddenLayers");

    const resolvedPreset: TrainingPreset | undefined =
      preset === "small" ||
      preset === "medium" ||
      preset === "large" ||
      preset === "xlarge"
        ? preset
        : undefined;

    const customHiddenLayers =
      typeof hiddenLayersInput === "string" &&
      hiddenLayersInput.trim().length > 0
        ? parseHiddenLayersInput(hiddenLayersInput)
        : undefined;

    const hiddenLayers = resolveHiddenLayers(
      resolvedPreset,
      customHiddenLayers,
    );

    const submittedConfig: PolicyTunableConfig = {
      windowSize: coercePositiveInt(
        form.get("windowSize"),
        DEFAULT_TRAINING_CONFIG.windowSize,
      ),
      epochs: coercePositiveInt(
        form.get("epochs"),
        DEFAULT_TRAINING_CONFIG.epochs,
      ),
      batchSize: coercePositiveInt(
        form.get("batchSize"),
        DEFAULT_TRAINING_CONFIG.batchSize,
      ),
      learningRate: coercePositiveNumber(
        form.get("learningRate"),
        DEFAULT_TRAINING_CONFIG.learningRate,
      ),
      dropoutRate: coerceNonNegativeNumber(
        form.get("dropoutRate"),
        DEFAULT_TRAINING_CONFIG.dropoutRate,
      ),
    };

    try {
      const run = createTrainingRun({
        holdoutWeeks: coercePositiveInt(
          form.get("holdoutWeeks"),
          DEFAULT_TRAINING_CONFIG.holdoutWeeks,
        ),
        windowSize: submittedConfig.windowSize,
        epochs: submittedConfig.epochs,
        batchSize: submittedConfig.batchSize,
        learningRate: submittedConfig.learningRate,
        dropoutRate: submittedConfig.dropoutRate,
        positiveClassWeight: coercePositiveNumber(
          form.get("positiveClassWeight"),
          DEFAULT_TRAINING_CONFIG.positiveClassWeight,
        ),
        earlyStoppingPatience: coercePositiveInt(
          form.get("earlyStoppingPatience"),
          DEFAULT_TRAINING_CONFIG.earlyStoppingPatience,
        ),
        earlyStoppingMinDelta: coercePositiveNumber(
          form.get("earlyStoppingMinDelta"),
          DEFAULT_TRAINING_CONFIG.earlyStoppingMinDelta,
        ),
        trainingSeed: coercePositiveInt(
          form.get("trainingSeed"),
          DEFAULT_TRAINING_CONFIG.trainingSeed,
        ),
        hiddenLayers,
      });

      const openDecision = getLatestOpenDecision();
      if (openDecision) {
        let decisionConfig = defaultPolicyConfig();
        try {
          const action = JSON.parse(
            openDecision.actionJson,
          ) as Partial<PolicyTunableConfig>;
          decisionConfig = {
            windowSize: Number(action.windowSize) || decisionConfig.windowSize,
            epochs: Number(action.epochs) || decisionConfig.epochs,
            batchSize: Number(action.batchSize) || decisionConfig.batchSize,
            learningRate:
              Number(action.learningRate) || decisionConfig.learningRate,
            dropoutRate:
              Number(action.dropoutRate) || decisionConfig.dropoutRate,
          };
        } catch {
          decisionConfig = defaultPolicyConfig();
        }

        const wasOverridden =
          decisionConfig.windowSize !== submittedConfig.windowSize ||
          decisionConfig.epochs !== submittedConfig.epochs ||
          decisionConfig.batchSize !== submittedConfig.batchSize ||
          Math.abs(decisionConfig.learningRate - submittedConfig.learningRate) >
            0.0000001 ||
          Math.abs(decisionConfig.dropoutRate - submittedConfig.dropoutRate) >
            0.0000001;

        attachDecisionToRun({
          decisionId: openDecision.id,
          runId: run.id,
          wasOverridden,
          overrideConfig: wasOverridden ? submittedConfig : undefined,
        });
      }

      const started = startTrainingRun(run.id);

      return {
        success: true,
        message: started
          ? `Created training run #${run.id} and started training.`
          : `Created training run #${run.id}. Training is already active for this run.`,
        run,
      };
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to create run.";
      return fail(400, {
        success: false,
        message,
      });
    }
  },

  progress: async ({ request }) => {
    const form = await request.formData();
    const runId = Number(form.get("runId"));

    if (!Number.isInteger(runId) || runId <= 0) {
      return fail(400, {
        success: false,
        message: "runId must be a positive integer.",
      });
    }

    const progress = getRunProgress(runId);
    if (!progress) {
      return fail(404, {
        success: false,
        message: `Run #${runId} was not found.`,
      });
    }

    return {
      success: true,
      progress,
    } satisfies { success: boolean; progress: RunSummary };
  },
};
