import {
  getFeedbackLoopStatus,
  getPinnedPredictionRunId,
  getPolicyUpdateHistory,
  getRewardTrend,
  getSuccessHistory,
  getSuccessTrend,
  resolvePendingFeedback,
} from "$lib/server/ml/policy";
import type { Actions } from "./$types";

export const load = async () => {
  const selectedRunId = getPinnedPredictionRunId();
  const feedbackStatus = getFeedbackLoopStatus();
  const history = getPolicyUpdateHistory(120);
  const trend = getRewardTrend(120);
  const successHistory = getSuccessHistory(120, selectedRunId);
  const successTrend = getSuccessTrend(120, selectedRunId);

  return {
    feedbackStatus,
    history,
    trend,
    successHistory,
    successTrend,
    selectedRunId,
  };
};

export const actions: Actions = {
  refreshResults: async () => {
    const selectedRunId = getPinnedPredictionRunId();
    const { resolvedCount } = resolvePendingFeedback(200, selectedRunId);
    return {
      success: true,
      refreshMessage:
        resolvedCount > 0
          ? `Refreshed ${selectedRunId ? `run #${selectedRunId}` : "all runs"}: ${resolvedCount} new prediction outcome${resolvedCount === 1 ? "" : "s"} scored.`
          : `${selectedRunId ? `Run #${selectedRunId}` : "Results"} is up to date; there were no new saved predictions to score.`,
    };
  },
};
