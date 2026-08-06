import { json } from "@sveltejs/kit";

import {
  getFeedbackLoopStatus,
  getPinnedPredictionRunId,
  getPolicyUpdateHistory,
  getRewardTrend,
  getSuccessHistory,
  getSuccessTrend,
} from "$lib/server/ml/policy";

export const GET = async ({ url }) => {
  const limitParam = url.searchParams.get("limit");
  const limit = limitParam ? Number(limitParam) : 120;
  const runIdParam = url.searchParams.get("runId");
  const runId = runIdParam ? Number(runIdParam) : getPinnedPredictionRunId();

  if (!Number.isInteger(limit) || limit <= 0 || limit > 500) {
    return json(
      {
        success: false,
        message: "limit must be a positive integer between 1 and 500.",
      },
      { status: 400 },
    );
  }

  if (runId !== null && (!Number.isInteger(runId) || runId <= 0)) {
    return json(
      { success: false, message: "runId must be a positive integer." },
      { status: 400 },
    );
  }

  return json({
    success: true,
    feedbackStatus: getFeedbackLoopStatus(),
    history: getPolicyUpdateHistory(limit),
    trend: getRewardTrend(limit),
    selectedRunId: runId,
    successHistory: getSuccessHistory(limit, runId),
    successTrend: getSuccessTrend(limit, runId),
  });
};
