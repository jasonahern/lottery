import { json } from "@sveltejs/kit";

import {
  getFeedbackLoopStatus,
  getPolicyUpdateHistory,
  getRewardTrend,
  getSuccessHistory,
  getSuccessTrend,
} from "$lib/server/ml/policy";

export const GET = async ({ url }) => {
  const limitParam = url.searchParams.get("limit");
  const limit = limitParam ? Number(limitParam) : 120;

  if (!Number.isInteger(limit) || limit <= 0 || limit > 500) {
    return json(
      {
        success: false,
        message: "limit must be a positive integer between 1 and 500.",
      },
      { status: 400 },
    );
  }

  return json({
    success: true,
    feedbackStatus: getFeedbackLoopStatus(),
    history: getPolicyUpdateHistory(limit),
    trend: getRewardTrend(limit),
    successHistory: getSuccessHistory(limit),
    successTrend: getSuccessTrend(limit),
  });
};
