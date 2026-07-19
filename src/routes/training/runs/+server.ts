import { json } from "@sveltejs/kit";

import {
  getLatestRuns,
  getRunDetail,
  getTrainingRecommendation,
} from "$lib/server/ml/runs";
import { getFeedbackLoopStatus } from "$lib/server/ml/policy";

export const GET = async ({ url }) => {
  const runIdParam = url.searchParams.get("runId");
  const limitParam = url.searchParams.get("limit");

  const runId = runIdParam ? Number(runIdParam) : null;
  const limit = limitParam ? Number(limitParam) : 20;

  if (runId !== null) {
    if (!Number.isInteger(runId) || runId <= 0) {
      return json(
        { success: false, message: "runId must be a positive integer." },
        { status: 400 },
      );
    }

    const detail = getRunDetail(runId, { epochs: 500, tests: 200 });
    if (!detail) {
      return json(
        { success: false, message: `Run #${runId} not found.` },
        { status: 404 },
      );
    }

    return json({
      success: true,
      detail,
    });
  }

  if (!Number.isInteger(limit) || limit <= 0 || limit > 200) {
    return json(
      {
        success: false,
        message: "limit must be a positive integer between 1 and 200.",
      },
      { status: 400 },
    );
  }

  return json({
    success: true,
    runs: getLatestRuns(limit),
    recommendation: getTrainingRecommendation(),
    feedbackStatus: getFeedbackLoopStatus(),
  });
};
