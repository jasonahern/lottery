import {
  getFeedbackLoopStatus,
  getPolicyUpdateHistory,
  getRewardTrend,
  getSuccessHistory,
  getSuccessTrend,
} from "$lib/server/ml/policy";

export const load = async () => {
  const feedbackStatus = getFeedbackLoopStatus();
  const history = getPolicyUpdateHistory(120);
  const trend = getRewardTrend(120);
  const successHistory = getSuccessHistory(120);
  const successTrend = getSuccessTrend(120);

  return {
    feedbackStatus,
    history,
    trend,
    successHistory,
    successTrend,
  };
};
