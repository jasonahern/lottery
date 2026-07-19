import {
  choosePolicyDecision,
  computeRewardV1,
  defaultPolicyConfig,
  type PolicySignal,
  type PolicyTunableConfig,
} from "./policy";

export type ReplayEvent = {
  signal: PolicySignal;
  matchCount: number;
  topKHit: boolean;
};

export type ReplaySummary = {
  events: number;
  averageReward: number;
  rewards: number[];
  configsUsed: PolicyTunableConfig[];
};

export function replayPolicyHistory(
  events: ReplayEvent[],
  initialConfig: PolicyTunableConfig = defaultPolicyConfig(),
): ReplaySummary {
  if (events.length === 0) {
    return {
      events: 0,
      averageReward: 0,
      rewards: [],
      configsUsed: [],
    };
  }

  let currentConfig = { ...initialConfig };
  const rewards: number[] = [];
  const configsUsed: PolicyTunableConfig[] = [];

  for (const [idx, event] of events.entries()) {
    const decision = choosePolicyDecision({
      baseConfig: currentConfig,
      signal: event.signal,
      randomValue: (idx % 10) / 10,
      context: {
        latestRunId: idx + 1,
        signal: event.signal,
        latestTrainLoss: null,
        latestValLoss: null,
        latestHoldoutScore: null,
      },
    });

    currentConfig = { ...decision.config };
    configsUsed.push(currentConfig);

    const reward = computeRewardV1(event.matchCount, event.topKHit);
    rewards.push(reward);
  }

  const averageReward = rewards.reduce((acc, n) => acc + n, 0) / rewards.length;

  return {
    events: events.length,
    averageReward: Number(averageReward.toFixed(4)),
    rewards,
    configsUsed,
  };
}
