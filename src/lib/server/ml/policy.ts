import { and, desc, eq, isNotNull, sql } from "drizzle-orm";

import { db } from "$lib/server/db";
import {
  lotteryDrawBalls,
  lotteryDraws,
  nnFeedbackOutcomes,
  nnPolicySettings,
  nnPolicyUpdates,
  nnPredictionDecisions,
} from "$lib/server/db/schema";

import {
  DEFAULT_TRAINING_CONFIG,
  type TrainingConfig,
  type TrainingPreset,
} from "./config";

export type PolicyMode = "off" | "shadow" | "active";

export type PolicySignal = "regularize" | "scale_up" | "keep";

export type PolicyTunableConfig = Pick<
  TrainingConfig,
  "windowSize" | "epochs" | "batchSize" | "learningRate" | "dropoutRate"
>;

export type PolicyContext = {
  latestRunId: number | null;
  signal: PolicySignal;
  latestTrainLoss: number | null;
  latestValLoss: number | null;
  latestHoldoutScore: number | null;
  distinctNumberCount?: number;
  drawCount?: number;
  recentAverageReward: number;
  recentRewardCount: number;
};

export type PolicyAction = PolicyTunableConfig & {
  actionKey: string;
};

export type PolicyDecision = {
  config: PolicyTunableConfig;
  actionKey: string;
  algorithmVersion: string;
  explorationRate: number;
  wasExploration: boolean;
  mode: PolicyMode;
  source: "policy" | "fallback";
  rationale: string;
  context: PolicyContext;
};

export type FeedbackLoopStatus = {
  mode: PolicyMode;
  pendingDecisions: number;
  resolvedOutcomes: number;
  recentAverageReward: number;
  recentRewardCount: number;
  lastReward: number | null;
  lastResolvedAt: string | null;
  explorationRate: number;
};

export type PolicyUpdateHistoryItem = {
  id: number;
  decisionId: number | null;
  algorithmVersion: string;
  policyMode: PolicyMode;
  rewardValue: number;
  baselineReward: number | null;
  advantage: number | null;
  explorationRate: number;
  createdAt: string;
};

export type RewardTrendPoint = {
  index: number;
  reward: number;
  rollingAverage: number;
  createdAt: string;
};

export type SuccessHistoryItem = {
  id: number;
  decisionId: number;
  drawId: number | null;
  matchCount: number;
  actualCount: number;
  successPercent: number;
  createdAt: string;
};

export type SuccessTrendPoint = {
  index: number;
  successPercent: number;
  rollingAverage: number;
  createdAt: string;
};

export const REWARD_VERSION = "v1";
export const POLICY_ALGORITHM_VERSION = "bandit_v1";
export const DEFAULT_EXPLORATION_RATE = 0.1;
export const ACTIVE_MODE_MIN_AVG_REWARD = -0.25;
export const ACTIVE_MODE_MIN_SAMPLE_COUNT = 20;
const POLICY_MODE_SETTING_KEY = "policy_mode";
const PINNED_PREDICTION_RUN_ID_KEY = "pinned_prediction_run_id";

function getSettingValue(key: string): string | null {
  try {
    const row = db
      .select({ value: nnPolicySettings.value })
      .from(nnPolicySettings)
      .where(eq(nnPolicySettings.key, key))
      .orderBy(desc(nnPolicySettings.id))
      .get();

    return row?.value ?? null;
  } catch {
    return null;
  }
}

function setSettingValue(key: string, value: string): void {
  const existing = (() => {
    try {
      return db
        .select({ id: nnPolicySettings.id })
        .from(nnPolicySettings)
        .where(eq(nnPolicySettings.key, key))
        .orderBy(desc(nnPolicySettings.id))
        .get();
    } catch {
      return null;
    }
  })();

  if (existing) {
    db.update(nnPolicySettings)
      .set({ value, updatedAt: new Date() })
      .where(eq(nnPolicySettings.id, existing.id))
      .run();
    return;
  }

  db.insert(nnPolicySettings)
    .values({
      key,
      value,
      updatedAt: new Date(),
    })
    .run();
}

function clearSettingValue(key: string): void {
  try {
    db.delete(nnPolicySettings).where(eq(nnPolicySettings.key, key)).run();
  } catch {
    // Ignore when settings table is not yet migrated.
  }
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }

  return Math.max(min, Math.min(max, Math.round(value)));
}

function clampFloat(
  value: number,
  min: number,
  max: number,
  precision = 6,
): number {
  if (!Number.isFinite(value)) {
    return min;
  }

  const bounded = Math.max(min, Math.min(max, value));
  return Number(bounded.toFixed(precision));
}

export function getPolicyMode(): PolicyMode {
  const dbValue = getSettingValue(POLICY_MODE_SETTING_KEY)?.toLowerCase();
  if (dbValue === "off" || dbValue === "shadow" || dbValue === "active") {
    return dbValue;
  }

  const raw = (process.env.POLICY_MODE ?? "shadow").toLowerCase();
  if (raw === "off" || raw === "shadow" || raw === "active") {
    return raw;
  }

  return "shadow";
}

export function setPolicyMode(mode: PolicyMode): void {
  setSettingValue(POLICY_MODE_SETTING_KEY, mode);
}

export function getPinnedPredictionRunId(): number | null {
  const raw = getSettingValue(PINNED_PREDICTION_RUN_ID_KEY);
  if (!raw) {
    return null;
  }

  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    return null;
  }

  return parsed;
}

export function setPinnedPredictionRunId(runId: number): void {
  setSettingValue(PINNED_PREDICTION_RUN_ID_KEY, String(runId));
}

export function clearPinnedPredictionRunId(): void {
  clearSettingValue(PINNED_PREDICTION_RUN_ID_KEY);
}

export function computeRewardV1(
  matchCount: number,
  topKHit: boolean,
  overfitPenalty = 0,
): number {
  let reward = -0.5;

  if (matchCount >= 3) {
    reward = 3;
  } else if (matchCount === 2) {
    reward = 1;
  } else if (matchCount === 1 || topKHit) {
    reward = 0.2;
  }

  reward -= Math.max(0, overfitPenalty);

  return Number(reward.toFixed(4));
}

function actionKeyFromConfig(config: PolicyTunableConfig): string {
  return [
    config.windowSize,
    config.epochs,
    config.batchSize,
    config.learningRate,
    config.dropoutRate,
  ].join("|");
}

function toTunableConfig(config: PolicyTunableConfig): PolicyTunableConfig {
  return {
    windowSize: clampInt(config.windowSize, 1, 52),
    epochs: clampInt(config.epochs, 1, 500),
    batchSize: clampInt(config.batchSize, 1, 1024),
    learningRate: clampFloat(config.learningRate, 0.000001, 0.1),
    dropoutRate: clampFloat(config.dropoutRate, 0, 0.99),
  };
}

function buildActionCandidates(
  baseConfig: PolicyTunableConfig,
  signal: PolicySignal,
): PolicyAction[] {
  const base = toTunableConfig(baseConfig);

  const candidates: PolicyTunableConfig[] = [base];

  if (signal === "regularize") {
    candidates.push(
      {
        ...base,
        windowSize: base.windowSize + 2,
        epochs: Math.floor(base.epochs * 0.8),
        batchSize: Math.floor(base.batchSize * 0.75),
        learningRate: base.learningRate * 0.7,
        dropoutRate: base.dropoutRate + 0.05,
      },
      {
        ...base,
        windowSize: base.windowSize + 1,
        epochs: Math.floor(base.epochs * 0.9),
        batchSize: Math.floor(base.batchSize * 0.9),
        learningRate: base.learningRate * 0.8,
        dropoutRate: base.dropoutRate + 0.08,
      },
    );
  } else if (signal === "scale_up") {
    candidates.push(
      {
        ...base,
        windowSize: base.windowSize + 1,
        epochs: Math.floor(base.epochs * 1.15),
        batchSize: Math.floor(base.batchSize * 1.25),
        learningRate: base.learningRate * 0.85,
        dropoutRate: base.dropoutRate + 0.02,
      },
      {
        ...base,
        windowSize: base.windowSize,
        epochs: Math.floor(base.epochs * 1.25),
        batchSize: Math.floor(base.batchSize * 1.1),
        learningRate: base.learningRate * 0.9,
        dropoutRate: base.dropoutRate + 0.01,
      },
    );
  } else {
    candidates.push(
      {
        ...base,
        windowSize: base.windowSize,
        epochs: Math.floor(base.epochs * 1.05),
        batchSize: base.batchSize,
        learningRate: base.learningRate * 0.95,
        dropoutRate: base.dropoutRate + 0.01,
      },
      {
        ...base,
        windowSize: base.windowSize + 1,
        epochs: Math.floor(base.epochs * 0.95),
        batchSize: Math.floor(base.batchSize * 0.95),
        learningRate: base.learningRate,
        dropoutRate: base.dropoutRate,
      },
    );
  }

  const deduped = new Map<string, PolicyAction>();
  for (const candidate of candidates) {
    const normalized = toTunableConfig(candidate);
    const actionKey = actionKeyFromConfig(normalized);
    if (!deduped.has(actionKey)) {
      deduped.set(actionKey, {
        ...normalized,
        actionKey,
      });
    }
  }

  return [...deduped.values()];
}

function getActionRewardAverages(limit = 400): Map<string, number> {
  let rows: Array<{ chosenActionJson: string; rewardValue: string }> = [];
  try {
    rows = db
      .select({
        chosenActionJson: nnPolicyUpdates.chosenActionJson,
        rewardValue: nnPolicyUpdates.rewardValue,
      })
      .from(nnPolicyUpdates)
      .orderBy(desc(nnPolicyUpdates.id))
      .limit(limit)
      .all();
  } catch {
    return new Map();
  }

  const running = new Map<string, { total: number; count: number }>();
  for (const row of rows) {
    const reward = Number(row.rewardValue);
    if (!Number.isFinite(reward)) {
      continue;
    }

    let actionKey: string | null = null;
    try {
      const parsed = JSON.parse(row.chosenActionJson) as { actionKey?: string };
      actionKey = parsed.actionKey ?? null;
    } catch {
      actionKey = null;
    }

    if (!actionKey) {
      continue;
    }

    const found = running.get(actionKey) ?? { total: 0, count: 0 };
    found.total += reward;
    found.count += 1;
    running.set(actionKey, found);
  }

  const averaged = new Map<string, number>();
  for (const [key, value] of running.entries()) {
    if (value.count > 0) {
      averaged.set(key, value.total / value.count);
    }
  }

  return averaged;
}

function summarizeRecentRewards(limit = 50): {
  average: number;
  count: number;
  lastReward: number | null;
  lastResolvedAt: string | null;
} {
  let rows: Array<{ rewardValue: string; createdAt: Date }> = [];
  try {
    rows = db
      .select({
        rewardValue: nnFeedbackOutcomes.rewardValue,
        createdAt: nnFeedbackOutcomes.createdAt,
      })
      .from(nnFeedbackOutcomes)
      .orderBy(desc(nnFeedbackOutcomes.id))
      .limit(limit)
      .all();
  } catch {
    return {
      average: 0,
      count: 0,
      lastReward: null,
      lastResolvedAt: null,
    };
  }

  if (rows.length === 0) {
    return {
      average: 0,
      count: 0,
      lastReward: null,
      lastResolvedAt: null,
    };
  }

  const rewards = rows
    .map((row) => Number(row.rewardValue))
    .filter((value) => Number.isFinite(value));

  const average =
    rewards.length > 0
      ? rewards.reduce((acc, value) => acc + value, 0) / rewards.length
      : 0;

  return {
    average,
    count: rewards.length,
    lastReward: rewards.length > 0 ? rewards[0] : null,
    lastResolvedAt: rows[0]?.createdAt?.toISOString() ?? null,
  };
}

export function choosePolicyDecision(params: {
  baseConfig: PolicyTunableConfig;
  context: Omit<PolicyContext, "recentAverageReward" | "recentRewardCount">;
  signal: PolicySignal;
  randomValue?: number;
}): PolicyDecision {
  const mode = getPolicyMode();
  const rewardSummary = summarizeRecentRewards();
  const context: PolicyContext = {
    ...params.context,
    signal: params.signal,
    recentAverageReward: Number(rewardSummary.average.toFixed(4)),
    recentRewardCount: rewardSummary.count,
  };

  const fallbackConfig = toTunableConfig(params.baseConfig);

  if (mode === "off") {
    return {
      config: fallbackConfig,
      actionKey: actionKeyFromConfig(fallbackConfig),
      algorithmVersion: POLICY_ALGORITHM_VERSION,
      explorationRate: DEFAULT_EXPLORATION_RATE,
      wasExploration: false,
      mode,
      source: "fallback",
      rationale: "Policy mode is off; using fallback recommendation.",
      context,
    };
  }

  const shouldSafetyFallback =
    mode === "active" &&
    rewardSummary.count >= ACTIVE_MODE_MIN_SAMPLE_COUNT &&
    rewardSummary.average < ACTIVE_MODE_MIN_AVG_REWARD;

  if (shouldSafetyFallback) {
    return {
      config: fallbackConfig,
      actionKey: actionKeyFromConfig(fallbackConfig),
      algorithmVersion: POLICY_ALGORITHM_VERSION,
      explorationRate: DEFAULT_EXPLORATION_RATE,
      wasExploration: false,
      mode,
      source: "fallback",
      rationale:
        "Safety fallback: recent average reward is below threshold in active mode.",
      context,
    };
  }

  const candidates = buildActionCandidates(params.baseConfig, params.signal);
  const averagesByAction = getActionRewardAverages();
  const epsilon = DEFAULT_EXPLORATION_RATE;
  const randomDraw =
    params.randomValue === undefined ? Math.random() : params.randomValue;

  const shouldExplore = randomDraw < epsilon;

  let chosen: PolicyAction;
  if (shouldExplore) {
    chosen = candidates[Math.floor((randomDraw * 1000) % candidates.length)];
  } else {
    let best = candidates[0];
    let bestScore = averagesByAction.get(best.actionKey) ?? -Infinity;

    for (const candidate of candidates.slice(1)) {
      const score = averagesByAction.get(candidate.actionKey) ?? -Infinity;
      if (score > bestScore) {
        best = candidate;
        bestScore = score;
      }
    }

    chosen = best;
  }

  return {
    config: {
      windowSize: chosen.windowSize,
      epochs: chosen.epochs,
      batchSize: chosen.batchSize,
      learningRate: chosen.learningRate,
      dropoutRate: chosen.dropoutRate,
    },
    actionKey: chosen.actionKey,
    algorithmVersion: POLICY_ALGORITHM_VERSION,
    explorationRate: epsilon,
    wasExploration: shouldExplore,
    mode,
    source: "policy",
    rationale: shouldExplore
      ? "Exploration step picked a candidate action."
      : "Selected best-known action by historical average reward.",
    context,
  };
}

export function recordPredictionDecision(params: {
  runId: number | null;
  targetDrawNumber: number;
  basedOnDrawNumber: number;
  basedOnDrawDate: string;
  predictedNumbers: number[];
  source: "policy" | "fallback";
  policyMode: PolicyMode;
  algorithmVersion: string;
  context: PolicyContext;
  action: {
    actionKey: string;
  } & PolicyTunableConfig;
}): number {
  const existing = db
    .select({ id: nnPredictionDecisions.id })
    .from(nnPredictionDecisions)
    .where(
      and(
        params.runId === null
          ? sql`${nnPredictionDecisions.runId} IS NULL`
          : eq(nnPredictionDecisions.runId, params.runId),
        eq(nnPredictionDecisions.targetDrawNumber, params.targetDrawNumber),
        eq(nnPredictionDecisions.algorithmVersion, params.algorithmVersion),
        eq(nnPredictionDecisions.predictedNumbers, JSON.stringify(params.predictedNumbers)),
      ),
    )
    .get();
  if (existing) return existing.id;

  const inserted = db
    .insert(nnPredictionDecisions)
    .values({
      runId: params.runId,
      targetDrawNumber: params.targetDrawNumber,
      basedOnDrawNumber: params.basedOnDrawNumber,
      basedOnDrawDate: params.basedOnDrawDate,
      predictedNumbers: JSON.stringify(params.predictedNumbers),
      source: params.source,
      policyMode: params.policyMode,
      algorithmVersion: params.algorithmVersion,
      contextJson: JSON.stringify(params.context),
      actionJson: JSON.stringify(params.action),
    })
    .returning({ id: nnPredictionDecisions.id })
    .get();

  if (!inserted) {
    throw new Error("Failed to write prediction decision.");
  }

  return inserted.id;
}

export function attachDecisionToRun(params: {
  decisionId: number;
  runId: number;
  wasOverridden: boolean;
  overrideConfig?: PolicyTunableConfig;
}): void {
  db.update(nnPredictionDecisions)
    .set({
      usedForRunId: params.runId,
      wasOverridden: params.wasOverridden,
      overrideConfigJson: params.overrideConfig
        ? JSON.stringify(params.overrideConfig)
        : null,
    })
    .where(eq(nnPredictionDecisions.id, params.decisionId))
    .run();
}

function numbersForDraw(drawId: number): number[] {
  return db
    .select({ value: lotteryDrawBalls.value })
    .from(lotteryDrawBalls)
    .where(
      and(
        eq(lotteryDrawBalls.drawId, drawId),
        eq(lotteryDrawBalls.isBonus, false),
      ),
    )
    .orderBy(lotteryDrawBalls.position)
    .all()
    .map((row) => row.value)
    .sort((a, b) => a - b);
}

function parseNumbersSafe(raw: string): number[] {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      return parsed
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value))
        .map((value) => Math.floor(value))
        .sort((a, b) => a - b);
    }
  } catch {
    return [];
  }

  return [];
}

function countMatches(predicted: number[], actual: number[]): number {
  const actualSet = new Set(actual);
  return predicted.filter((n) => actualSet.has(n)).length;
}

export function resolvePendingFeedback(limit = 200): {
  resolvedCount: number;
} {
  const pending = db
    .select({
      id: nnPredictionDecisions.id,
      targetDrawId: nnPredictionDecisions.targetDrawId,
      targetDrawNumber: nnPredictionDecisions.targetDrawNumber,
      predictedNumbers: nnPredictionDecisions.predictedNumbers,
      contextJson: nnPredictionDecisions.contextJson,
      actionJson: nnPredictionDecisions.actionJson,
      policyMode: nnPredictionDecisions.policyMode,
      algorithmVersion: nnPredictionDecisions.algorithmVersion,
    })
    .from(nnPredictionDecisions)
    .where(eq(nnPredictionDecisions.resolved, false))
    .orderBy(desc(nnPredictionDecisions.id))
    .limit(limit)
    .all();

  let resolvedCount = 0;

  for (const decision of pending) {
    const matchedDraw = decision.targetDrawId
      ? db
          .select({ id: lotteryDraws.id })
          .from(lotteryDraws)
          .where(eq(lotteryDraws.id, decision.targetDrawId))
          .get()
      : db
          .select({ id: lotteryDraws.id })
          .from(lotteryDraws)
          .where(eq(lotteryDraws.drawNumber, decision.targetDrawNumber))
          .orderBy(desc(lotteryDraws.id))
          .get();

    if (!matchedDraw) {
      continue;
    }

    const existingOutcome = db
      .select({ id: nnFeedbackOutcomes.id })
      .from(nnFeedbackOutcomes)
      .where(eq(nnFeedbackOutcomes.decisionId, decision.id))
      .get();

    if (existingOutcome) {
      db.update(nnPredictionDecisions)
        .set({
          resolved: true,
          targetDrawId: matchedDraw.id,
          resolvedAt: new Date(),
        })
        .where(eq(nnPredictionDecisions.id, decision.id))
        .run();
      continue;
    }

    const predicted = parseNumbersSafe(decision.predictedNumbers);
    const actual = numbersForDraw(matchedDraw.id);
    if (predicted.length === 0 || actual.length === 0) {
      continue;
    }

    const matchCount = countMatches(predicted, actual);
    const topKHit = matchCount > 0;
    const reward = computeRewardV1(matchCount, topKHit);

    const insertedOutcome = db
      .insert(nnFeedbackOutcomes)
      .values({
        decisionId: decision.id,
        drawId: matchedDraw.id,
        actualNumbers: JSON.stringify(actual),
        matchCount,
        topKHit,
        rewardValue: String(reward),
        rewardVersion: REWARD_VERSION,
      })
      .returning({ id: nnFeedbackOutcomes.id })
      .get();

    db.update(nnPredictionDecisions)
      .set({
        resolved: true,
        targetDrawId: matchedDraw.id,
        resolvedAt: new Date(),
      })
      .where(eq(nnPredictionDecisions.id, decision.id))
      .run();

    const baseline = summarizeRecentRewards(25).average;
    const context = (() => {
      try {
        return JSON.parse(decision.contextJson) as PolicyContext;
      } catch {
        return null;
      }
    })();

    const action = (() => {
      try {
        return JSON.parse(decision.actionJson) as Record<string, unknown>;
      } catch {
        return { actionKey: "unknown" };
      }
    })();

    db.insert(nnPolicyUpdates)
      .values({
        decisionId: decision.id,
        feedbackOutcomeId: insertedOutcome?.id ?? null,
        algorithmVersion: decision.algorithmVersion,
        policyMode: decision.policyMode,
        inputFeaturesJson: JSON.stringify(context ?? {}),
        chosenActionJson: JSON.stringify(action),
        rewardValue: String(reward),
        baselineReward: String(Number(baseline.toFixed(4))),
        advantage: String(Number((reward - baseline).toFixed(4))),
        explorationRate: String(DEFAULT_EXPLORATION_RATE),
      })
      .run();

    resolvedCount += 1;
  }

  return { resolvedCount };
}

export function getFeedbackLoopStatus(): FeedbackLoopStatus {
  const mode = getPolicyMode();

  const pendingRow = (() => {
    try {
      return db
        .select({ count: nnPredictionDecisions.id })
        .from(nnPredictionDecisions)
        .where(eq(nnPredictionDecisions.resolved, false))
        .all();
    } catch {
      return [];
    }
  })();

  const resolvedRow = (() => {
    try {
      return db
        .select({ count: nnFeedbackOutcomes.id })
        .from(nnFeedbackOutcomes)
        .all();
    } catch {
      return [];
    }
  })();

  const summary = summarizeRecentRewards();

  return {
    mode,
    pendingDecisions: pendingRow.length,
    resolvedOutcomes: resolvedRow.length,
    recentAverageReward: Number(summary.average.toFixed(4)),
    recentRewardCount: summary.count,
    lastReward: summary.lastReward,
    lastResolvedAt: summary.lastResolvedAt,
    explorationRate: DEFAULT_EXPLORATION_RATE,
  };
}

export function getPolicyUpdateHistory(limit = 100): PolicyUpdateHistoryItem[] {
  let rows: Array<{
    id: number;
    decisionId: number | null;
    algorithmVersion: string;
    policyMode: string;
    rewardValue: string;
    baselineReward: string | null;
    advantage: string | null;
    explorationRate: string;
    createdAt: Date;
  }> = [];

  try {
    rows = db
      .select({
        id: nnPolicyUpdates.id,
        decisionId: nnPolicyUpdates.decisionId,
        algorithmVersion: nnPolicyUpdates.algorithmVersion,
        policyMode: nnPolicyUpdates.policyMode,
        rewardValue: nnPolicyUpdates.rewardValue,
        baselineReward: nnPolicyUpdates.baselineReward,
        advantage: nnPolicyUpdates.advantage,
        explorationRate: nnPolicyUpdates.explorationRate,
        createdAt: nnPolicyUpdates.createdAt,
      })
      .from(nnPolicyUpdates)
      .orderBy(desc(nnPolicyUpdates.id))
      .limit(limit)
      .all();
  } catch {
    return [];
  }

  return rows.map((row) => ({
    id: row.id,
    decisionId: row.decisionId,
    algorithmVersion: row.algorithmVersion,
    policyMode:
      row.policyMode === "off" ||
      row.policyMode === "active" ||
      row.policyMode === "shadow"
        ? row.policyMode
        : "shadow",
    rewardValue: Number(row.rewardValue),
    baselineReward:
      row.baselineReward !== null && Number.isFinite(Number(row.baselineReward))
        ? Number(row.baselineReward)
        : null,
    advantage:
      row.advantage !== null && Number.isFinite(Number(row.advantage))
        ? Number(row.advantage)
        : null,
    explorationRate: Number(row.explorationRate),
    createdAt: row.createdAt.toISOString(),
  }));
}

export function getRewardTrend(limit = 120): RewardTrendPoint[] {
  const history = getPolicyUpdateHistory(limit).slice().reverse();

  const rollingWindow = 10;
  const points: RewardTrendPoint[] = [];

  for (let i = 0; i < history.length; i += 1) {
    const reward = history[i].rewardValue;
    const start = Math.max(0, i - (rollingWindow - 1));
    const windowRewards = history
      .slice(start, i + 1)
      .map((entry) => entry.rewardValue);
    const rollingAverage =
      windowRewards.reduce((acc, value) => acc + value, 0) /
      windowRewards.length;

    points.push({
      index: i + 1,
      reward,
      rollingAverage: Number(rollingAverage.toFixed(4)),
      createdAt: history[i].createdAt,
    });
  }

  return points;
}

function parseActualNumberCount(actualNumbersJson: string): number {
  try {
    const parsed = JSON.parse(actualNumbersJson) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.filter((value) => Number.isFinite(Number(value))).length;
    }
  } catch {
    return 0;
  }

  return 0;
}

export function getSuccessHistory(limit = 120): SuccessHistoryItem[] {
  let rows: Array<{
    id: number;
    decisionId: number;
    drawId: number | null;
    matchCount: number;
    actualNumbers: string;
    createdAt: Date;
  }> = [];

  try {
    rows = db
      .select({
        id: nnFeedbackOutcomes.id,
        decisionId: nnFeedbackOutcomes.decisionId,
        drawId: nnFeedbackOutcomes.drawId,
        matchCount: nnFeedbackOutcomes.matchCount,
        actualNumbers: nnFeedbackOutcomes.actualNumbers,
        createdAt: nnFeedbackOutcomes.createdAt,
      })
      .from(nnFeedbackOutcomes)
      .orderBy(desc(nnFeedbackOutcomes.id))
      .limit(limit)
      .all();
  } catch {
    return [];
  }

  return rows.map((row) => {
    const actualCount = parseActualNumberCount(row.actualNumbers);
    const successPercent =
      actualCount > 0 ? (row.matchCount / actualCount) * 100 : 0;

    return {
      id: row.id,
      decisionId: row.decisionId,
      drawId: row.drawId,
      matchCount: row.matchCount,
      actualCount,
      successPercent: Number(successPercent.toFixed(2)),
      createdAt: row.createdAt.toISOString(),
    };
  });
}

export function getSuccessTrend(limit = 120): SuccessTrendPoint[] {
  const history = getSuccessHistory(limit).slice().reverse();

  const rollingWindow = 10;
  const points: SuccessTrendPoint[] = [];

  for (let i = 0; i < history.length; i += 1) {
    const successPercent = history[i].successPercent;
    const start = Math.max(0, i - (rollingWindow - 1));
    const windowValues = history
      .slice(start, i + 1)
      .map((entry) => entry.successPercent);
    const rollingAverage =
      windowValues.reduce((acc, value) => acc + value, 0) / windowValues.length;

    points.push({
      index: i + 1,
      successPercent,
      rollingAverage: Number(rollingAverage.toFixed(2)),
      createdAt: history[i].createdAt,
    });
  }

  return points;
}

export function getLatestOpenDecision(): {
  id: number;
  actionJson: string;
} | null {
  const row = db
    .select({
      id: nnPredictionDecisions.id,
      actionJson: nnPredictionDecisions.actionJson,
    })
    .from(nnPredictionDecisions)
    .where(
      and(
        eq(nnPredictionDecisions.resolved, false),
        isNotNull(nnPredictionDecisions.actionJson),
      ),
    )
    .orderBy(desc(nnPredictionDecisions.id))
    .get();

  return row ?? null;
}

export function resolvePresetFromDecisionAction(
  actionJson: string,
): TrainingPreset | null {
  try {
    const action = JSON.parse(actionJson) as { preset?: string };
    if (
      action.preset === "small" ||
      action.preset === "medium" ||
      action.preset === "large" ||
      action.preset === "xlarge"
    ) {
      return action.preset;
    }
  } catch {
    return null;
  }

  return null;
}

export function defaultPolicyConfig(): PolicyTunableConfig {
  return {
    windowSize: DEFAULT_TRAINING_CONFIG.windowSize,
    epochs: DEFAULT_TRAINING_CONFIG.epochs,
    batchSize: DEFAULT_TRAINING_CONFIG.batchSize,
    learningRate: DEFAULT_TRAINING_CONFIG.learningRate,
    dropoutRate: DEFAULT_TRAINING_CONFIG.dropoutRate,
  };
}
