import { and, asc, desc, eq, inArray, isNotNull, sql } from "drizzle-orm";

import { db } from "$lib/server/db";
import {
  lotteryDraws,
  lotteryEntries,
  nnEpochMetrics,
  nnPredictionEvaluations,
  nnTestResults,
  nnTrainingRuns,
} from "$lib/server/db/schema";

import {
  DEFAULT_TRAINING_CONFIG,
  type TrainingConfig,
  validateTrainingConfig,
} from "./config";
import {
  choosePolicyDecision,
  defaultPolicyConfig,
  getFeedbackLoopStatus,
  type FeedbackLoopStatus,
  type PolicySignal,
} from "./policy";
import {
  assertNoLeakage,
  assertDrawIntegrity,
  createWindowedSamples,
  loadParsedDraws,
  parseCsvDataRowToNormalized,
  parseLotteryRow,
  splitByHoldoutWeeks,
} from "./data";

export type RunSummary = {
  id: number;
  status: string;
  currentEpoch: number;
  totalEpochs: number;
  windowSize: number;
  samplesProcessed: number;
  samplesTotal: number | null;
  roundsRemaining: number;
  holdoutWeeks: number;
  hiddenLayers: number[];
  paramCount: number | null;
  holdoutScore: number | null;
  finalTrainLoss: number | null;
  finalValLoss: number | null;
  createdAt: Date;
  startedAt: Date | null;
  endedAt: Date | null;
};

export type RunEpochMetric = {
  epoch: number;
  trainLoss: number | null;
  valLoss: number | null;
  elapsedMs: number | null;
};

export type RunTestResult = {
  drawId: number | null;
  drawNumber: number | null;
  drawDate: string | null;
  gameName: string | null;
  drawSequence: number | null;
  drawRound: number | null;
  machine: string | null;
  predictedNumbers: number[];
  actualNumbers: number[];
  frequencyPredictedNumbers: number[] | null;
  frequencyMatchCount: number | null;
  randomPredictedNumbers: number[] | null;
  randomMatchCount: number | null;
  heuristicPredictedNumbers: number[] | null;
  heuristicMatchCount: number | null;
  ensemblePredictedNumbers: number[] | null;
  ensembleMatchCount: number | null;
  matchCount: number;
  topKHit: boolean;
};

export type RunDetail = {
  run: RunSummary;
  epochMetrics: RunEpochMetric[];
  testResults: RunTestResult[];
  methodSummaries: PredictionEvaluationSummary[];
};

export type PredictionMethod =
  | "neural"
  | "frequency"
  | "random"
  | "heuristic"
  | "ensemble";

export type PredictionEvaluationSummary = {
  method: PredictionMethod;
  sampleCount: number;
  averageMatches: number;
  topKHitRate: number;
  zeroMatchRate: number;
  twoPlusMatchRate: number;
  threePlusMatchRate: number;
  standardError: number;
  confidenceLow95: number;
  confidenceHigh95: number;
};

export type RunComparisonSummary = {
  runId: number;
  status: string;
  holdoutWeeks: number;
  windowSize: number;
  createdAt: Date;
  methodSummaries: PredictionEvaluationSummary[];
  modelAverage: number | null;
  frequencyAverage: number | null;
  randomAverage: number | null;
  heuristicAverage: number | null;
  ensembleAverage: number | null;
  modelVsFrequency: number | null;
  modelVsRandom: number | null;
  modelVsHeuristic: number | null;
  winner: PredictionMethod | null;
};

export type DatasetStats = {
  drawCount: number;
  distinctNumberCount: number;
  maxObservedNumber: number;
  conflictingDrawIdentities: number;
  randomExpectedMatches: number;
};

export type PredictionRunCandidate = {
  id: number;
  createdAt: Date;
  holdoutScore: number | null;
  paramCount: number | null;
  hiddenLayers: number[];
  windowSize: number;
};

export type TrainingRecommendation = {
  kind: "scale_up" | "regularize" | "keep";
  message: string;
  suggestedPreset?: "small" | "medium" | "large" | "xlarge";
  recommendedConfig?: {
    windowSize: number;
    epochs: number;
    batchSize: number;
    learningRate: number;
    dropoutRate: number;
  };
  source?: "policy" | "fallback";
  policyMode?: "off" | "shadow" | "active";
  algorithmVersion?: string;
  explorationRate?: number;
  wasExploration?: boolean;
  policyRationale?: string;
  feedbackStatus?: FeedbackLoopStatus;
};

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

function parseLatestRunConfig(
  value: string,
): Pick<
  TrainingConfig,
  "windowSize" | "epochs" | "batchSize" | "learningRate" | "dropoutRate"
> {
  try {
    const parsed = JSON.parse(value) as Partial<TrainingConfig>;
    return {
      windowSize: clampInt(
        parsed.windowSize ?? DEFAULT_TRAINING_CONFIG.windowSize,
        1,
        52,
      ),
      epochs: clampInt(parsed.epochs ?? DEFAULT_TRAINING_CONFIG.epochs, 1, 500),
      batchSize: clampInt(
        parsed.batchSize ?? DEFAULT_TRAINING_CONFIG.batchSize,
        1,
        1024,
      ),
      learningRate: clampFloat(
        parsed.learningRate ?? DEFAULT_TRAINING_CONFIG.learningRate,
        0.000001,
        0.1,
      ),
      dropoutRate: clampFloat(
        parsed.dropoutRate ?? DEFAULT_TRAINING_CONFIG.dropoutRate,
        0,
        0.99,
      ),
    };
  } catch {
    return {
      windowSize: DEFAULT_TRAINING_CONFIG.windowSize,
      epochs: DEFAULT_TRAINING_CONFIG.epochs,
      batchSize: DEFAULT_TRAINING_CONFIG.batchSize,
      learningRate: DEFAULT_TRAINING_CONFIG.learningRate,
      dropoutRate: DEFAULT_TRAINING_CONFIG.dropoutRate,
    };
  }
}

function estimateParamCount(
  inputSize: number,
  hiddenLayers: number[],
  outputSize: number,
): number {
  let params = 0;
  let previousWidth = inputSize;

  for (const width of hiddenLayers) {
    params += previousWidth * width + width;
    previousWidth = width;
  }

  params += previousWidth * outputSize + outputSize;
  return params;
}

function parseNullableNumber(value: string | null): number | null {
  if (value === null) {
    return null;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function buildTrainingConfig(
  input: Partial<TrainingConfig>,
): TrainingConfig {
  return {
    ...DEFAULT_TRAINING_CONFIG,
    ...input,
    hiddenLayers: input.hiddenLayers ?? DEFAULT_TRAINING_CONFIG.hiddenLayers,
  };
}

export function createTrainingRun(
  configInput: Partial<TrainingConfig> = {},
): RunSummary {
  const config = buildTrainingConfig(configInput);
  const validationErrors = validateTrainingConfig(config);

  if (validationErrors.length > 0) {
    throw new Error(validationErrors.join(" "));
  }

  const allParsedDraws = loadParsedDraws();
  const eraStartMs = config.currentEraStartDate
    ? Date.parse(config.currentEraStartDate)
    : null;
  const parsedDraws =
    eraStartMs === null
      ? allParsedDraws
      : allParsedDraws.filter((draw) => draw.drawDate.getTime() >= eraStartMs);
  assertDrawIntegrity(parsedDraws);
  const split = splitByHoldoutWeeks(parsedDraws, config.holdoutWeeks);
  const allWindowedSamples = createWindowedSamples(
    parsedDraws,
    config.windowSize,
  );
  const cutoffMs = split.cutoffDate.getTime();
  const trainSamples = allWindowedSamples.filter((sample) => {
    const targetInTrain = sample.targetDraw.drawDate.getTime() <= cutoffMs;
    const inputsInTrain = sample.inputWindow.every(
      (draw) => draw.drawDate.getTime() <= cutoffMs,
    );
    return targetInTrain && inputsInTrain;
  });
  const testSamples = allWindowedSamples.filter(
    (sample) => sample.targetDraw.drawDate.getTime() > cutoffMs,
  );

  const holdoutDrawIds = new Set(split.test.map((draw) => draw.id));
  assertNoLeakage(trainSamples, holdoutDrawIds);

  if (trainSamples.length === 0) {
    throw new Error("Training samples are empty. Decrease windowSize.");
  }

  const allNumbers = new Set<number>();
  for (const draw of parsedDraws) {
    for (const n of draw.winningNumbers) {
      allNumbers.add(n);
    }
  }

  const outputSize = allNumbers.size;
  const inputSize = config.windowSize * outputSize;
  const paramCount = estimateParamCount(
    inputSize,
    config.hiddenLayers,
    outputSize,
  );

  const inserted = db
    .insert(nnTrainingRuns)
    .values({
      status: "queued",
      holdoutWeeks: config.holdoutWeeks,
      windowSize: config.windowSize,
      totalEpochs: config.epochs,
      currentEpoch: 0,
      trainSamples: trainSamples.length,
      testSamples: testSamples.length,
      samplesTotal: trainSamples.length * config.epochs,
      samplesProcessed: 0,
      modelFamily: config.modelFamily,
      inputEncoding: "windowed_multi_hot_v1",
      lossVersion: "weighted_bce_v1",
      trainingSeed: config.trainingSeed,
      hiddenLayersJson: JSON.stringify(config.hiddenLayers),
      hyperparamsJson: JSON.stringify(config),
      paramCount,
    })
    .returning({
      id: nnTrainingRuns.id,
      status: nnTrainingRuns.status,
      currentEpoch: nnTrainingRuns.currentEpoch,
      totalEpochs: nnTrainingRuns.totalEpochs,
      windowSize: nnTrainingRuns.windowSize,
      samplesProcessed: nnTrainingRuns.samplesProcessed,
      samplesTotal: nnTrainingRuns.samplesTotal,
      holdoutWeeks: nnTrainingRuns.holdoutWeeks,
      hiddenLayersJson: nnTrainingRuns.hiddenLayersJson,
      createdAt: nnTrainingRuns.createdAt,
      startedAt: nnTrainingRuns.startedAt,
      endedAt: nnTrainingRuns.endedAt,
    })
    .get();

  if (!inserted) {
    throw new Error("Failed to create training run.");
  }

  return {
    id: inserted.id,
    status: inserted.status,
    currentEpoch: inserted.currentEpoch,
    totalEpochs: inserted.totalEpochs,
    windowSize: inserted.windowSize,
    samplesProcessed: inserted.samplesProcessed,
    samplesTotal: inserted.samplesTotal,
    roundsRemaining: Math.max(0, inserted.totalEpochs - inserted.currentEpoch),
    holdoutWeeks: inserted.holdoutWeeks,
    hiddenLayers: JSON.parse(inserted.hiddenLayersJson) as number[],
    paramCount,
    holdoutScore: null,
    finalTrainLoss: null,
    finalValLoss: null,
    createdAt: inserted.createdAt,
    startedAt: inserted.startedAt,
    endedAt: inserted.endedAt,
  };
}

export function markRunRunning(runId: number): void {
  db.update(nnTrainingRuns)
    .set({ status: "running", startedAt: new Date() })
    .where(eq(nnTrainingRuns.id, runId))
    .run();
}

export function markRunFailed(runId: number, errorMessage: string): void {
  db.update(nnTrainingRuns)
    .set({ status: "failed", errorMessage, endedAt: new Date() })
    .where(eq(nnTrainingRuns.id, runId))
    .run();
}

export function markRunCompleted(runId: number): void {
  db.update(nnTrainingRuns)
    .set({ status: "completed", endedAt: new Date() })
    .where(eq(nnTrainingRuns.id, runId))
    .run();
}

export function updateRunEpochProgress(params: {
  runId: number;
  epoch: number;
  totalEpochs: number;
  trainLoss?: number;
  valLoss?: number;
  elapsedMs?: number;
  samplesProcessed?: number;
}): void {
  const {
    runId,
    epoch,
    totalEpochs,
    trainLoss,
    valLoss,
    elapsedMs,
    samplesProcessed,
  } = params;

  db.transaction((tx) => {
    tx.insert(nnEpochMetrics)
      .values({
        runId,
        epoch,
        trainLoss: trainLoss === undefined ? null : String(trainLoss),
        valLoss: valLoss === undefined ? null : String(valLoss),
        elapsedMs,
      })
      .run();

    tx.update(nnTrainingRuns)
      .set({
        currentEpoch: epoch,
        samplesProcessed,
        finalTrainLoss: trainLoss === undefined ? null : String(trainLoss),
        finalValLoss: valLoss === undefined ? null : String(valLoss),
      })
      .where(and(eq(nnTrainingRuns.id, runId), sql`${epoch} <= ${totalEpochs}`))
      .run();
  });
}

export function addTestResult(params: {
  runId: number;
  drawId?: number;
  drawDate?: string;
  predictedNumbers: number[];
  actualNumbers: number[];
  frequencyPredictedNumbers?: number[];
  frequencyMatchCount?: number;
  randomPredictedNumbers?: number[];
  randomMatchCount?: number;
  matchCount: number;
  topKHit: boolean;
}): void {
  db.insert(nnTestResults)
    .values({
      runId: params.runId,
      drawId: params.drawId,
      drawDate: params.drawDate,
      predictedNumbers: JSON.stringify(params.predictedNumbers),
      actualNumbers: JSON.stringify(params.actualNumbers),
      frequencyPredictedNumbers: params.frequencyPredictedNumbers
        ? JSON.stringify(params.frequencyPredictedNumbers)
        : null,
      frequencyMatchCount: params.frequencyMatchCount ?? null,
      randomPredictedNumbers: params.randomPredictedNumbers
        ? JSON.stringify(params.randomPredictedNumbers)
        : null,
      randomMatchCount: params.randomMatchCount ?? null,
      matchCount: params.matchCount,
      topKHit: params.topKHit,
    })
    .run();
}

export function addPredictionEvaluations(
  evaluations: Array<{
    runId: number;
    drawId?: number;
    drawDate?: string;
    method: PredictionMethod;
    predictedNumbers: number[];
    actualNumbers: number[];
    matchCount: number;
    topKHit: boolean;
  }>,
): void {
  if (evaluations.length === 0) {
    return;
  }

  db.insert(nnPredictionEvaluations)
    .values(
      evaluations.map((evaluation) => ({
        runId: evaluation.runId,
        drawId: evaluation.drawId,
        drawDate: evaluation.drawDate,
        method: evaluation.method,
        predictedNumbers: JSON.stringify(evaluation.predictedNumbers),
        actualNumbers: JSON.stringify(evaluation.actualNumbers),
        matchCount: evaluation.matchCount,
        topKHit: evaluation.topKHit,
      })),
    )
    .run();
}

export function getLatestRuns(limit = 20): RunSummary[] {
  const runs = db
    .select({
      id: nnTrainingRuns.id,
      status: nnTrainingRuns.status,
      currentEpoch: nnTrainingRuns.currentEpoch,
      totalEpochs: nnTrainingRuns.totalEpochs,
      windowSize: nnTrainingRuns.windowSize,
      samplesProcessed: nnTrainingRuns.samplesProcessed,
      samplesTotal: nnTrainingRuns.samplesTotal,
      holdoutWeeks: nnTrainingRuns.holdoutWeeks,
      hiddenLayersJson: nnTrainingRuns.hiddenLayersJson,
      paramCount: nnTrainingRuns.paramCount,
      holdoutScore: nnTrainingRuns.holdoutScore,
      finalTrainLoss: nnTrainingRuns.finalTrainLoss,
      finalValLoss: nnTrainingRuns.finalValLoss,
      createdAt: nnTrainingRuns.createdAt,
      startedAt: nnTrainingRuns.startedAt,
      endedAt: nnTrainingRuns.endedAt,
    })
    .from(nnTrainingRuns)
    .orderBy(desc(nnTrainingRuns.id))
    .limit(limit)
    .all();

  return runs.map((run) => ({
    id: run.id,
    status: run.status,
    currentEpoch: run.currentEpoch,
    totalEpochs: run.totalEpochs,
    windowSize: run.windowSize,
    samplesProcessed: run.samplesProcessed,
    samplesTotal: run.samplesTotal,
    roundsRemaining: Math.max(0, run.totalEpochs - run.currentEpoch),
    holdoutWeeks: run.holdoutWeeks,
    hiddenLayers: JSON.parse(run.hiddenLayersJson) as number[],
    paramCount: run.paramCount,
    holdoutScore: parseNullableNumber(run.holdoutScore),
    finalTrainLoss: parseNullableNumber(run.finalTrainLoss),
    finalValLoss: parseNullableNumber(run.finalValLoss),
    createdAt: run.createdAt,
    startedAt: run.startedAt,
    endedAt: run.endedAt,
  }));
}

export function getPredictionRunCandidates(
  limit = 30,
): PredictionRunCandidate[] {
  const rows = db
    .select({
      id: nnTrainingRuns.id,
      createdAt: nnTrainingRuns.createdAt,
      holdoutScore: nnTrainingRuns.holdoutScore,
      paramCount: nnTrainingRuns.paramCount,
      hiddenLayersJson: nnTrainingRuns.hiddenLayersJson,
      windowSize: nnTrainingRuns.windowSize,
    })
    .from(nnTrainingRuns)
    .where(
      and(
        eq(nnTrainingRuns.status, "completed"),
        isNotNull(nnTrainingRuns.modelArtifactPath),
      ),
    )
    .orderBy(desc(nnTrainingRuns.id))
    .limit(limit)
    .all();

  return rows.map((row) => ({
    id: row.id,
    createdAt: row.createdAt,
    holdoutScore: parseNullableNumber(row.holdoutScore),
    paramCount: row.paramCount,
    hiddenLayers: JSON.parse(row.hiddenLayersJson) as number[],
    windowSize: row.windowSize,
  }));
}

export function getRunProgress(runId: number): RunSummary | null {
  const run = db
    .select({
      id: nnTrainingRuns.id,
      status: nnTrainingRuns.status,
      currentEpoch: nnTrainingRuns.currentEpoch,
      totalEpochs: nnTrainingRuns.totalEpochs,
      windowSize: nnTrainingRuns.windowSize,
      samplesProcessed: nnTrainingRuns.samplesProcessed,
      samplesTotal: nnTrainingRuns.samplesTotal,
      holdoutWeeks: nnTrainingRuns.holdoutWeeks,
      hiddenLayersJson: nnTrainingRuns.hiddenLayersJson,
      paramCount: nnTrainingRuns.paramCount,
      holdoutScore: nnTrainingRuns.holdoutScore,
      finalTrainLoss: nnTrainingRuns.finalTrainLoss,
      finalValLoss: nnTrainingRuns.finalValLoss,
      createdAt: nnTrainingRuns.createdAt,
      startedAt: nnTrainingRuns.startedAt,
      endedAt: nnTrainingRuns.endedAt,
    })
    .from(nnTrainingRuns)
    .where(eq(nnTrainingRuns.id, runId))
    .get();

  if (!run) {
    return null;
  }

  return {
    id: run.id,
    status: run.status,
    currentEpoch: run.currentEpoch,
    totalEpochs: run.totalEpochs,
    windowSize: run.windowSize,
    samplesProcessed: run.samplesProcessed,
    samplesTotal: run.samplesTotal,
    roundsRemaining: Math.max(0, run.totalEpochs - run.currentEpoch),
    holdoutWeeks: run.holdoutWeeks,
    hiddenLayers: JSON.parse(run.hiddenLayersJson) as number[],
    paramCount: run.paramCount,
    holdoutScore: parseNullableNumber(run.holdoutScore),
    finalTrainLoss: parseNullableNumber(run.finalTrainLoss),
    finalValLoss: parseNullableNumber(run.finalValLoss),
    createdAt: run.createdAt,
    startedAt: run.startedAt,
    endedAt: run.endedAt,
  };
}

export function getRunDetail(
  runId: number,
  limits?: { epochs?: number; tests?: number },
): RunDetail | null {
  const run = getRunProgress(runId);
  if (!run) {
    return null;
  }

  const epochRows = db
    .select({
      epoch: nnEpochMetrics.epoch,
      trainLoss: nnEpochMetrics.trainLoss,
      valLoss: nnEpochMetrics.valLoss,
      elapsedMs: nnEpochMetrics.elapsedMs,
    })
    .from(nnEpochMetrics)
    .where(eq(nnEpochMetrics.runId, runId))
    .orderBy(asc(nnEpochMetrics.epoch))
    .limit(limits?.epochs ?? 500)
    .all();

  const testRows = db
    .select({
      drawId: nnTestResults.drawId,
      drawDate: nnTestResults.drawDate,
      predictedNumbers: nnTestResults.predictedNumbers,
      actualNumbers: nnTestResults.actualNumbers,
      frequencyPredictedNumbers: nnTestResults.frequencyPredictedNumbers,
      frequencyMatchCount: nnTestResults.frequencyMatchCount,
      randomPredictedNumbers: nnTestResults.randomPredictedNumbers,
      randomMatchCount: nnTestResults.randomMatchCount,
      matchCount: nnTestResults.matchCount,
      topKHit: nnTestResults.topKHit,
    })
    .from(nnTestResults)
    .where(eq(nnTestResults.runId, runId))
    .orderBy(desc(nnTestResults.id))
    .limit(limits?.tests ?? 100)
    .all();

  const evaluationRows = db
    .select({
      drawId: nnPredictionEvaluations.drawId,
      method: nnPredictionEvaluations.method,
      predictedNumbers: nnPredictionEvaluations.predictedNumbers,
      matchCount: nnPredictionEvaluations.matchCount,
    })
    .from(nnPredictionEvaluations)
    .where(eq(nnPredictionEvaluations.runId, runId))
    .all();

  const evaluationsByDrawAndMethod = new Map<
    string,
    { predictedNumbers: number[]; matchCount: number }
  >();
  for (const evaluation of evaluationRows) {
    if (typeof evaluation.drawId !== "number") {
      continue;
    }

    evaluationsByDrawAndMethod.set(
      `${evaluation.drawId}:${evaluation.method}`,
      {
        predictedNumbers: JSON.parse(evaluation.predictedNumbers) as number[],
        matchCount: evaluation.matchCount,
      },
    );
  }

  const drawIds = testRows
    .map((row) => row.drawId)
    .filter((drawId): drawId is number => typeof drawId === "number");

  const normalizedDrawRows = drawIds.length
    ? db
        .select({
          id: lotteryDraws.id,
          drawNumber: lotteryDraws.drawNumber,
          drawDate: lotteryDraws.drawDate,
          gameName: lotteryDraws.gameName,
          drawSequence: lotteryDraws.drawSequence,
          drawRound: lotteryDraws.drawRound,
          machine: lotteryDraws.machine,
        })
        .from(lotteryDraws)
        .where(inArray(lotteryDraws.id, drawIds))
        .all()
    : [];

  const legacyRows = drawIds.length
    ? db
        .select({
          id: lotteryEntries.id,
          numbers: lotteryEntries.numbers,
        })
        .from(lotteryEntries)
        .where(inArray(lotteryEntries.id, drawIds))
        .all()
    : [];

  const drawMetaById = new Map<
    number,
    {
      drawNumber: number;
      drawDate: string;
      gameName: string | null;
      drawSequence: number | null;
      drawRound: number;
      machine: string | null;
    }
  >();

  for (const row of normalizedDrawRows) {
    drawMetaById.set(row.id, {
      drawNumber: row.drawNumber,
      drawDate: row.drawDate.toISOString().slice(0, 10),
      gameName: row.gameName,
      drawSequence: row.drawSequence,
      drawRound: row.drawRound,
      machine: row.machine,
    });
  }

  for (const row of legacyRows) {
    if (drawMetaById.has(row.id)) {
      continue;
    }

    const parsed = parseLotteryRow({
      id: row.id,
      numbers: row.numbers,
      createdAt: null,
    });
    const normalized = parseCsvDataRowToNormalized(row.numbers, row.id);

    drawMetaById.set(row.id, {
      drawNumber: parsed.drawNumber,
      drawDate: parsed.drawDate.toISOString().slice(0, 10),
      gameName: normalized.draw.gameName,
      drawSequence: normalized.draw.drawSequence,
      drawRound: normalized.draw.drawRound,
      machine: normalized.draw.machine,
    });
  }

  return {
    run,
    epochMetrics: epochRows.map((row) => ({
      epoch: row.epoch,
      trainLoss: parseNullableNumber(row.trainLoss),
      valLoss: parseNullableNumber(row.valLoss),
      elapsedMs: row.elapsedMs,
    })),
    testResults: testRows.map((row) => ({
      drawId: row.drawId,
      drawNumber: row.drawId
        ? (drawMetaById.get(row.drawId)?.drawNumber ?? null)
        : null,
      drawDate: row.drawId
        ? (drawMetaById.get(row.drawId)?.drawDate ?? row.drawDate)
        : row.drawDate,
      gameName: row.drawId
        ? (drawMetaById.get(row.drawId)?.gameName ?? null)
        : null,
      drawSequence: row.drawId
        ? (drawMetaById.get(row.drawId)?.drawSequence ?? null)
        : null,
      drawRound: row.drawId
        ? (drawMetaById.get(row.drawId)?.drawRound ?? null)
        : null,
      machine: row.drawId
        ? (drawMetaById.get(row.drawId)?.machine ?? null)
        : null,
      predictedNumbers: JSON.parse(row.predictedNumbers) as number[],
      actualNumbers: JSON.parse(row.actualNumbers) as number[],
      frequencyPredictedNumbers: row.frequencyPredictedNumbers
        ? (JSON.parse(row.frequencyPredictedNumbers) as number[])
        : null,
      frequencyMatchCount: row.frequencyMatchCount,
      randomPredictedNumbers: row.randomPredictedNumbers
        ? (JSON.parse(row.randomPredictedNumbers) as number[])
        : null,
      randomMatchCount: row.randomMatchCount,
      heuristicPredictedNumbers:
        row.drawId !== null
          ? (evaluationsByDrawAndMethod.get(`${row.drawId}:heuristic`)
              ?.predictedNumbers ?? null)
          : null,
      heuristicMatchCount:
        row.drawId !== null
          ? (evaluationsByDrawAndMethod.get(`${row.drawId}:heuristic`)
              ?.matchCount ?? null)
          : null,
      ensemblePredictedNumbers:
        row.drawId !== null
          ? (evaluationsByDrawAndMethod.get(`${row.drawId}:ensemble`)
              ?.predictedNumbers ?? null)
          : null,
      ensembleMatchCount:
        row.drawId !== null
          ? (evaluationsByDrawAndMethod.get(`${row.drawId}:ensemble`)
              ?.matchCount ?? null)
          : null,
      matchCount: row.matchCount,
      topKHit: row.topKHit,
    })),
    methodSummaries: getRunEvaluationSummaries(runId),
  };
}

function summarizeEvaluations(
  rows: Array<{ method: string; matchCount: number; topKHit: boolean }>,
): PredictionEvaluationSummary[] {
  const byMethod = new Map<
    PredictionMethod,
    Array<{ matchCount: number; topKHit: boolean }>
  >();

  for (const row of rows) {
    if (
      row.method !== "neural" &&
      row.method !== "frequency" &&
      row.method !== "random" &&
      row.method !== "heuristic" &&
      row.method !== "ensemble"
    ) {
      continue;
    }

    const values = byMethod.get(row.method) ?? [];
    values.push({
      matchCount: row.matchCount,
      topKHit: row.topKHit,
    });
    byMethod.set(row.method, values);
  }

  return [...byMethod.entries()]
    .map(([method, values]) => {
      const sampleCount = values.length;
      const averageMatches =
        values.reduce((acc, value) => acc + value.matchCount, 0) / sampleCount;
      const topKHitRate =
        values.filter((value) => value.topKHit).length / sampleCount;
      const zeroMatchRate =
        values.filter((value) => value.matchCount === 0).length / sampleCount;
      const twoPlusMatchRate =
        values.filter((value) => value.matchCount >= 2).length / sampleCount;
      const threePlusMatchRate =
        values.filter((value) => value.matchCount >= 3).length / sampleCount;
      const variance =
        values.reduce(
          (sum, value) => sum + (value.matchCount - averageMatches) ** 2,
          0,
        ) / Math.max(1, sampleCount - 1);
      const standardError = Math.sqrt(variance / sampleCount);

      return {
        method,
        sampleCount,
        averageMatches: Number(averageMatches.toFixed(4)),
        topKHitRate: Number(topKHitRate.toFixed(4)),
        zeroMatchRate: Number(zeroMatchRate.toFixed(4)),
        twoPlusMatchRate: Number(twoPlusMatchRate.toFixed(4)),
        threePlusMatchRate: Number(threePlusMatchRate.toFixed(4)),
        standardError: Number(standardError.toFixed(4)),
        confidenceLow95: Number(Math.max(0, averageMatches - 1.96 * standardError).toFixed(4)),
        confidenceHigh95: Number((averageMatches + 1.96 * standardError).toFixed(4)),
      };
    })
    .sort((a, b) => b.averageMatches - a.averageMatches);
}

function legacyEvaluationRowsForRun(
  runId: number,
): Array<{ method: PredictionMethod; matchCount: number; topKHit: boolean }> {
  const rows = db
    .select({
      matchCount: nnTestResults.matchCount,
      topKHit: nnTestResults.topKHit,
      frequencyMatchCount: nnTestResults.frequencyMatchCount,
      randomMatchCount: nnTestResults.randomMatchCount,
    })
    .from(nnTestResults)
    .where(eq(nnTestResults.runId, runId))
    .all();

  return rows.flatMap((row) => {
    const values: Array<{
      method: PredictionMethod;
      matchCount: number;
      topKHit: boolean;
    }> = [
      {
        method: "neural",
        matchCount: row.matchCount,
        topKHit: row.topKHit,
      },
    ];

    if (typeof row.frequencyMatchCount === "number") {
      values.push({
        method: "frequency",
        matchCount: row.frequencyMatchCount,
        topKHit: row.frequencyMatchCount > 0,
      });
    }

    if (typeof row.randomMatchCount === "number") {
      values.push({
        method: "random",
        matchCount: row.randomMatchCount,
        topKHit: row.randomMatchCount > 0,
      });
    }

    return values;
  });
}

export function getRunEvaluationSummaries(
  runId: number,
): PredictionEvaluationSummary[] {
  const rows = db
    .select({
      method: nnPredictionEvaluations.method,
      matchCount: nnPredictionEvaluations.matchCount,
      topKHit: nnPredictionEvaluations.topKHit,
    })
    .from(nnPredictionEvaluations)
    .where(eq(nnPredictionEvaluations.runId, runId))
    .all();

  if (rows.length > 0) {
    return summarizeEvaluations(rows);
  }

  return summarizeEvaluations(legacyEvaluationRowsForRun(runId));
}

export function getRunComparisonSummaries(
  limit = 20,
): RunComparisonSummary[] {
  const runs = db
    .select({
      id: nnTrainingRuns.id,
      status: nnTrainingRuns.status,
      holdoutWeeks: nnTrainingRuns.holdoutWeeks,
      windowSize: nnTrainingRuns.windowSize,
      createdAt: nnTrainingRuns.createdAt,
    })
    .from(nnTrainingRuns)
    .orderBy(desc(nnTrainingRuns.id))
    .limit(limit)
    .all();

  if (runs.length === 0) {
    return [];
  }

  const evaluations = db
    .select({
      runId: nnPredictionEvaluations.runId,
      method: nnPredictionEvaluations.method,
      matchCount: nnPredictionEvaluations.matchCount,
      topKHit: nnPredictionEvaluations.topKHit,
    })
    .from(nnPredictionEvaluations)
    .where(
      inArray(
        nnPredictionEvaluations.runId,
        runs.map((run) => run.id),
      ),
    )
    .all();

  const runsWithEvaluations = new Set(
    evaluations.map((evaluation) => evaluation.runId),
  );

  return runs.map((run) => {
    const methodSummaries = runsWithEvaluations.has(run.id)
      ? summarizeEvaluations(
          evaluations.filter((evaluation) => evaluation.runId === run.id),
        )
      : summarizeEvaluations(legacyEvaluationRowsForRun(run.id));
    const averageByMethod = new Map(
      methodSummaries.map((summary) => [
        summary.method,
        summary.averageMatches,
      ]),
    );
    const modelAverage = averageByMethod.get("neural") ?? null;
    const frequencyAverage = averageByMethod.get("frequency") ?? null;
    const randomAverage = averageByMethod.get("random") ?? null;
    const heuristicAverage = averageByMethod.get("heuristic") ?? null;
    const ensembleAverage = averageByMethod.get("ensemble") ?? null;
    const winner = methodSummaries[0]?.method ?? null;

    return {
      runId: run.id,
      status: run.status,
      holdoutWeeks: run.holdoutWeeks,
      windowSize: run.windowSize,
      createdAt: run.createdAt,
      methodSummaries,
      modelAverage,
      frequencyAverage,
      randomAverage,
      heuristicAverage,
      ensembleAverage,
      modelVsFrequency:
        modelAverage !== null && frequencyAverage !== null
          ? Number((modelAverage - frequencyAverage).toFixed(4))
          : null,
      modelVsRandom:
        modelAverage !== null && randomAverage !== null
          ? Number((modelAverage - randomAverage).toFixed(4))
          : null,
      modelVsHeuristic:
        modelAverage !== null && heuristicAverage !== null
          ? Number((modelAverage - heuristicAverage).toFixed(4))
          : null,
      winner,
    };
  });
}

export function getDatasetStats(): DatasetStats {
  const parsedDraws = loadParsedDraws();
  const distinct = new Set<number>();
  let maxObservedNumber = 0;
  const identityCounts = new Map<string, number>();

  for (const draw of parsedDraws) {
    const key = `${draw.drawNumber}:${draw.drawRound}:${draw.drawDate.getTime()}`;
    identityCounts.set(key, (identityCounts.get(key) ?? 0) + 1);
    for (const n of draw.winningNumbers) {
      distinct.add(n);
      if (n > maxObservedNumber) {
        maxObservedNumber = n;
      }
    }
  }

  return {
    drawCount: parsedDraws.length,
    distinctNumberCount: distinct.size,
    maxObservedNumber,
    conflictingDrawIdentities: [...identityCounts.values()].filter((count) => count > 1).length,
    randomExpectedMatches: Number(((6 * 6) / Math.max(1, maxObservedNumber)).toFixed(4)),
  };
}

function nextPresetForHiddenLayers(
  hiddenLayers: number[],
): "small" | "medium" | "large" | "xlarge" | null {
  const signature = hiddenLayers.join(",");
  if (signature === "64,32") {
    return "medium";
  }
  if (signature === "128,64") {
    return "large";
  }
  if (signature === "256,128") {
    return "xlarge";
  }

  return null;
}

export function getTrainingRecommendation(): TrainingRecommendation {
  const recentCompleted = db
    .select({
      id: nnTrainingRuns.id,
      hiddenLayersJson: nnTrainingRuns.hiddenLayersJson,
      hyperparamsJson: nnTrainingRuns.hyperparamsJson,
      holdoutScore: nnTrainingRuns.holdoutScore,
      finalTrainLoss: nnTrainingRuns.finalTrainLoss,
      finalValLoss: nnTrainingRuns.finalValLoss,
    })
    .from(nnTrainingRuns)
    .where(and(eq(nnTrainingRuns.status, "completed"), eq(nnTrainingRuns.isValid, true)))
    .orderBy(desc(nnTrainingRuns.id))
    .limit(4)
    .all();

  const feedbackStatus = getFeedbackLoopStatus();

  const latest = recentCompleted[0] ?? null;
  const latestConfig = latest
    ? parseLatestRunConfig(latest.hyperparamsJson)
    : defaultPolicyConfig();
  const latestScore = latest ? parseNullableNumber(latest.holdoutScore) : null;
  const latestTrainLoss = latest
    ? parseNullableNumber(latest.finalTrainLoss)
    : null;
  const latestValLoss = latest
    ? parseNullableNumber(latest.finalValLoss)
    : null;

  let kind: "scale_up" | "regularize" | "keep" = "keep";
  let message =
    "Current capacity looks reasonable. Keep architecture and tune other hyperparameters next.";
  let suggestedPreset: "small" | "medium" | "large" | "xlarge" | undefined;
  let signal: PolicySignal = "keep";

  if (recentCompleted.length < 2) {
    message =
      "Need at least two completed runs before making a tuning recommendation.";
  } else if (
    latestScore === null ||
    latestTrainLoss === null ||
    latestValLoss === null
  ) {
    message =
      "Latest run is missing metrics. Complete another run for recommendations.";
  } else {
    const previousScores = recentCompleted
      .slice(1)
      .map((row) => parseNullableNumber(row.holdoutScore))
      .filter((score): score is number => score !== null);

    const bestPreviousScore =
      previousScores.length > 0 ? Math.max(...previousScores) : null;

    const isOverfittingSignal = latestValLoss > latestTrainLoss * 1.2;
    if (isOverfittingSignal) {
      kind = "regularize";
      signal = "regularize";
      message =
        "Validation loss is significantly above training loss. Increase dropout or reduce neurons before scaling up.";
    } else {
      const noImprovement =
        bestPreviousScore !== null && latestScore <= bestPreviousScore;
      if (noImprovement && recentCompleted.length >= 4) {
        const latestHidden = JSON.parse(latest.hiddenLayersJson) as number[];
        const suggested = nextPresetForHiddenLayers(latestHidden);
        if (suggested) {
          kind = "scale_up";
          signal = "scale_up";
          suggestedPreset = suggested;
          message = `No holdout improvement in recent runs. Consider scaling up to ${suggested} preset.`;
        }
      }
    }
  }

  const policyDecision = choosePolicyDecision({
    baseConfig: latestConfig,
    signal,
    context: {
      latestRunId: latest?.id ?? null,
      signal,
      latestTrainLoss,
      latestValLoss,
      latestHoldoutScore: latestScore,
    },
  });

  return {
    kind,
    message,
    suggestedPreset,
    recommendedConfig: {
      windowSize: clampInt(policyDecision.config.windowSize, 1, 52),
      epochs: clampInt(policyDecision.config.epochs, 1, 500),
      batchSize: clampInt(policyDecision.config.batchSize, 1, 1024),
      learningRate: clampFloat(
        policyDecision.config.learningRate,
        0.000001,
        0.1,
      ),
      dropoutRate: clampFloat(policyDecision.config.dropoutRate, 0, 0.99),
    },
    source: policyDecision.source,
    policyMode: policyDecision.mode,
    algorithmVersion: policyDecision.algorithmVersion,
    explorationRate: policyDecision.explorationRate,
    wasExploration: policyDecision.wasExploration,
    policyRationale: policyDecision.rationale,
    feedbackStatus,
  };
}

export function getRunsForArtifactCleanup(
  keepCount = 20,
): Array<{ id: number; modelArtifactPath: string | null }> {
  const topByScore = db
    .select({
      id: nnTrainingRuns.id,
      modelArtifactPath: nnTrainingRuns.modelArtifactPath,
    })
    .from(nnTrainingRuns)
    .where(
      and(
        eq(nnTrainingRuns.status, "completed"),
        eq(nnTrainingRuns.isValid, true),
        sql`${nnTrainingRuns.modelArtifactPath} IS NOT NULL`,
      ),
    )
    .orderBy(
      desc(sql`CAST(${nnTrainingRuns.holdoutScore} AS REAL)`),
      desc(nnTrainingRuns.id),
    )
    .limit(keepCount)
    .all();

  const keepIds = new Set(topByScore.map((row) => row.id));

  const completedWithArtifacts = db
    .select({
      id: nnTrainingRuns.id,
      modelArtifactPath: nnTrainingRuns.modelArtifactPath,
    })
    .from(nnTrainingRuns)
    .where(
      and(
        eq(nnTrainingRuns.status, "completed"),
        eq(nnTrainingRuns.isValid, true),
        sql`${nnTrainingRuns.modelArtifactPath} IS NOT NULL`,
      ),
    )
    .orderBy(desc(nnTrainingRuns.id))
    .all();

  return completedWithArtifacts.filter((row) => !keepIds.has(row.id));
}

export function clearArtifactPathsForRuns(runIds: number[]): void {
  if (runIds.length === 0) {
    return;
  }

  db.update(nnTrainingRuns)
    .set({ modelArtifactPath: null, modelSha256: null })
    .where(inArray(nnTrainingRuns.id, runIds))
    .run();
}
