import * as tf from "@tensorflow/tfjs";
import { eq } from "drizzle-orm";

import { db } from "$lib/server/db";
import { nnTrainingRuns } from "$lib/server/db/schema";

import {
  DEFAULT_TRAINING_CONFIG,
  type TrainingConfig,
  validateTrainingConfig,
} from "./config";
import {
  createWindowedSamples,
  assertDrawIntegrity,
  loadParsedDraws,
  type ParsedDraw,
  splitByHoldoutWeeks,
} from "./data";
import {
  buildMultiHotInputVector,
  buildNumberVocabulary,
  buildRecentFrequencyScores,
  buildTargetMultiHot,
  canonicalizeNumbers,
  countMatches,
  pickFrequencyBaselineNumbers,
  pickTopKNumbers,
} from "./features";
import { buildHeuristicScores, pickHeuristicNumbers } from "./heuristic";
import {
  blendEnsembleScores,
  buildSeededRandomScores,
  DEFAULT_ENSEMBLE_WEIGHTS,
  type EnsembleMethod,
  updateEnsembleWeights,
  calculateCalibrationReward,
} from "./ensemble";
import {
  addPredictionEvaluations,
  addTestResult,
  markRunCompleted,
  markRunFailed,
  markRunRunning,
  updateRunEpochProgress,
} from "./runs";
import { cleanupOldArtifacts, saveModelArtifact } from "./artifacts";

type WindowedVectorSample = {
  input: number[];
  inputWindow: ParsedDraw[];
  targetMultiHot: number[];
  targetNumbers: number[];
  targetDrawId: number;
  targetDrawDate: string;
};

const runningRunIds = new Set<number>();

function disposeWeights(weights: readonly tf.Tensor[] | null): void {
  weights?.forEach((weight) => weight.dispose());
}

function getRunConfig(runId: number): TrainingConfig {
  const loaded = db
    .select({
      hyperparamsJson: nnTrainingRuns.hyperparamsJson,
    })
    .from(nnTrainingRuns)
    .where(eq(nnTrainingRuns.id, runId))
    .get();

  if (!loaded) {
    throw new Error(`Training run ${runId} not found.`);
  }

  const parsed = {
    ...DEFAULT_TRAINING_CONFIG,
    ...(JSON.parse(loaded.hyperparamsJson) as Partial<TrainingConfig>),
  };
  const errors = validateTrainingConfig(parsed);
  if (errors.length > 0) {
    throw new Error(`Run ${runId} has invalid config: ${errors.join(" ")}`);
  }

  return parsed;
}

function prepareSamples(config: TrainingConfig) {
  const allDraws = loadParsedDraws();
  const eraStartMs = config.currentEraStartDate
    ? Date.parse(config.currentEraStartDate)
    : null;
  const draws =
    eraStartMs === null
      ? allDraws
      : allDraws.filter((draw) => draw.drawDate.getTime() >= eraStartMs);
  assertDrawIntegrity(draws);
  const split = splitByHoldoutWeeks(draws, config.holdoutWeeks);
  const cutoffMs = split.cutoffDate.getTime();

  const vocabulary = buildNumberVocabulary(draws);
  const indexByNumber = new Map(vocabulary.map((n, idx) => [n, idx]));

  const allWindowed = createWindowedSamples(draws, config.windowSize);

  const trainingSamples = allWindowed
    .filter((sample) => {
      const targetInTrain = sample.targetDraw.drawDate.getTime() <= cutoffMs;
      const inputsInTrain = sample.inputWindow.every(
        (draw) => draw.drawDate.getTime() <= cutoffMs,
      );
      return targetInTrain && inputsInTrain;
    })
    .map(
      (sample): WindowedVectorSample => ({
        input: buildMultiHotInputVector(sample.inputWindow, vocabulary),
        inputWindow: sample.inputWindow,
        targetMultiHot: buildTargetMultiHot(
          sample.targetDraw.winningNumbers,
          indexByNumber,
        ),
        targetNumbers: canonicalizeNumbers(sample.targetDraw.winningNumbers),
        targetDrawId: sample.targetDraw.id,
        targetDrawDate: sample.targetDraw.drawDate.toISOString().slice(0, 10),
      }),
    );

  const holdoutSamples = allWindowed
    .filter((sample) => sample.targetDraw.drawDate.getTime() > cutoffMs)
    .map(
      (sample): WindowedVectorSample => ({
        input: buildMultiHotInputVector(sample.inputWindow, vocabulary),
        inputWindow: sample.inputWindow,
        targetMultiHot: buildTargetMultiHot(
          sample.targetDraw.winningNumbers,
          indexByNumber,
        ),
        targetNumbers: canonicalizeNumbers(sample.targetDraw.winningNumbers),
        targetDrawId: sample.targetDraw.id,
        targetDrawDate: sample.targetDraw.drawDate.toISOString().slice(0, 10),
      }),
    );

  if (trainingSamples.length === 0) {
    throw new Error("No training samples available after holdout split.");
  }

  if (holdoutSamples.length === 0) {
    throw new Error(
      "No holdout samples available; increase holdout weeks or decrease window size.",
    );
  }

  return {
    trainingSamples,
    holdoutSamples,
    outputNumbers: vocabulary,
  };
}

function buildModel(
  config: TrainingConfig,
  inputSize: number,
  outputSize: number,
): tf.Sequential {
  const model = tf.sequential();

  config.hiddenLayers.forEach((units, idx) => {
    model.add(
      tf.layers.dense({
        units,
        activation: config.activation,
        inputShape: idx === 0 ? [inputSize] : undefined,
        kernelInitializer: tf.initializers.glorotUniform({ seed: config.trainingSeed + idx }),
      }),
    );

    if (config.dropoutRate > 0) {
      model.add(tf.layers.dropout({ rate: config.dropoutRate, seed: config.trainingSeed + idx }));
    }
  });

  model.add(
    tf.layers.dense({
      units: outputSize,
      activation: "sigmoid",
      kernelInitializer: tf.initializers.glorotUniform({ seed: config.trainingSeed + 1000 }),
    }),
  );

  model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: (yTrue, yPred) =>
      tf.tidy(() => {
        const one = tf.scalar(1);
        const clipped = tf.clipByValue(yPred, 1e-7, 1 - 1e-7);
        const positives = yTrue.mul(tf.log(clipped)).mul(config.positiveClassWeight);
        const negatives = one.sub(yTrue).mul(tf.log(one.sub(clipped)));
        return tf.neg(tf.mean(positives.add(negatives)));
      }),
  });

  return model;
}

async function runTrainingJob(runId: number): Promise<void> {
  const config = getRunConfig(runId);
  const { trainingSamples, holdoutSamples, outputNumbers } =
    prepareSamples(config);

  const calibrationCount = Math.max(1, Math.floor(trainingSamples.length * 0.2));
  const baseTrainingSamples = trainingSamples.slice(0, -calibrationCount);
  const calibrationSamples = trainingSamples.slice(-calibrationCount);
  const inputSize = trainingSamples[0].input.length;
  const outputSize = trainingSamples[0].targetMultiHot.length;

  let xTrain: tf.Tensor2D | null = null;
  let yTrain: tf.Tensor2D | null = null;
  let xCalibration: tf.Tensor2D | null = null;
  let yCalibration: tf.Tensor2D | null = null;
  let model: tf.Sequential | null = null;
  let bestWeights: tf.Tensor[] | null = null;

  try {
    xTrain = tf.tensor2d(
      baseTrainingSamples.map((s) => s.input),
      [baseTrainingSamples.length, inputSize],
    );
    yTrain = tf.tensor2d(
      baseTrainingSamples.map((s) => s.targetMultiHot),
      [baseTrainingSamples.length, outputSize],
    );
    xCalibration = tf.tensor2d(calibrationSamples.map((s) => s.input), [calibrationSamples.length, inputSize]);
    yCalibration = tf.tensor2d(calibrationSamples.map((s) => s.targetMultiHot), [calibrationSamples.length, outputSize]);

    model = buildModel(config, inputSize, outputSize);
    const samplesPerEpoch = baseTrainingSamples.length;

    markRunRunning(runId);

    const startedAt = Date.now();
    let bestValLoss = Number.POSITIVE_INFINITY;
    let bestEpoch: number | null = null;

    await model.fit(xTrain, yTrain, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      shuffle: false,
      validationData: [xCalibration, yCalibration],
      callbacks: [{
        onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
          const epochNumber = epoch + 1;
          if (typeof logs?.val_loss === "number" && logs.val_loss < bestValLoss) {
            bestValLoss = logs.val_loss;
            bestEpoch = epochNumber;
            disposeWeights(bestWeights);
            bestWeights = model?.getWeights().map((weight) => weight.clone()) ?? null;
          }
          const elapsedMs = Date.now() - startedAt;
          updateRunEpochProgress({
            runId,
            epoch: epochNumber,
            totalEpochs: config.epochs,
            trainLoss: typeof logs?.loss === "number" ? logs.loss : undefined,
            valLoss:
              typeof logs?.val_loss === "number" ? logs.val_loss : undefined,
            elapsedMs,
            samplesProcessed: epochNumber * samplesPerEpoch,
          });
        },
      }, tf.callbacks.earlyStopping({
        monitor: "val_loss",
        patience: config.earlyStoppingPatience,
        minDelta: config.earlyStoppingMinDelta,
      })],
    });

    if (bestWeights) {
      model.setWeights(bestWeights);
    }

    let totalMatches = 0;
    let ensembleWeights = { ...DEFAULT_ENSEMBLE_WEIGHTS };

    // Calibrate ensemble weights on a chronological partition that the neural
    // model did not train on. Random remains a zero-weight evaluation control.
    for (const sample of calibrationSamples) {
      const inputTensor = tf.tensor2d([sample.input], [1, inputSize]);
      const prediction = model.predict(inputTensor) as tf.Tensor;
      const scores = Array.from(await prediction.data());
      const frequencyScores = buildRecentFrequencyScores(sample.inputWindow, outputNumbers);
      const heuristicScores = buildHeuristicScores(sample.inputWindow, outputNumbers);
      const rewards: Record<EnsembleMethod, number> = {
        neural: calculateCalibrationReward(scores, outputNumbers, sample.targetNumbers),
        frequency: calculateCalibrationReward(frequencyScores, outputNumbers, sample.targetNumbers),
        heuristic: calculateCalibrationReward(heuristicScores, outputNumbers, sample.targetNumbers),
        random: 0,
      };
      ensembleWeights = updateEnsembleWeights(ensembleWeights, rewards);
      inputTensor.dispose();
      prediction.dispose();
    }

    const methodTotals = { neural: 0, frequency: 0, random: 0, heuristic: 0, ensemble: 0 };

    for (const sample of holdoutSamples) {
      const inputTensor = tf.tensor2d([sample.input], [1, inputSize]);
      const prediction = model.predict(inputTensor) as tf.Tensor;
      const scores = Array.from(await prediction.data());
      const recentFrequencyScores = buildRecentFrequencyScores(
        sample.inputWindow,
        outputNumbers,
      );
      const predictedNumbers = pickTopKNumbers(
        scores,
        outputNumbers,
        sample.targetNumbers.length,
      );
      const matchCount = countMatches(predictedNumbers, sample.targetNumbers);
      const topKHit = matchCount > 0;
      const frequencyPredictedNumbers = pickFrequencyBaselineNumbers(
        sample.inputWindow,
        outputNumbers,
        sample.targetNumbers.length,
      );
      const frequencyMatchCount = countMatches(
        frequencyPredictedNumbers,
        sample.targetNumbers,
      );
      const randomScores = buildSeededRandomScores(
        outputNumbers.length,
        sample.targetDrawId,
      );
      const randomPredictedNumbers = pickTopKNumbers(
        randomScores,
        outputNumbers,
        sample.targetNumbers.length,
      );
      const randomMatchCount = countMatches(
        randomPredictedNumbers,
        sample.targetNumbers,
      );
      const heuristicPredictedNumbers = pickHeuristicNumbers(
        sample.inputWindow,
        outputNumbers,
        sample.targetNumbers.length,
      );
      const heuristicMatchCount = countMatches(
        heuristicPredictedNumbers,
        sample.targetNumbers,
      );
      const heuristicScores = buildHeuristicScores(sample.inputWindow, outputNumbers);
      const ensembleScores = blendEnsembleScores(
        {
          neural: scores,
          frequency: recentFrequencyScores,
          heuristic: heuristicScores,
          random: randomScores,
        },
        ensembleWeights,
      );
      const ensemblePredictedNumbers = pickTopKNumbers(
        ensembleScores,
        outputNumbers,
        sample.targetNumbers.length,
      );
      const ensembleMatchCount = countMatches(
        ensemblePredictedNumbers,
        sample.targetNumbers,
      );

      addTestResult({
        runId,
        drawId: sample.targetDrawId,
        drawDate: sample.targetDrawDate,
        predictedNumbers,
        actualNumbers: sample.targetNumbers,
        frequencyPredictedNumbers,
        frequencyMatchCount,
        randomPredictedNumbers,
        randomMatchCount,
        matchCount,
        topKHit,
      });
      addPredictionEvaluations([
        {
          runId,
          drawId: sample.targetDrawId,
          drawDate: sample.targetDrawDate,
          method: "neural",
          predictedNumbers,
          actualNumbers: sample.targetNumbers,
          matchCount,
          topKHit,
        },
        {
          runId,
          drawId: sample.targetDrawId,
          drawDate: sample.targetDrawDate,
          method: "frequency",
          predictedNumbers: frequencyPredictedNumbers,
          actualNumbers: sample.targetNumbers,
          matchCount: frequencyMatchCount,
          topKHit: frequencyMatchCount > 0,
        },
        {
          runId,
          drawId: sample.targetDrawId,
          drawDate: sample.targetDrawDate,
          method: "random",
          predictedNumbers: randomPredictedNumbers,
          actualNumbers: sample.targetNumbers,
          matchCount: randomMatchCount,
          topKHit: randomMatchCount > 0,
        },
        {
          runId,
          drawId: sample.targetDrawId,
          drawDate: sample.targetDrawDate,
          method: "heuristic",
          predictedNumbers: heuristicPredictedNumbers,
          actualNumbers: sample.targetNumbers,
          matchCount: heuristicMatchCount,
          topKHit: heuristicMatchCount > 0,
        },
        {
          runId,
          drawId: sample.targetDrawId,
          drawDate: sample.targetDrawDate,
          method: "ensemble",
          predictedNumbers: ensemblePredictedNumbers,
          actualNumbers: sample.targetNumbers,
          matchCount: ensembleMatchCount,
          topKHit: ensembleMatchCount > 0,
        },
      ]);

      totalMatches += matchCount;
      methodTotals.neural += matchCount;
      methodTotals.frequency += frequencyMatchCount;
      methodTotals.random += randomMatchCount;
      methodTotals.heuristic += heuristicMatchCount;
      methodTotals.ensemble += ensembleMatchCount;

      inputTensor.dispose();
      prediction.dispose();
    }

    const averageMatches = totalMatches / holdoutSamples.length;

    const serializedWeights = model.weights.map((weight) => {
      const values = Array.from(weight.read().dataSync());
      return {
        name: weight.name,
        shape: weight.shape.map((dim) => dim ?? -1),
        values,
      };
    });

    await saveModelArtifact({
      runId,
      createdAt: new Date().toISOString(),
      modelFamily: config.modelFamily,
      hiddenLayers: config.hiddenLayers,
      inputSize,
      outputSize,
      outputNumbers,
      weights: serializedWeights,
      ensembleWeights,
      ensembleVersion: "calibrated-experts-v2",
      inputEncoding: "windowed_multi_hot_v1",
      lossVersion: "weighted_bce_v1",
      trainingSeed: config.trainingSeed,
    });

    await cleanupOldArtifacts(20);

    db.update(nnTrainingRuns)
      .set({
        holdoutScore: (methodTotals.ensemble / holdoutSamples.length).toFixed(4),
        neuralHoldoutScore: averageMatches.toFixed(4),
        ensembleHoldoutScore: (methodTotals.ensemble / holdoutSamples.length).toFixed(4),
        frequencyHoldoutScore: (methodTotals.frequency / holdoutSamples.length).toFixed(4),
        heuristicHoldoutScore: (methodTotals.heuristic / holdoutSamples.length).toFixed(4),
        randomHoldoutScore: (methodTotals.random / holdoutSamples.length).toFixed(4),
        inputEncoding: "windowed_multi_hot_v1",
        lossVersion: "weighted_bce_v1",
        trainingSeed: config.trainingSeed,
        bestEpoch,
      })
      .where(eq(nnTrainingRuns.id, runId))
      .run();

    markRunCompleted(runId);
  } finally {
    xTrain?.dispose();
    yTrain?.dispose();
    xCalibration?.dispose();
    yCalibration?.dispose();
    model?.dispose();
    disposeWeights(bestWeights);
    await tf.nextFrame();
  }
}

export function startTrainingRun(runId: number): boolean {
  if (runningRunIds.has(runId)) {
    return false;
  }

  runningRunIds.add(runId);

  void runTrainingJob(runId)
    .catch((error) => {
      const message =
        error instanceof Error ? error.message : "Unknown training error.";
      markRunFailed(runId, message);
    })
    .finally(() => {
      runningRunIds.delete(runId);
    });

  return true;
}
