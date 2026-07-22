import { readFile } from "node:fs/promises";

import * as tf from "@tensorflow/tfjs";
import { and, desc, eq, isNotNull } from "drizzle-orm";

import { db } from "$lib/server/db";
import { nnTrainingRuns } from "$lib/server/db/schema";

import { buildDrawEventHistory, loadParsedDraws } from "./data";
import type { ReliabilityGateDiagnostics } from "./backtest";
import {
  buildInputVector,
  buildMultiHotInputVector,
  buildRecentFrequencyScores,
  LOTTERY_TICKET_SIZE,
  pickTopKNumbers,
} from "./features";
import { buildHeuristicScores } from "./heuristic";
import {
  blendEnsembleScores,
  buildSeededRandomScores,
  DEFAULT_ENSEMBLE_WEIGHTS,
  getEffectiveEnsembleWeights,
  type EnsembleScores,
  type EnsembleWeights,
} from "./ensemble";
import {
  choosePolicyDecision,
  recordPredictionDecision,
  type PolicyTunableConfig,
} from "./policy";

type SerializedModelArtifact = {
  runId: number;
  hiddenLayers: number[];
  inputSize: number;
  outputSize: number;
  outputNumbers: number[];
  weights: Array<{
    name: string;
    shape: number[];
    values: number[];
  }>;
  members?: Array<{
    seed: number;
    weights: Array<{ name: string; shape: number[]; values: number[] }>;
  }>;
  averagingMethod?: "equal_probability_mean";
  ensembleWeights?: EnsembleWeights;
  ensembleVersion?: string;
  ensembleReliability?: ReliabilityGateDiagnostics;
  inputEncoding?: "sorted_scalar_v1" | "windowed_multi_hot_v1";
  lossVersion?: string;
  trainingSeed?: number;
};

export type NextDrawPrediction = {
  runId: number;
  decisionId: number;
  nextDrawNumber: number;
  basedOnDrawNumber: number;
  basedOnDrawDate: string;
  predictedNumbers: number[];
  neuralPredictedNumbers: number[];
  frequencyPredictedNumbers: number[];
  heuristicPredictedNumbers: number[];
  randomPredictedNumbers: number[];
  ensembleWeights: EnsembleWeights;
  ensembleReliability: SerializedModelArtifact["ensembleReliability"] | null;
  neuralSeeds: number[];
  neuralAveragingMethod: "equal_probability_mean";
  generatedAt: string;
};

function buildInferenceModel(params: {
  hiddenLayers: number[];
  inputSize: number;
  outputSize: number;
  activation: "relu" | "tanh" | "sigmoid";
  weights: Array<{ shape: number[]; values: number[] }>;
}): tf.Sequential {
  const model = tf.sequential();

  params.hiddenLayers.forEach((units, idx) => {
    model.add(
      tf.layers.dense({
        units,
        activation: params.activation,
        inputShape: idx === 0 ? [params.inputSize] : undefined,
      }),
    );
  });

  model.add(
    tf.layers.dense({
      units: params.outputSize,
      activation: "sigmoid",
    }),
  );

  const tensors = params.weights.map((weight) => {
    if (weight.shape.some((dim) => dim <= 0)) {
      throw new Error("Artifact contains an invalid weight shape.");
    }

    return tf.tensor(weight.values, weight.shape);
  });

  model.setWeights(tensors);
  tensors.forEach((tensor) => tensor.dispose());

  return model;
}

function buildInputVectorFromLatest(
  parsedDraws: ReturnType<typeof loadParsedDraws>,
  windowSize: number,
  maxNumber: number,
  outputNumbers: number[],
  inputEncoding: "sorted_scalar_v1" | "windowed_multi_hot_v1",
): {
  input: number[];
  latestWindow: ReturnType<typeof loadParsedDraws>;
  latestDrawNumber: number;
  latestDrawDate: string;
  numbersPerDraw: number;
} {
  const ordered = buildDrawEventHistory(parsedDraws);

  if (ordered.length < windowSize) {
    throw new Error("Not enough draws available for next-draw prediction.");
  }

  const latestWindow = ordered.slice(-windowSize);
  const latestDraw = ordered[ordered.length - 1];

  return {
    input:
      inputEncoding === "windowed_multi_hot_v1"
        ? buildMultiHotInputVector(latestWindow, outputNumbers)
        : buildInputVector(latestWindow, maxNumber),
    latestWindow,
    latestDrawNumber: latestDraw.drawNumber,
    latestDrawDate: latestDraw.drawDate.toISOString().slice(0, 10),
    numbersPerDraw: LOTTERY_TICKET_SIZE,
  };
}

export async function getNextDrawPrediction(
  preferredRunId?: number,
): Promise<NextDrawPrediction | null> {
  const run = preferredRunId
    ? db
        .select({
          id: nnTrainingRuns.id,
          status: nnTrainingRuns.status,
          modelArtifactPath: nnTrainingRuns.modelArtifactPath,
          hyperparamsJson: nnTrainingRuns.hyperparamsJson,
          isValid: nnTrainingRuns.isValid,
        })
        .from(nnTrainingRuns)
        .where(eq(nnTrainingRuns.id, preferredRunId))
        .get()
    : db
        .select({
          id: nnTrainingRuns.id,
          status: nnTrainingRuns.status,
          modelArtifactPath: nnTrainingRuns.modelArtifactPath,
          hyperparamsJson: nnTrainingRuns.hyperparamsJson,
          isValid: nnTrainingRuns.isValid,
        })
        .from(nnTrainingRuns)
        .where(
          and(
            eq(nnTrainingRuns.status, "completed"),
            eq(nnTrainingRuns.isValid, true),
            isNotNull(nnTrainingRuns.modelArtifactPath),
          ),
        )
        .orderBy(desc(nnTrainingRuns.id))
        .get();

  if (!run || !run.modelArtifactPath || run.status !== "completed" || !run.isValid) {
    return null;
  }

  const rawArtifact = await readFile(run.modelArtifactPath, "utf8");
  const artifact = JSON.parse(rawArtifact) as SerializedModelArtifact;
  const hyperparams = JSON.parse(run.hyperparamsJson) as {
    windowSize?: number;
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
    dropoutRate?: number;
    activation?: "relu" | "tanh" | "sigmoid";
  };

  const windowSize = hyperparams.windowSize ?? 10;
  const activation = hyperparams.activation ?? "relu";
  const baseConfig: PolicyTunableConfig = {
    windowSize,
    epochs: hyperparams.epochs ?? 80,
    batchSize: hyperparams.batchSize ?? 32,
    learningRate: hyperparams.learningRate ?? 0.001,
    dropoutRate: hyperparams.dropoutRate ?? 0.25,
  };

  const draws = loadParsedDraws();

  if (draws.length === 0) {
    return null;
  }

  const maxNumber = Math.max(...artifact.outputNumbers);
  if (!Number.isFinite(maxNumber) || maxNumber <= 0) {
    throw new Error("Invalid output number vocabulary in artifact.");
  }

  const inferenceInput = buildInputVectorFromLatest(
    draws,
    windowSize,
    maxNumber,
    artifact.outputNumbers,
    artifact.inputEncoding ?? "sorted_scalar_v1",
  );

  const models: tf.Sequential[] = [];
  let inputTensor: tf.Tensor2D | null = null;
  let predictionTensor: tf.Tensor | null = null;

  try {
    const artifactMembers = artifact.members?.length
      ? artifact.members
      : [{ seed: artifact.trainingSeed ?? 42, weights: artifact.weights }];
    for (const member of artifactMembers) {
      models.push(buildInferenceModel({ hiddenLayers: artifact.hiddenLayers,
        inputSize: artifact.inputSize, outputSize: artifact.outputSize, activation,
        weights: member.weights.map((weight) => ({ shape: weight.shape, values: weight.values })) }));
    }

    inputTensor = tf.tensor2d([inferenceInput.input], [1, artifact.inputSize]);
    const scoreTotals = new Array(artifact.outputSize).fill(0);
    for (const memberModel of models) {
      predictionTensor = memberModel.predict(inputTensor) as tf.Tensor;
      const memberScores = await predictionTensor.data();
      memberScores.forEach((value, index) => scoreTotals[index] += value);
      predictionTensor.dispose(); predictionTensor = null;
    }
    const scores = scoreTotals.map((value) => value / models.length);
    const recentFrequencyScores = buildRecentFrequencyScores(
      inferenceInput.latestWindow,
      artifact.outputNumbers,
    );
    const heuristicScores = buildHeuristicScores(
      inferenceInput.latestWindow,
      artifact.outputNumbers,
    );
    const randomScores = buildSeededRandomScores(
      artifact.outputNumbers.length,
      inferenceInput.latestDrawNumber + 1,
    );
    const componentScores: EnsembleScores = {
      neural: scores,
      frequency: recentFrequencyScores,
      heuristic: heuristicScores,
      random: randomScores,
    };
    const learnedEnsembleWeights = artifact.ensembleWeights ?? DEFAULT_ENSEMBLE_WEIGHTS;
    const effectiveEnsembleWeights = getEffectiveEnsembleWeights(
      componentScores,
      learnedEnsembleWeights,
    );
    const blendedScores = blendEnsembleScores(componentScores, learnedEnsembleWeights);
    const neuralPredictedNumbers = pickTopKNumbers(
      scores,
      artifact.outputNumbers,
      inferenceInput.numbersPerDraw,
    );
    const frequencyPredictedNumbers = pickTopKNumbers(
      recentFrequencyScores,
      artifact.outputNumbers,
      inferenceInput.numbersPerDraw,
    );
    const heuristicPredictedNumbers = pickTopKNumbers(
      heuristicScores,
      artifact.outputNumbers,
      inferenceInput.numbersPerDraw,
    );
    const randomPredictedNumbers = pickTopKNumbers(
      randomScores,
      artifact.outputNumbers,
      inferenceInput.numbersPerDraw,
    );
    const predictedNumbers = pickTopKNumbers(
      blendedScores,
      artifact.outputNumbers,
      inferenceInput.numbersPerDraw,
    );

    const policyDecision = choosePolicyDecision({
      baseConfig,
      signal: "keep",
      context: {
        latestRunId: run.id,
        signal: "keep",
        latestTrainLoss: null,
        latestValLoss: null,
        latestHoldoutScore: null,
      },
    });

    const decisionId = recordPredictionDecision({
      runId: run.id,
      targetDrawNumber: inferenceInput.latestDrawNumber + 1,
      basedOnDrawNumber: inferenceInput.latestDrawNumber,
      basedOnDrawDate: inferenceInput.latestDrawDate,
      predictedNumbers,
      source: policyDecision.source,
      policyMode: policyDecision.mode,
      algorithmVersion: policyDecision.algorithmVersion,
      context: policyDecision.context,
      action: {
        ...policyDecision.config,
        actionKey: policyDecision.actionKey,
      },
    });

    return {
      runId: run.id,
      decisionId,
      nextDrawNumber: inferenceInput.latestDrawNumber + 1,
      basedOnDrawNumber: inferenceInput.latestDrawNumber,
      basedOnDrawDate: inferenceInput.latestDrawDate,
      predictedNumbers,
      neuralPredictedNumbers,
      frequencyPredictedNumbers,
      heuristicPredictedNumbers,
      randomPredictedNumbers,
      ensembleWeights: effectiveEnsembleWeights,
      ensembleReliability: artifact.ensembleReliability ?? null,
      neuralSeeds: artifactMembers.map((member) => member.seed),
      neuralAveragingMethod: "equal_probability_mean",
      generatedAt: new Date().toISOString(),
    };
  } finally {
    predictionTensor?.dispose();
    inputTensor?.dispose();
    models.forEach((model) => model.dispose());
    await tf.nextFrame();
  }
}
