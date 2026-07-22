import * as tf from "@tensorflow/tfjs";

import { evaluateReliabilityGate, type ReliabilityGateDiagnostics } from "./backtest";
import { buildRecentFrequencyScores, countMatches, pickTopKNumbers } from "./features";
import { buildHeuristicScores } from "./heuristic";
import {
  blendEnsembleScores, buildSeededRandomScores, calculateCalibrationReward,
  collapseDuplicateExpertWeights, DEFAULT_ENSEMBLE_WEIGHTS,
  type EnsembleMethod, type EnsembleScores, type EnsembleWeights, updateEnsembleWeights,
} from "./ensemble";
import type { ParsedDraw } from "./data";

export type EnsembleCalibrationSample = {
  input: number[];
  inputWindow: ParsedDraw[];
  targetNumbers: number[];
  targetDrawNumber: number;
  targetDrawDate: string;
};

export async function calibrateEnsemble(params: {
  model: tf.Sequential;
  samples: EnsembleCalibrationSample[];
  outputNumbers: number[];
  inputSize: number;
  confidenceLevel: number;
  minimumAdvantage: number;
  minimumGroups: number;
  bootstrapIterations: number;
  seed: number;
}): Promise<{ weights: EnsembleWeights; calibratedWeights: EnsembleWeights; diagnostics: ReliabilityGateDiagnostics }> {
  const gateCount = Math.max(1, Math.floor(params.samples.length * 0.25));
  const learningSamples = params.samples.slice(0, -gateCount);
  const gateSamples = params.samples.slice(-gateCount);
  const learningRecords: Array<{ scores: EnsembleScores; targetNumbers: number[] }> = [];

  for (const sample of learningSamples) {
    const input = tf.tensor2d([sample.input], [1, params.inputSize]);
    const prediction = params.model.predict(input) as tf.Tensor;
    const neural = Array.from(await prediction.data());
    learningRecords.push({
      scores: {
        neural,
        frequency: buildRecentFrequencyScores(sample.inputWindow, params.outputNumbers),
        heuristic: buildHeuristicScores(sample.inputWindow, params.outputNumbers),
        random: buildSeededRandomScores(params.outputNumbers.length, sample.targetDrawNumber),
      },
      targetNumbers: sample.targetNumbers,
    });
    input.dispose(); prediction.dispose();
  }

  let weights = collapseDuplicateExpertWeights(
    learningRecords.map((record) => record.scores), DEFAULT_ENSEMBLE_WEIGHTS,
  );
  for (const record of learningRecords) {
    const rewards: Record<EnsembleMethod, number> = {
      neural: calculateCalibrationReward(record.scores.neural, params.outputNumbers, record.targetNumbers),
      frequency: calculateCalibrationReward(record.scores.frequency, params.outputNumbers, record.targetNumbers),
      heuristic: calculateCalibrationReward(record.scores.heuristic, params.outputNumbers, record.targetNumbers),
      random: 0,
    };
    weights = updateEnsembleWeights(weights, rewards);
  }

  const observations = [];
  for (const sample of gateSamples) {
    const input = tf.tensor2d([sample.input], [1, params.inputSize]);
    const prediction = params.model.predict(input) as tf.Tensor;
    const neural = Array.from(await prediction.data());
    const componentScores: EnsembleScores = {
      neural,
      frequency: buildRecentFrequencyScores(sample.inputWindow, params.outputNumbers),
      heuristic: buildHeuristicScores(sample.inputWindow, params.outputNumbers),
      random: buildSeededRandomScores(params.outputNumbers.length, sample.targetDrawNumber),
    };
    observations.push({
      groupKey: sample.targetDrawDate,
      neuralMatches: countMatches(pickTopKNumbers(neural, params.outputNumbers, sample.targetNumbers.length), sample.targetNumbers),
      ensembleMatches: countMatches(pickTopKNumbers(blendEnsembleScores(componentScores, weights), params.outputNumbers, sample.targetNumbers.length), sample.targetNumbers),
    });
    input.dispose(); prediction.dispose();
  }
  const diagnostics = evaluateReliabilityGate(observations, {
    confidenceLevel: params.confidenceLevel, minimumAdvantage: params.minimumAdvantage,
    minimumGroups: params.minimumGroups, bootstrapIterations: params.bootstrapIterations,
    seed: params.seed,
  });
  const calibratedWeights = { ...weights };
  if (diagnostics.selectedMethod === "neural") {
    weights = { neural: 1, frequency: 0, heuristic: 0, random: 0 };
  }
  return { weights, calibratedWeights, diagnostics };
}
