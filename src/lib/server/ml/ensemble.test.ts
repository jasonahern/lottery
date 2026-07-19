import assert from "node:assert/strict";
import test from "node:test";

import {
  blendEnsembleScores,
  buildSeededRandomScores,
  DEFAULT_ENSEMBLE_WEIGHTS,
  normalizeScores,
  updateEnsembleWeights,
  calculateCalibrationReward,
  getEffectiveEnsembleWeights,
  collapseDuplicateExpertWeights,
} from "./ensemble";

test("normalizeScores converts incomparable scales to ranks", () => {
  assert.deepEqual(normalizeScores([100, -2, 4]), [1, 0, 0.5]);
  assert.deepEqual(normalizeScores([0, 0, 1]), [0.25, 0.25, 1]);
});

test("blendEnsembleScores follows the dominant expert", () => {
  const scores = { neural: [0, 1], frequency: [1, 0], heuristic: [1, 0], random: [1, 0] };
  const blended = blendEnsembleScores(scores, { neural: 1, frequency: 0, heuristic: 0, random: 0 });
  assert.ok(blended[1] > blended[0]);
});

test("successful experts gain weight and weights remain normalized", () => {
  const updated = updateEnsembleWeights(DEFAULT_ENSEMBLE_WEIGHTS, {
    neural: 1,
    frequency: 0,
    heuristic: 0,
    random: 0,
  });
  assert.ok(updated.neural > DEFAULT_ENSEMBLE_WEIGHTS.neural);
  assert.ok(Math.abs(Object.values(updated).reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

test("random score vectors are reproducible", () => {
  assert.deepEqual(buildSeededRandomScores(5, 42), buildSeededRandomScores(5, 42));
});

test("calibration reward values the complete ranking of actual numbers", () => {
  assert.equal(calculateCalibrationReward([0.9, 0.5, 0.1], [1, 2, 3], [1]), 1);
  assert.equal(calculateCalibrationReward([0.9, 0.5, 0.1], [1, 2, 3], [3]), 0);
});

test("duplicate experts count as one capped signal", () => {
  const scores = {
    neural: [1, 0, 0],
    frequency: [0, 1, 0],
    heuristic: [0, 1, 0],
    random: [0, 0, 1],
  };
  const effective = getEffectiveEnsembleWeights(scores, {
    neural: 0.14,
    frequency: 0.31,
    heuristic: 0.55,
    random: 0,
  });
  assert.ok(Math.abs(effective.neural - 0.4) < 1e-12);
  assert.ok(Math.abs(effective.frequency + effective.heuristic - 0.6) < 1e-12);
  assert.equal(effective.random, 0);
});

test("duplicate experts are removed before ensemble learning", () => {
  const collapsed = collapseDuplicateExpertWeights([
    {
      neural: [1, 0, 0],
      frequency: [0, 1, 0],
      heuristic: [0, 1, 0],
      random: [0, 0, 1],
    },
    {
      neural: [0, 1, 0],
      frequency: [1, 0, 0],
      heuristic: [1, 0, 0],
      random: [0, 0, 1],
    },
  ], DEFAULT_ENSEMBLE_WEIGHTS);

  assert.equal(collapsed.heuristic, 0);
  assert.ok(Math.abs(collapsed.frequency - 0.55) < 1e-12);
  assert.ok(Math.abs(collapsed.neural - 0.45) < 1e-12);
});
