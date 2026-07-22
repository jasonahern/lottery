import assert from "node:assert/strict";
import test from "node:test";

import { createRollingEventFolds, createRollingFolds, evaluateReliabilityGate, randomExpectedMatches } from "./backtest";

test("createRollingFolds keeps chronological partitions disjoint", () => {
  const folds = createRollingFolds(Array.from({ length: 30 }, (_, i) => i), {
    minimumTrain: 10,
    calibrationSize: 4,
    holdoutSize: 4,
    folds: 3,
  });
  assert.equal(folds.length, 3);
  for (const fold of folds) {
    assert.ok(Math.max(...fold.train) < Math.min(...fold.calibration));
    assert.ok(Math.max(...fold.calibration) < Math.min(...fold.holdout));
  }
});

test("random expectation matches hypergeometric mean", () => {
  assert.equal(Number(randomExpectedMatches(6, 6, 59).toFixed(4)), 0.6102);
});

test("confidence gate rejects an uncertain ensemble advantage", () => {
  const diagnostics = evaluateReliabilityGate(
    Array.from({ length: 40 }, (_, index) => ({
      groupKey: String(index),
      neuralMatches: index % 2,
      ensembleMatches: index % 3 === 0 ? 1 : 0,
    })),
    { confidenceLevel: 0.95, minimumAdvantage: 0.05, minimumGroups: 30, bootstrapIterations: 500, seed: 42 },
  );
  assert.equal(diagnostics.selectedMethod, "neural");
  assert.notEqual(diagnostics.reason, "ensemble_selected");
});

test("confidence gate accepts a consistent ensemble advantage", () => {
  const diagnostics = evaluateReliabilityGate(
    Array.from({ length: 40 }, (_, index) => ({
      groupKey: String(index), neuralMatches: 0, ensembleMatches: 1,
    })),
    { confidenceLevel: 0.95, minimumAdvantage: 0.05, minimumGroups: 30, bootstrapIterations: 500, seed: 42 },
  );
  assert.equal(diagnostics.selectedMethod, "ensemble");
  assert.ok(diagnostics.confidenceLow > 0);
});

test("rolling event folds keep two rounds together", () => {
  const rows = Array.from({ length: 20 }, (_, date) => [
    { date, round: 1 }, { date, round: 2 },
  ]).flat();
  const folds = createRollingEventFolds(rows, (row) => String(row.date), {
    minimumTrainGroups: 8, calibrationGroups: 4, holdoutGroups: 4, folds: 2,
  });
  for (const fold of folds) {
    for (const partition of [fold.train, fold.calibration, fold.holdout]) {
      const counts = new Map<number, number>();
      partition.forEach((row) => counts.set(row.date, (counts.get(row.date) ?? 0) + 1));
      assert.ok([...counts.values()].every((count) => count === 2));
    }
  }
});
