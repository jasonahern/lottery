import assert from "node:assert/strict";
import test from "node:test";

import { createRollingFolds, randomExpectedMatches } from "./backtest";

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
