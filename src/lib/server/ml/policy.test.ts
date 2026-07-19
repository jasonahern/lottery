import assert from "node:assert/strict";
import test from "node:test";

import {
  choosePolicyDecision,
  computeRewardV1,
  defaultPolicyConfig,
} from "./policy";

test("computeRewardV1 returns expected values", () => {
  assert.equal(computeRewardV1(0, false), -0.5);
  assert.equal(computeRewardV1(1, true), 0.2);
  assert.equal(computeRewardV1(2, true), 1);
  assert.equal(computeRewardV1(4, true), 3);
});

test("choosePolicyDecision returns bounded config", () => {
  const decision = choosePolicyDecision({
    baseConfig: {
      windowSize: 999,
      epochs: 2000,
      batchSize: 5000,
      learningRate: 2,
      dropoutRate: 2,
    },
    signal: "regularize",
    randomValue: 0.99,
    context: {
      latestRunId: 1,
      signal: "regularize",
      latestTrainLoss: 0.2,
      latestValLoss: 0.4,
      latestHoldoutScore: 1,
    },
  });

  assert.ok(
    decision.config.windowSize >= 1 && decision.config.windowSize <= 52,
  );
  assert.ok(decision.config.epochs >= 1 && decision.config.epochs <= 500);
  assert.ok(
    decision.config.batchSize >= 1 && decision.config.batchSize <= 1024,
  );
  assert.ok(
    decision.config.learningRate >= 0.000001 &&
      decision.config.learningRate <= 0.1,
  );
  assert.ok(
    decision.config.dropoutRate >= 0 && decision.config.dropoutRate <= 0.99,
  );
});

test("defaultPolicyConfig mirrors training defaults", () => {
  const base = defaultPolicyConfig();
  assert.equal(base.windowSize, 10);
  assert.equal(base.epochs, 80);
  assert.equal(base.batchSize, 32);
  assert.equal(base.learningRate, 0.001);
  assert.equal(base.dropoutRate, 0.25);
});
