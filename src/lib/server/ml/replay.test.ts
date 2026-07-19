import assert from "node:assert/strict";
import test from "node:test";

import { replayPolicyHistory } from "./replay";

test("replayPolicyHistory returns deterministic summary shape", () => {
  const summary = replayPolicyHistory([
    { signal: "keep", matchCount: 0, topKHit: false },
    { signal: "regularize", matchCount: 1, topKHit: true },
    { signal: "scale_up", matchCount: 2, topKHit: true },
  ]);

  assert.equal(summary.events, 3);
  assert.equal(summary.rewards.length, 3);
  assert.equal(summary.configsUsed.length, 3);
  assert.ok(Number.isFinite(summary.averageReward));
});
