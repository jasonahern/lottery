import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_TRAINING_CONFIG,
  parseHiddenLayersInput,
  parseTrainingSeedsInput,
  resolveHiddenLayers,
  validateTrainingConfig,
} from "./config";

test("parseTrainingSeedsInput normalizes unique positive integer seeds", () => {
  assert.deepEqual(parseTrainingSeedsInput("42, 137, 42, 2026"), [42, 137, 2026]);
  assert.deepEqual(parseTrainingSeedsInput("bad, -1, 7.5, 176"), [176]);
});

test("parseHiddenLayersInput parses and normalizes values", () => {
  const parsed = parseHiddenLayersInput("128, 64, 32");
  assert.deepEqual(parsed, [128, 64, 32]);
});

test("resolveHiddenLayers prefers custom layers", () => {
  const result = resolveHiddenLayers("small", [300, 150]);
  assert.deepEqual(result, [300, 150]);
});

test("validateTrainingConfig catches invalid layer count and size", () => {
  const errors = validateTrainingConfig({
    ...DEFAULT_TRAINING_CONFIG,
    hiddenLayers: [2048, 128, 64, 32, 16],
  });

  assert.ok(errors.some((error) => error.includes("at most")));
  assert.ok(errors.some((error) => error.includes("<= 1024")));
});
