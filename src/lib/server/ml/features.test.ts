import assert from "node:assert/strict";
import test from "node:test";

import {
  blendScores,
  buildInputVector,
  buildMultiHotInputVector,
  buildNumberVocabulary,
  buildRecentFrequencyScores,
  canonicalizeNumbers,
  pickSeededRandomNumbers,
  pickTopKNumbers,
  LOTTERY_TICKET_SIZE,
} from "./features";
import type { ParsedDraw } from "./data";

const draw = (
  id: number,
  winningNumbers: number[],
  drawDate = new Date(2026, 0, id),
): ParsedDraw => ({
  id,
  drawNumber: id,
  drawDate,
  winningNumbers,
  bonusNumber: null,
  machine: null,
  ballSet: null,
  drawRound: 1,
});

test("canonicalizeNumbers sorts lottery sets for stable modeling", () => {
  assert.deepEqual(canonicalizeNumbers([45, 9, 22, 36, 13, 40]), [
    9, 13, 22, 36, 40, 45,
  ]);
});

test("buildInputVector ignores draw-ball order within each draw", () => {
  assert.deepEqual(buildInputVector([draw(1, [3, 1, 2])], 3), [1 / 3, 2 / 3, 1]);
});

test("buildMultiHotInputVector preserves draw chronology and number identity", () => {
  const vocabulary = [1, 2, 3, 4];
  const vector = buildMultiHotInputVector(
    [draw(1, [1, 3]), draw(2, [2, 4])],
    vocabulary,
  );
  assert.deepEqual(vector, [1, 0, 1, 0, 0, 1, 0, 1]);
});

test("buildNumberVocabulary includes every possible number up to observed max", () => {
  assert.deepEqual(buildNumberVocabulary([draw(1, [1, 5])]), [1, 2, 3, 4, 5]);
});

test("pickTopKNumbers uses lower numbers as deterministic tie-breakers", () => {
  assert.deepEqual(pickTopKNumbers([0.5, 0.5, 0.1], [2, 1, 3], 2), [1, 2]);
});

test("ticket size remains six regardless of multi-round result size", () => {
  assert.equal(LOTTERY_TICKET_SIZE, 6);
  assert.equal(pickTopKNumbers(Array(12).fill(1), Array.from({ length: 12 }, (_, i) => i + 1), LOTTERY_TICKET_SIZE).length, 6);
});

test("blendScores adds a bounded recent-frequency prior", () => {
  const prior = buildRecentFrequencyScores(
    [draw(1, [1, 1, 2]), draw(2, [2, 3])],
    [1, 2, 3],
  );

  assert.deepEqual(prior, [0.4, 0.4, 0.2]);
  assert.deepEqual(blendScores([1, 0, 0], prior, 0.5), [0.7, 0.2, 0.1]);
});

test("pickSeededRandomNumbers is deterministic and sorted", () => {
  assert.deepEqual(
    pickSeededRandomNumbers([1, 2, 3, 4, 5, 6], 3, 42),
    pickSeededRandomNumbers([1, 2, 3, 4, 5, 6], 3, 42),
  );
});
