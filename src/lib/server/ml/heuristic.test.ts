import assert from "node:assert/strict";
import test from "node:test";

import type { ParsedDraw } from "./data";
import { pickHeuristicNumbers } from "./heuristic";

const draw = (
  id: number,
  winningNumbers: number[],
  machine: string | null = null,
  ballSet: number | null = null,
): ParsedDraw => ({
  id,
  drawNumber: id,
  drawDate: new Date(2026, 0, id),
  winningNumbers,
  bonusNumber: null,
  machine,
  ballSet,
  drawRound: 1,
});

test("pickHeuristicNumbers ranks recent and contextual numbers", () => {
  const picked = pickHeuristicNumbers(
    [
      draw(1, [1, 2, 3], "A", 1),
      draw(2, [1, 2, 4], "B", 2),
      draw(3, [1, 5, 6], "A", 1),
      draw(4, [7, 8, 9], "A", 1),
    ],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    3,
  );

  assert.equal(picked.length, 3);
  assert.ok(picked.includes(1));
});
