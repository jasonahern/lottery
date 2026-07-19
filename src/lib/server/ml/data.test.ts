import test from "node:test";
import assert from "node:assert/strict";

import {
  assertNoLeakage,
  assertDrawIntegrity,
  createWindowedSamples,
  parseCsvDataRowToNormalized,
  parseLotteryRow,
  parseCsvRowsWithRounds,
  splitByHoldoutWeeks,
} from "./data";

test("parseLotteryRow parses known CSV row format", () => {
  const parsed = parseLotteryRow({
    id: 1,
    createdAt: null,
    numbers: "3188,Sat,11,Jul,2026,2,4,11,19,25,39,8,6,Z",
  });

  assert.equal(parsed.drawNumber, 3188);
  assert.equal(parsed.winningNumbers.length, 6);
  assert.equal(parsed.bonusNumber, 8);
});

test("splitByHoldoutWeeks keeps last draws in test partition", () => {
  const draws = [
    parseLotteryRow({
      id: 1,
      createdAt: null,
      numbers: "3001,Sat,01,Jun,2024,1,2,3,4,5,6,7,1,Z",
    }),
    parseLotteryRow({
      id: 2,
      createdAt: null,
      numbers: "3002,Wed,08,Jun,2024,2,3,4,5,6,7,8,1,Z",
    }),
    parseLotteryRow({
      id: 3,
      createdAt: null,
      numbers: "3003,Sat,15,Jun,2024,3,4,5,6,7,8,9,1,Z",
    }),
    parseLotteryRow({
      id: 4,
      createdAt: null,
      numbers: "3004,Wed,22,Jun,2024,4,5,6,7,8,9,10,1,Z",
    }),
  ];

  const split = splitByHoldoutWeeks(draws, 1);
  assert.ok(split.train.length > 0);
  assert.ok(split.test.length > 0);

  const maxTrain = Math.max(...split.train.map((d) => d.drawDate.getTime()));
  const minTest = Math.min(...split.test.map((d) => d.drawDate.getTime()));
  assert.ok(maxTrain <= minTest);
});

test("assertNoLeakage throws when holdout draw leaks into train samples", () => {
  const draws = [
    parseLotteryRow({
      id: 1,
      createdAt: null,
      numbers: "3001,Sat,01,Jun,2024,1,2,3,4,5,6,7,1,Z",
    }),
    parseLotteryRow({
      id: 2,
      createdAt: null,
      numbers: "3002,Wed,08,Jun,2024,2,3,4,5,6,7,8,1,Z",
    }),
    parseLotteryRow({
      id: 3,
      createdAt: null,
      numbers: "3003,Sat,15,Jun,2024,3,4,5,6,7,8,9,1,Z",
    }),
  ];

  const samples = createWindowedSamples(draws, 2);
  assert.equal(samples.length, 1);

  assert.throws(() => {
    assertNoLeakage(samples, new Set([3]));
  });
});

test("assertDrawIntegrity rejects duplicate draw identities", () => {
  const first = parseLotteryRow({ id: 1, createdAt: null, numbers: "3001,Sat,01,Jun,2024,1,2,3,4,5,6,7,1,Z" });
  const duplicate = { ...first, id: 2 };
  assert.throws(() => assertDrawIntegrity([first, duplicate]), /Duplicate draw identity/);
});

test("parseCsvDataRowToNormalized maps draw metadata and balls", () => {
  const normalized = parseCsvDataRowToNormalized(
    "3188,Sat,11,Jul,2026,45,9,22,36,13,40,41,6643227,0,Lotto 6,1",
    6301,
  );

  assert.equal(normalized.draw.legacyEntryId, 6301);
  assert.equal(normalized.draw.drawNumber, 3188);
  assert.equal(normalized.draw.dayName, "Sat");
  assert.equal(normalized.draw.gameName, "UK Lotto");
  assert.equal(normalized.draw.machine, "Lotto 6");
  assert.equal(normalized.draw.wins, 0);
  assert.equal(normalized.draw.ballSet, 1);
  assert.equal(normalized.draw.drawSequence, 1);
  assert.equal(normalized.draw.jackpotAmount, 6643227);

  assert.equal(normalized.balls.length, 7);
  assert.deepEqual(
    normalized.balls.filter((b) => !b.isBonus).map((b) => b.value),
    [45, 9, 22, 36, 13, 40],
  );
  assert.equal(normalized.balls.find((b) => b.isBonus)?.value, 41);
});

test("parseCsvRowsWithRounds assigns two same-number rows to separate rounds", () => {
  const rows = parseCsvRowsWithRounds([
    "3190,Sat,18,Jul,2026,1,2,3,4,5,6,7,100,0,Lotto 4,3",
    "3190,Sat,18,Jul,2026,8,9,10,11,12,13,14,100,0,Lotto 5,4",
  ]);
  assert.deepEqual(rows.map((row) => row.parsed.draw.drawRound), [1, 2]);
});

test("createWindowedSamples never mixes rounds", () => {
  const draws = Array.from({ length: 6 }, (_, index) =>
    [1, 2].map((drawRound) => ({
      id: index * 2 + drawRound,
      drawNumber: 4000 + index,
      drawDate: new Date(2026, 5, index + 1),
      winningNumbers: [1, 2, 3, 4, 5, 6],
      bonusNumber: 7,
      machine: `Machine ${drawRound}`,
      ballSet: drawRound,
      drawRound,
    })),
  ).flat();
  const samples = createWindowedSamples(draws, 3);
  assert.ok(samples.length > 0);
  assert.ok(samples.every((sample) => sample.inputWindow.every(
    (draw) => draw.drawRound === sample.targetDraw.drawRound,
  )));
});
