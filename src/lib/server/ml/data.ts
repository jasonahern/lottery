import { asc, inArray } from "drizzle-orm";

import { db } from "$lib/server/db";
import {
  lotteryDrawBalls,
  lotteryDraws,
  lotteryEntries,
} from "$lib/server/db/schema";

type LotteryRow = {
  id: number;
  numbers: string;
  createdAt: Date | null;
};

export type NormalizedDrawInput = {
  draw: {
    legacyEntryId?: number;
    drawNumber: number;
    drawDate: Date;
    dayName: string | null;
    gameName: string | null;
    machine: string | null;
    wins: number | null;
    ballSet: number | null;
    drawSequence: number | null;
    drawRound: number;
    jackpotAmount: number | null;
    sourceRow: string;
  };
  balls: Array<{
    position: number;
    value: number;
    isBonus: boolean;
  }>;
};

export type ParsedDraw = {
  id: number;
  drawNumber: number;
  drawDate: Date;
  winningNumbers: number[];
  bonusNumber: number | null;
  machine: string | null;
  ballSet: number | null;
  drawRound: number;
};

export type TemporalSplit = {
  train: ParsedDraw[];
  test: ParsedDraw[];
  cutoffDate: Date;
};

const NUMBER_PATTERN = /^\d+$/;

function parseMonth(monthToken: string): number {
  const monthMap: Record<string, number> = {
    jan: 0,
    feb: 1,
    mar: 2,
    apr: 3,
    may: 4,
    jun: 5,
    jul: 6,
    aug: 7,
    sep: 8,
    oct: 9,
    nov: 10,
    dec: 11,
  };

  const month = monthMap[monthToken.trim().toLowerCase()];
  if (month === undefined) {
    throw new Error(`Unknown month token: ${monthToken}`);
  }

  return month;
}

function toNumber(token: string, label: string): number {
  const trimmed = token.trim();
  if (!NUMBER_PATTERN.test(trimmed)) {
    throw new Error(`Invalid ${label} value: ${token}`);
  }

  return Number(trimmed);
}

function toNullableNumber(token: string | undefined): number | null {
  if (!token) {
    return null;
  }

  const trimmed = token.trim();
  return NUMBER_PATTERN.test(trimmed) ? Number(trimmed) : null;
}

function normalizeText(token: string | undefined): string | null {
  if (!token) {
    return null;
  }

  const trimmed = token.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function parseLotteryRow(row: LotteryRow): ParsedDraw {
  const parts = row.numbers.split(",").map((p) => p.trim());

  // Expected format:
  // drawNo, dayName, day, month, year, n1, n2, n3, n4, n5, n6, bonus, set, machine
  if (parts.length < 12) {
    throw new Error(`Unexpected row format for entry ${row.id}`);
  }

  const drawNumber = toNumber(parts[0], "draw number");
  const day = toNumber(parts[2], "day");
  const month = parseMonth(parts[3]);
  const year = toNumber(parts[4], "year");

  const winningNumbers = parts
    .slice(5, 11)
    .map((token, idx) => toNumber(token, `winning number ${idx + 1}`));
  const bonusToken = parts[11];
  const bonusNumber = NUMBER_PATTERN.test(bonusToken)
    ? Number(bonusToken)
    : null;

  return {
    id: row.id,
    drawNumber,
    drawDate: new Date(year, month, day),
    winningNumbers,
    bonusNumber,
    machine: normalizeText(parts[14]),
    ballSet: toNullableNumber(parts[15]),
    drawRound: 1,
  };
}

export function parseCsvDataRowToNormalized(
  rowText: string,
  legacyEntryId?: number,
  drawRound = 1,
): NormalizedDrawInput {
  const parts = rowText.split(",").map((p) => p.trim());
  if (parts.length < 12) {
    throw new Error("CSV row does not have required draw columns.");
  }

  const drawNumber = toNumber(parts[0], "draw number");
  const dayName = normalizeText(parts[1]);
  const day = toNumber(parts[2], "day");
  const month = parseMonth(parts[3]);
  const year = toNumber(parts[4], "year");

  const mainNumbers = parts
    .slice(5, 11)
    .map((token, idx) => toNumber(token, `winning number ${idx + 1}`));
  const bonusNumber = toNullableNumber(parts[11]);

  const jackpotAmount = toNullableNumber(parts[12]);
  const wins = toNullableNumber(parts[13]);
  const machine = normalizeText(parts[14]);
  const ballSet = toNullableNumber(parts[15]);
  const gameName = "UK Lotto";

  const balls = mainNumbers.map((value, idx) => ({
    position: idx + 1,
    value,
    isBonus: false,
  }));

  if (bonusNumber !== null) {
    balls.push({
      position: balls.length + 1,
      value: bonusNumber,
      isBonus: true,
    });
  }

  return {
    draw: {
      legacyEntryId,
      drawNumber,
      drawDate: new Date(year, month, day),
      dayName,
      gameName,
      machine,
      wins,
      ballSet,
      drawSequence: ballSet,
      drawRound,
      jackpotAmount,
      sourceRow: rowText,
    },
    balls,
  };
}

export function parseCsvRowsWithRounds(rows: string[]): Array<{
  row: string;
  parsed: NormalizedDrawInput;
}> {
  const seenSourceRows = new Set<string>();
  const roundByDraw = new Map<number, number>();
  const result: Array<{ row: string; parsed: NormalizedDrawInput }> = [];
  for (const row of rows) {
    if (seenSourceRows.has(row)) continue;
    seenSourceRows.add(row);
    const drawNumber = Number(row.split(",", 1)[0]);
    const drawRound = (roundByDraw.get(drawNumber) ?? 0) + 1;
    roundByDraw.set(drawNumber, drawRound);
    if (drawRound > 2) {
      throw new Error(`Draw ${drawNumber} contains more than two rounds.`);
    }
    result.push({ row, parsed: parseCsvDataRowToNormalized(row, undefined, drawRound) });
  }
  return result;
}

export function loadParsedDrawsFromNormalized(): ParsedDraw[] {
  const draws = db
    .select({
      id: lotteryDraws.id,
      drawNumber: lotteryDraws.drawNumber,
      drawDate: lotteryDraws.drawDate,
      machine: lotteryDraws.machine,
      ballSet: lotteryDraws.ballSet,
      drawRound: lotteryDraws.drawRound,
    })
    .from(lotteryDraws)
    .orderBy(asc(lotteryDraws.drawDate), asc(lotteryDraws.id))
    .all();

  if (draws.length === 0) {
    return [];
  }

  const drawIds = draws.map((d) => d.id);
  const balls = db
    .select({
      drawId: lotteryDrawBalls.drawId,
      position: lotteryDrawBalls.position,
      value: lotteryDrawBalls.value,
      isBonus: lotteryDrawBalls.isBonus,
    })
    .from(lotteryDrawBalls)
    .where(inArray(lotteryDrawBalls.drawId, drawIds))
    .orderBy(asc(lotteryDrawBalls.drawId), asc(lotteryDrawBalls.position))
    .all();

  const ballsByDraw = new Map<number, typeof balls>();
  for (const ball of balls) {
    const arr = ballsByDraw.get(ball.drawId) ?? [];
    arr.push(ball);
    ballsByDraw.set(ball.drawId, arr);
  }

  return draws.map((draw) => {
    const drawBalls = ballsByDraw.get(draw.id) ?? [];
    const main = drawBalls.filter((b) => !b.isBonus).map((b) => b.value);
    const bonus = drawBalls.find((b) => b.isBonus)?.value ?? null;

    return {
      id: draw.id,
      drawNumber: draw.drawNumber,
      drawDate: draw.drawDate,
      winningNumbers: main,
      bonusNumber: bonus,
      machine: draw.machine,
      ballSet: draw.ballSet,
      drawRound: draw.drawRound,
    };
  });
}

export function loadParsedDrawsFromLegacy(): ParsedDraw[] {
  const rows = db
    .select({
      id: lotteryEntries.id,
      numbers: lotteryEntries.numbers,
      createdAt: lotteryEntries.createdAt,
    })
    .from(lotteryEntries)
    .orderBy(asc(lotteryEntries.id))
    .all();

  return rows.map((row) => parseLotteryRow(row));
}

export function backfillNormalizedFromLegacyIfNeeded(): number {
  const existing = db
    .select({ id: lotteryDraws.id })
    .from(lotteryDraws)
    .limit(1)
    .get();
  if (existing) {
    return 0;
  }

  const legacyRows = db
    .select({
      id: lotteryEntries.id,
      numbers: lotteryEntries.numbers,
    })
    .from(lotteryEntries)
    .orderBy(asc(lotteryEntries.id))
    .all();

  if (legacyRows.length === 0) {
    return 0;
  }

  db.transaction((tx) => {
    const roundByDraw = new Map<number, number>();
    for (const legacy of legacyRows) {
      const drawNumber = Number(legacy.numbers.split(",", 1)[0]);
      const drawRound = (roundByDraw.get(drawNumber) ?? 0) + 1;
      roundByDraw.set(drawNumber, drawRound);
      const parsed = parseCsvDataRowToNormalized(legacy.numbers, legacy.id, drawRound);
      const insertedDraw = tx
        .insert(lotteryDraws)
        .values({
          legacyEntryId: legacy.id,
          drawNumber: parsed.draw.drawNumber,
          drawDate: parsed.draw.drawDate,
          dayName: parsed.draw.dayName,
          gameName: parsed.draw.gameName,
          machine: parsed.draw.machine,
          wins: parsed.draw.wins,
          ballSet: parsed.draw.ballSet,
          drawSequence: parsed.draw.drawSequence,
          drawRound: parsed.draw.drawRound,
          jackpotAmount: parsed.draw.jackpotAmount,
          sourceRow: parsed.draw.sourceRow,
        })
        .returning({ id: lotteryDraws.id })
        .get();

      if (!insertedDraw) {
        throw new Error("Failed to backfill normalized draw row.");
      }

      tx.insert(lotteryDrawBalls)
        .values(
          parsed.balls.map((ball) => ({
            drawId: insertedDraw.id,
            position: ball.position,
            value: ball.value,
            isBonus: ball.isBonus,
          })),
        )
        .run();
    }
  });

  return legacyRows.length;
}

export function loadParsedDraws(): ParsedDraw[] {
  const normalized = loadParsedDrawsFromNormalized();
  if (normalized.length > 0) {
    return normalized;
  }

  backfillNormalizedFromLegacyIfNeeded();
  const normalizedAfterBackfill = loadParsedDrawsFromNormalized();
  if (normalizedAfterBackfill.length > 0) {
    return normalizedAfterBackfill;
  }

  return loadParsedDrawsFromLegacy();
}

export function assertDrawIntegrity(draws: ParsedDraw[]): void {
  const identities = new Map<string, ParsedDraw>();
  for (const draw of draws) {
    if (draw.winningNumbers.length !== 6 || new Set(draw.winningNumbers).size !== 6) {
      throw new Error(`Draw ${draw.drawNumber} must contain six unique main numbers.`);
    }
    if (draw.winningNumbers.some((number) => !Number.isInteger(number) || number < 1 || number > 59)) {
      throw new Error(`Draw ${draw.drawNumber} contains a number outside 1-59.`);
    }
    const key = `${draw.drawNumber}:${draw.drawRound}:${draw.drawDate.getTime()}`;
    const previous = identities.get(key);
    if (previous) {
      throw new Error(`Duplicate draw identity detected for draw ${draw.drawNumber}.`);
    }
    identities.set(key, draw);
  }
}

export function splitByHoldoutWeeks(
  draws: ParsedDraw[],
  holdoutWeeks: number,
): TemporalSplit {
  if (holdoutWeeks <= 0) {
    throw new Error("holdoutWeeks must be > 0");
  }

  const sorted = [...draws].sort(
    (a, b) => a.drawDate.getTime() - b.drawDate.getTime(),
  );
  if (sorted.length === 0) {
    throw new Error("No draws available to split.");
  }

  const latestDate = sorted[sorted.length - 1].drawDate;
  const cutoffMs =
    latestDate.getTime() - holdoutWeeks * 7 * 24 * 60 * 60 * 1000;
  const cutoffDate = new Date(cutoffMs);

  const train = sorted.filter((draw) => draw.drawDate.getTime() <= cutoffMs);
  const test = sorted.filter((draw) => draw.drawDate.getTime() > cutoffMs);

  if (train.length === 0) {
    throw new Error("Training split is empty. Reduce holdoutWeeks.");
  }

  if (test.length === 0) {
    throw new Error(
      "Test split is empty. Increase holdoutWeeks or verify draw dates.",
    );
  }

  return {
    train,
    test,
    cutoffDate,
  };
}

export type WindowedSample = {
  inputDrawIds: number[];
  targetDrawId: number;
  inputWindow: ParsedDraw[];
  targetDraw: ParsedDraw;
};

export function createWindowedSamples(
  draws: ParsedDraw[],
  windowSize: number,
): WindowedSample[] {
  if (windowSize <= 0) {
    throw new Error("windowSize must be > 0");
  }

  const sorted = [...draws].sort((a, b) => {
    const dateDelta = a.drawDate.getTime() - b.drawDate.getTime();
    if (dateDelta !== 0) {
      return dateDelta;
    }

    return a.drawNumber - b.drawNumber || a.id - b.id;
  });

  const samples: WindowedSample[] = [];
  const rounds = new Set(sorted.map((draw) => draw.drawRound));
  for (const round of rounds) {
    const roundDraws = sorted.filter((draw) => draw.drawRound === round);
    for (let i = windowSize; i < roundDraws.length; i += 1) {
      const inputWindow = roundDraws.slice(i - windowSize, i);
      const targetDraw = roundDraws[i];

      samples.push({
        inputDrawIds: inputWindow.map((d) => d.id),
        targetDrawId: targetDraw.id,
        inputWindow,
        targetDraw,
      });
    }
  }
  return samples.sort(
    (a, b) =>
      a.targetDraw.drawDate.getTime() - b.targetDraw.drawDate.getTime() ||
      a.targetDraw.drawRound - b.targetDraw.drawRound,
  );
}

export function assertNoLeakage(
  trainSamples: WindowedSample[],
  holdoutDrawIds: Set<number>,
): void {
  const leakingSample = trainSamples.find((sample) => {
    if (holdoutDrawIds.has(sample.targetDrawId)) {
      return true;
    }

    return sample.inputDrawIds.some((drawId) => holdoutDrawIds.has(drawId));
  });

  if (leakingSample) {
    throw new Error(
      `Detected data leakage in sample targeting draw ${leakingSample.targetDrawId}.`,
    );
  }

  const identityLeak = trainSamples.find((sample) =>
    sample.inputWindow.some(
      (draw) =>
        (draw.drawNumber === sample.targetDraw.drawNumber &&
          draw.drawRound === sample.targetDraw.drawRound) ||
        draw.drawDate.getTime() >= sample.targetDraw.drawDate.getTime() ||
        canonicalDrawKey(draw) === canonicalDrawKey(sample.targetDraw),
    ),
  );
  if (identityLeak) {
    throw new Error(`Detected chronological or duplicate leakage targeting draw ${identityLeak.targetDrawId}.`);
  }
}

function canonicalDrawKey(draw: ParsedDraw): string {
  return [...draw.winningNumbers].sort((a, b) => a - b).join(",");
}
