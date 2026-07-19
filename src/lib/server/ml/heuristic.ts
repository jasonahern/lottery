import type { ParsedDraw } from "./data";
import { pickTopKNumbers } from "./features";

const HORIZON_WEIGHTS = [
  { size: 10, weight: 0.3 },
  { size: 25, weight: 0.25 },
  { size: 50, weight: 0.18 },
  { size: 100, weight: 0.12 },
] as const;

const MACHINE_WEIGHT = 0.08;
const BALL_SET_WEIGHT = 0.07;

function frequencyScores(
  draws: ParsedDraw[],
  numbers: number[],
  predicate: (draw: ParsedDraw) => boolean = () => true,
): number[] {
  const counts = new Map<number, number>();
  let total = 0;

  for (const draw of draws) {
    if (!predicate(draw)) {
      continue;
    }

    for (const n of draw.winningNumbers) {
      counts.set(n, (counts.get(n) ?? 0) + 1);
      total += 1;
    }
  }

  if (total === 0) {
    return numbers.map(() => 0);
  }

  return numbers.map((n) => (counts.get(n) ?? 0) / total);
}

export function buildHeuristicScores(
  window: ParsedDraw[],
  numbers: number[],
  knownNextContext?: { machine?: string | null; ballSet?: number | null },
): number[] {
  const scores = numbers.map(() => 0);

  for (const horizon of HORIZON_WEIGHTS) {
    const recent = window.slice(-horizon.size);
    const horizonScores = frequencyScores(recent, numbers);
    horizonScores.forEach((score, idx) => {
      scores[idx] += score * horizon.weight;
    });
  }

  const machine = knownNextContext?.machine ?? null;
  if (machine) {
    const machineScores = frequencyScores(
      window,
      numbers,
      (draw) => draw.machine === machine,
    );
    machineScores.forEach((score, idx) => {
      scores[idx] += score * MACHINE_WEIGHT;
    });
  }

  const ballSet = knownNextContext?.ballSet ?? null;
  if (ballSet !== null) {
    const ballSetScores = frequencyScores(
      window,
      numbers,
      (draw) => draw.ballSet === ballSet,
    );
    ballSetScores.forEach((score, idx) => {
      scores[idx] += score * BALL_SET_WEIGHT;
    });
  }

  return scores;
}

export function pickHeuristicNumbers(
  window: ParsedDraw[],
  numbers: number[],
  k: number,
): number[] {
  return pickTopKNumbers(buildHeuristicScores(window, numbers), numbers, k);
}
