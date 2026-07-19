import type { ParsedDraw } from "./data";

export const FREQUENCY_PRIOR_WEIGHT = 0.15;
export const LOTTERY_TICKET_SIZE = 6;

export function canonicalizeNumbers(numbers: number[]): number[] {
  return [...numbers].sort((a, b) => a - b);
}

export function buildNumberVocabulary(draws: ParsedDraw[]): number[] {
  const maxNumber = Math.max(
    0,
    ...draws.flatMap((draw) => draw.winningNumbers),
  );

  return Array.from({ length: maxNumber }, (_, idx) => idx + 1);
}

export function buildInputVector(
  window: ParsedDraw[],
  maxNumber: number,
): number[] {
  const vector: number[] = [];
  for (const draw of window) {
    for (const n of canonicalizeNumbers(draw.winningNumbers)) {
      vector.push(n / maxNumber);
    }
  }

  return vector;
}

export function buildMultiHotInputVector(
  window: ParsedDraw[],
  numbers: number[],
): number[] {
  const indexByNumber = new Map(numbers.map((number, index) => [number, index]));
  return window.flatMap((draw) => buildTargetMultiHot(draw.winningNumbers, indexByNumber));
}

export function buildTargetMultiHot(
  numbers: number[],
  indexByNumber: Map<number, number>,
): number[] {
  const output = new Array(indexByNumber.size).fill(0);
  for (const n of numbers) {
    const idx = indexByNumber.get(n);
    if (idx !== undefined) {
      output[idx] = 1;
    }
  }

  return output;
}

export function pickTopKNumbers(
  scores: number[],
  numbers: number[],
  k: number,
): number[] {
  return scores
    .map((score, idx) => ({ score, number: numbers[idx] }))
    .sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }

      return a.number - b.number;
    })
    .slice(0, k)
    .map((item) => item.number)
    .sort((a, b) => a - b);
}

export function countMatches(predicted: number[], actual: number[]): number {
  const actualSet = new Set(actual);
  return predicted.filter((n) => actualSet.has(n)).length;
}

export function buildRecentFrequencyScores(
  window: ParsedDraw[],
  numbers: number[],
): number[] {
  const counts = new Map<number, number>();
  let total = 0;

  for (const draw of window) {
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

export function pickFrequencyBaselineNumbers(
  window: ParsedDraw[],
  numbers: number[],
  k: number,
): number[] {
  return pickTopKNumbers(
    buildRecentFrequencyScores(window, numbers),
    numbers,
    k,
  );
}

export function pickSeededRandomNumbers(
  numbers: number[],
  k: number,
  seed: number,
): number[] {
  let state = Math.max(1, Math.floor(seed)) % 2147483647;
  const shuffled = [...numbers];

  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    state = (state * 48271) % 2147483647;
    const j = state % (i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled.slice(0, k).sort((a, b) => a - b);
}

export function blendScores(
  modelScores: number[],
  priorScores: number[],
  priorWeight = FREQUENCY_PRIOR_WEIGHT,
): number[] {
  const boundedPriorWeight = Math.max(0, Math.min(1, priorWeight));
  const modelWeight = 1 - boundedPriorWeight;

  return modelScores.map(
    (score, idx) =>
      score * modelWeight + (priorScores[idx] ?? 0) * boundedPriorWeight,
  );
}
