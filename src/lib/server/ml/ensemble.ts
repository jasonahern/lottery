export type EnsembleMethod = "neural" | "frequency" | "heuristic" | "random";

export type EnsembleWeights = Record<EnsembleMethod, number>;
export type EnsembleScores = Record<EnsembleMethod, number[]>;

export const DEFAULT_ENSEMBLE_WEIGHTS: EnsembleWeights = {
  neural: 0.45,
  frequency: 0.2,
  heuristic: 0.35,
  random: 0,
};

export const ENSEMBLE_LEARNING_RATE = 1;

export function normalizeScores(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const ranked = scores
    .map((score, index) => ({ score: Number.isFinite(score) ? score : 0, index }))
    .sort((a, b) => a.score - b.score || b.index - a.index);
  const normalized = new Array(scores.length).fill(0);
  const denominator = Math.max(1, scores.length - 1);
  let start = 0;
  while (start < ranked.length) {
    let end = start + 1;
    while (end < ranked.length && ranked[end].score === ranked[start].score) end += 1;
    const averageRank = (start + end - 1) / 2 / denominator;
    for (let rank = start; rank < end; rank += 1) {
      normalized[ranked[rank].index] = averageRank;
    }
    start = end;
  }
  return normalized;
}

export function normalizeWeights(weights: EnsembleWeights): EnsembleWeights {
  const safe = Object.fromEntries(
    Object.entries(weights).map(([method, value]) => [method, Math.max(0, value)]),
  ) as EnsembleWeights;
  const total = Object.values(safe).reduce((sum, value) => sum + value, 0);
  if (total <= 0) return { ...DEFAULT_ENSEMBLE_WEIGHTS };
  return Object.fromEntries(
    Object.entries(safe).map(([method, value]) => [method, value / total]),
  ) as EnsembleWeights;
}

export function blendEnsembleScores(
  componentScores: EnsembleScores,
  weights: EnsembleWeights,
): number[] {
  const normalizedWeights = normalizeWeights(weights);
  const normalized = Object.fromEntries(
    Object.entries(componentScores).map(([method, scores]) => [method, normalizeScores(scores)]),
  ) as EnsembleScores;
  const size = normalized.neural.length;
  return Array.from({ length: size }, (_, index) =>
    (Object.keys(normalizedWeights) as EnsembleMethod[]).reduce(
      (sum, method) => sum + normalizedWeights[method] * (normalized[method][index] ?? 0),
      0,
    ),
  );
}

/** Multiplicative-weights update. Rewards must be in [0, 1]. */
export function updateEnsembleWeights(
  weights: EnsembleWeights,
  rewards: Record<EnsembleMethod, number>,
  learningRate = ENSEMBLE_LEARNING_RATE,
): EnsembleWeights {
  const current = normalizeWeights(weights);
  const updated = Object.fromEntries(
    (Object.keys(current) as EnsembleMethod[]).map((method) => [
      method,
      current[method] * Math.exp(learningRate * Math.max(0, Math.min(1, rewards[method]))),
    ]),
  ) as EnsembleWeights;
  return normalizeWeights(updated);
}

export function buildSeededRandomScores(size: number, seed: number): number[] {
  let state = Math.max(1, Math.floor(seed)) % 2147483647;
  return Array.from({ length: size }, () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  });
}

export function calculateCalibrationReward(
  scores: number[],
  numbers: number[],
  actualNumbers: number[],
): number {
  if (actualNumbers.length === 0 || numbers.length === 0) return 0;
  const ranked = scores
    .map((score, index) => ({ score, number: numbers[index] }))
    .sort((a, b) => b.score - a.score || a.number - b.number);
  const rankByNumber = new Map(ranked.map((item, rank) => [item.number, rank]));
  const denominator = Math.max(1, numbers.length - 1);
  return actualNumbers.reduce((sum, number) => {
    const rank = rankByNumber.get(number) ?? numbers.length - 1;
    return sum + (numbers.length - 1 - rank) / denominator;
  }, 0) / actualNumbers.length;
}
